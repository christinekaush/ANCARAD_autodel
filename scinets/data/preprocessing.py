"""

"""


__author__ = "Yngve Mardal Moe"
__email__ = "yngve.m.moe@gmail.com"


import numpy as np
from .._backend_utils import SubclassRegister
from abc import ABC, abstractmethod
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


preprocessor_register = SubclassRegister("Preprocessor")


def get_preprocessor(preprocessor):
    return preprocessor_register.get_item(preprocessor)


@preprocessor_register.link_base
class BasePreprocessor(ABC):
    @abstractmethod
    def __call__(self, images, targets):
        """The function being applied to the input images.
        """
        pass

    @abstractmethod
    def output_channels(self, input_channels):
        """The number of output channels as a function of input channels.
        """
        pass

    @abstractmethod
    def output_targets(self, input_targets):
        """The number of output channels as a function of input channels.
        """
        pass

class Preprocessor(BasePreprocessor):
    """Superclass for all preprocessors. Does nothing.
    """

    def __call__(self, images, targets):
        """The function being applied to the input images.
        """
        return images, targets

    def output_channels(self, input_channels):
        """The number of output channels as a function of input channels.
        """
        return input_channels

    def output_targets(self, input_targets):
        """The number of output channels as a function of input channels.
        """
        return input_targets


class PreprocessingPipeline(Preprocessor):
    """Create a preprocessing pipeline form a list of preprocessors.

    The output of the first preprocessor is used as argument for the second,
    and so forth. The output of the last preprocessor is then returned.
    """

    def __init__(self, preprocessor_dicts):
        def get_operator(preprocessor_dict):
            Preprocessor = get_preprocessor(preprocessor_dict["operator"])
            return Preprocessor(**preprocessor_dict["arguments"])

        self.preprocessors = [
            get_operator(preprocessor_dict) for preprocessor_dict in preprocessor_dicts
        ]

    def __call__(self, images, targets):
        for preprocessor in self.preprocessors:
            images, targets = preprocessor(images, targets)
        return images, targets

    def output_channels(self, input_channels):
        output_channels = input_channels
        for preprocessor in self.preprocessors:
            output_channels = preprocessor.output_channels(output_channels)
        return output_channels

    def output_targets(self, input_targets):
        output_targets = input_targets
        for preprocessor in self.preprocessors:
            output_targets = preprocessor.output_targets(output_targets)
        return output_targets


class ChannelRemoverPreprocessor(Preprocessor):
    """Used to remove a single channel from the inputs.
    """

    def __init__(self, channel):
        self.unwanted_channel = channel

    def __call__(self, images, targets):
        return np.delete(images, self.unwanted_channel, axis=-1), targets

    def output_channels(self, input_channels):
        return input_channels - 1

class ChannelKeeperPreprocessor(Preprocessor):
    """Used to keep channels from the inputs.
    """

    def __init__(self, channels):
        channels = channels if type(channels) == list else [channels]
        self.wanted_channels = channels
    
    def __call__(self, images, targets):
        all_channels = list(range(images.shape[-1]))
        unwanted_channels = np.delete(all_channels, self.wanted_channels)
        return np.delete(images, unwanted_channels, axis=-1), targets
    
    def output_channels(self, input_channels):
        return len(self.wanted_channels)



class WindowingPreprocessor(Preprocessor):
    """Used to set the dynamic range of an image.
    """

    def __init__(self, window_center, window_width, channel):
        self.window_center, self.window_width = window_center, window_width
        self.channel = channel

    def perform_windowing(self, image):
        image = image - self.window_center
        image[image < -self.window_width / 2] = -self.window_width / 2
        image[image > self.window_width / 2] = self.window_width / 2
        return image

    def __call__(self, images, targets):
        images = images.copy()
        images[..., self.channel] = self.perform_windowing(images[..., self.channel])
        return images, targets


class MultipleWindowsPreprocessor(WindowingPreprocessor):
    """Used to create multiple windows of the same channel.
    """

    def __init__(self, window_centers, window_widths, channel):
        self.window_centers = window_centers
        self.window_widths = window_widths
        self.channel = channel

    def generate_all_windows(self, images):
        channel = images[..., self.channel]
        new_channels = []
        for window_center, window_width in zip(self.window_centers, self.window_widths):
            self.window_center, self.window_width = window_center, window_width
            new_channel = self.perform_windowing(channel)
            new_channels.append(new_channel)

        return np.stack(new_channels, axis=-1)

    def __call__(self, images, targets):
        new_channels = self.generate_all_windows(images)

        # Replace current CT channel with all windowed versions
        images = np.delete(images, self.channel, axis=-1)
        images = np.concatenate((images, new_channels), axis=-1)
        return images, targets

    def output_channels(self, input_channels):
        return input_channels + len(self.window_widths) - 1


class HounsfieldWindowingPreprocessor(WindowingPreprocessor):
    """A windowing operator, with the option to set the Hounsfield unit offset.

    The Hounsfield unit offset is simply added to the window center,
    but this makes the window centers on the same scale as what radiologists
    use.
    """

    def __init__(self, window_center, window_width, channel=0, hounsfield_offset=1024, hounsfield=False):
        if hounsfield:
            window_center += hounsfield_offset
        super().__init__(window_center, window_width, channel)


class MultipleHounsfieldWindowsPreprocessor(MultipleWindowsPreprocessor):
    """Perform several windows of the CT channel with a Hounsfield unit offset.
    
    The Hounsfield unit offset is simply added to the window center,
    but this makes the window centers on the same scale as what radiologists
    use.
    """

    def __init__(
        self, window_centers, window_widths, channel=0, hounsfield_offset=1024
    ):
        window_centers = [wc + hounsfield_offset for wc in window_centers]
        super().__init__(window_centers, window_widths, channel)

class ElasticDeformPreprocesser(Preprocessor):
    """
    """

    def __init__(self, alpha, sigma, alpha_affine):
        """
        Parameters:
        ----------
        channels : int

        alpha : int
            scaling factor. Controls the intensity of the deformation.
            If alpha > threshold, the displacement become close to affine.
            If alpha is very large (>> threshold) the displacement become translations.

        sigma : float
            standard deviation for the filter, given in voxels. Elasticity coefficient.

        alpha_affine : float
            distorting the image grid. 

        """
        self.alpha = 112 if alpha is None else alpha
        self.sigma = 25 if sigma is None else sigma
        self.alpha_affine = 33 if alpha_affine is None else alpha_affine

    def elastic_transform(self, image, random_state=None):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
         https://www.microsoft.com/en-us/research/wp-content/uploads/2003/08/icdar03.pdf 

        Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
        Borrowed from: https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation
        """
        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape
        shape_size = shape[:2]

        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
        pts2 = pts1 + random_state.uniform(-self.alpha_affine, self.alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

        return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

    def perform_deformations(self, images):
        """
        """ 

        return self.elastic_transform(images).astype(np.int32, copy=False)

    def __call__(self, images, targets):
        imgs = images.copy()
        imgs = imgs.astype(np.float32, copy=False)
        imgs_tar = cv2.merge((imgs, targets.astype(np.float32, copy=False)))

        transformed = self.elastic_transform(imgs_tar).astype(np.int32, copy=False)
        deformed_imgs = transformed[:,:,:-1]
        deformed_tars = transformed[:,:,-1]

        ## add the deformed images, resulting in a dataset containing twice as many image slices
        #all_images = np.insert(imgs, 0, values=deformed_imgs, axis=0)
        #all_targets = np.insert(targets, 0, values=deformed_tars, axis=0)
        #return all_images, all_targets

        return deformed_imgs, deformed_tars.reshape(deformed_tars.shape[0],deformed_tars.shape[1],1)

class FlipImagePreprocessor(Preprocessor):
    def __init__(self, flip='horizontal'):
        self.flip = flip
    
    def flip_image(self, image):
        if self.flip == 'vertical':
            img = cv2.flip(image, 0)
        elif self.flip == 'horizontal':
            img = cv2.flip(image, 1)
        else: #flip both ways
            img = cv2.flip(image, -1)
        return img
    
    def __call__(self, images, targets):
        imgs = images.copy()
        imgs = imgs.astype(np.float32, copy=False)
        imgs_tar = cv2.merge((imgs, targets.astype(np.float32, copy=False)))

        transformed = self.flip_image(imgs_tar)
        deformed_imgs = transformed[:,:,:-1]
        deformed_tars = transformed[:,:,-1]

        return deformed_imgs, deformed_tars.reshape(deformed_tars.shape[0],deformed_tars.shape[1],1)


if __name__ == "__main__":
    pass
