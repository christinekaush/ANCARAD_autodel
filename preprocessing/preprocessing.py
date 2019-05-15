
import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


class ElasticDeformPreprocesser:
    """
    """

    def __init__(self, alpha=80, sigma=25, alpha_affine=15):
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
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine

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

    def __call__(self, images, targets):
        imgs = images.copy()
        imgs = imgs.astype(np.float32, copy=False)
        imgs_tar = cv2.merge((imgs, targets.astype(np.float32, copy=False)))

        transformed = self.elastic_transform(imgs_tar).astype(np.int32, copy=False)
        deformed_imgs = transformed[:,:,:-1]
        deformed_tars = transformed[:,:,-1]

        #deformed_tars = deformed_tars.reshape(deformed_tars.shape[0],deformed_tars.shape[1],1)
        imgs = imgs.reshape(1, imgs.shape[0], imgs.shape[1], imgs.shape[2])
        deformed_imgs = deformed_imgs.reshape(1, deformed_imgs.shape[0],deformed_imgs.shape[1], deformed_imgs.shape[2])
        
        return deformed_imgs, deformed_tars

class FlipImagePreprocessor:
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

        #deformed_tars = deformed_tars.reshape(deformed_tars.shape[0],deformed_tars.shape[1],1)
        imgs = imgs.reshape(1, imgs.shape[0], imgs.shape[1], imgs.shape[2])
        deformed_imgs = deformed_imgs.reshape(1, deformed_imgs.shape[0], deformed_imgs.shape[1], deformed_imgs.shape[2])

        return deformed_imgs, deformed_tars
