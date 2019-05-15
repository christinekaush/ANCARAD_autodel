# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 08:08:05 2019

@author: CKK1
"""

__author__ = "Christine Kiran Kaushal"
__email__ = "christine.kiran@gmail.com"

import numpy as np
from scipy.io import loadmat
import cv2
from random import sample
#from numba import jit
import preprocessing
import h5py


class DatasetMaker:
    
    def __init__(self, wanted_shape):
        
        self.max_xyn = wanted_shape 
        
        self.GTV_volumes = {}
        self.slice_id = {}
        self.slices = 0
        
        self.train_ids = []
        self.val_ids = []
        self.test_ids = []
        
        self.unwanted_s = []
        
    def pad(self, img):
        """
        Add padding to a single image
        """
        diff = self.max_xyn[0]-img.shape[0]
    
        padded_img = cv2.copyMakeBorder(img,
                                        round(diff/2), 
                                        (diff-round(diff/2)), 
                                        round(diff/2), 
                                        (diff-round(diff/2)), 
                                        cv2.BORDER_CONSTANT,
                                        value=0)
        shape = (padded_img).shape
                        
        if shape == self.max_xyn:
            return padded_img
        
        raise Exception('The wanted shape was not successfully obtained ...')
        
    def pad_seq(self, img_seq):
        """
        Add padding to imgaes in an image sequence
        """
    
        for ind, img in enumerate(img_seq):
            img_seq[ind] = self.pad(img).transpose()
    
        return img_seq
    
    @staticmethod
    def find_image(im, black_dim=70):
        """
        Check if black patches exist in the slice
        https://stackoverflow.com/questions/29663764/determine-if-an-image-exists-within-a-larger-image-and-if-so-find-it-using-py 
        """
        tpl = np.zeros((black_dim, black_dim))
        im = np.atleast_3d(im)
        tpl = np.atleast_3d(tpl)
        H, W, D = im.shape[:3]
        h, w = tpl.shape[:2]

        # Integral image and template sum per channel
        sat = im.cumsum(1).cumsum(0)
        tplsum = np.array([tpl[:, :, i].sum() for i in range(D)])
    
        # Calculate lookup table for all the possible windows
        iA, iB, iC, iD = sat[:-h, :-w], sat[:-h, w:], sat[h:, :-w], sat[h:, w:] 
        lookup = iD - iB - iC + iA
        # Possible matches
        possible_match = np.where(np.logical_and.reduce([lookup[..., i] == tplsum[i] for i in range(D)]))
    
        # Find exact match
        for y, x in zip(*possible_match):
            if np.all(im[y+1:y+h+1, x+1:x+w+1] == tpl):
                return (y+1, x+1)
    
        #raise Exception("Image not found")
        return None
        
    def find_unwanted_slices(self, seq, GTV, mod=None):
        """
        Remove slices that are mostly black in the GTV due to cut-off in the t2w-sequence
        
        Parameters:
        ----------
        
        seq : 
            
        GTV : 
            
        mod : string
            what kind of modality the image stems from
        """
        
        for ind,s in enumerate(seq):
            if sum(sum(GTV[ind])) != 0:
                val_sum = sum(s[np.where(GTV[ind]==1)])
                if val_sum == 0: #or val_sum < threshold: --> could find a minimum threshold for GTV-tissue
                    self.unwanted_s.append(ind)
                    
            elif sum(sum(GTV[ind])) == 0:
                if np.random.random(1)[0] < 0.8: #remove 80 % of the slices not containing GTV masks
                    self.unwanted_s.append(ind)
                    
            if mod == 't2w':
                # evaluate whether or not there is cut in the image:
                if self.find_image(s, black_dim=70) is not None:
                    self.unwanted_s.append(ind)

    def groom_medimg(self, pat_dir, name, img_type, patient_id='M000'):
        name = name+'.mat'
        path = pat_dir / img_type / name
        img = (loadmat(path, squeeze_me=True)['LVA_images'])
        if img_type == 'Base':
            if name == 'T2WMR.mat':
                if np.amax(img) > 500:
                    img = ((img)*76.88658/677.95636) #scale
                    
            return cv2.split(img)
        
        elif img_type == 'ROI':
            totvol = sum(sum(sum(img)))*3 #hver voxel er 1x1x3 mm^3 [mm^3]
            self.GTV_volumes[patient_id] = totvol
            return cv2.split(img)
        
        raise Exception('Image type does not exist.')
        
    def not_MRI(self, pat_dir, patient_id):
        img_dpct = self.groom_medimg(pat_dir, 'DPCT', 'Base')
        img_petct = self.groom_medimg(pat_dir, 'PETCT', 'Base')
        img_pet = self.groom_medimg(pat_dir, 'PET', 'Base')
        
        img_GTV = self.groom_medimg(pat_dir, 'GTV', 'ROI', patient_id)
    
        img_dpct = self.pad_seq(img_dpct)
        img_petct = self.pad_seq(img_petct)
        img_pet = self.pad_seq(img_pet)
        img_GTV = self.pad_seq(img_GTV)
        return img_dpct, img_petct, img_pet, img_GTV
    
    @staticmethod
    def merge_imgs(tot_slices, imgs):
        """
        tot_slices : total number of slices
        imgs (list) : list of the images to merge
        
        """
        if len(imgs) == 3:
            [img_dpct, img_petct, img_pet] = imgs
            # Merge the images so that each modality is a channel:

            merged = [cv2.merge((img_dpct[i], 
                                 img_petct[i],
                                 img_pet[i])) for i in range(tot_slices)]
        
        elif len(imgs) == 6:
            [img_dpct, img_petct, img_pet, img_adc, img_perf, img_t2w] = imgs
            
            # Merge the images so that each modality is a channel:
            merged = [cv2.merge((img_dpct[i], 
                                 img_petct[i], 
                                 img_pet[i],
                                 img_adc[i], 
                                 img_perf[i], 
                                 img_t2w[i])) for i in range(tot_slices)]
    
        #Finally save as an array:
        return np.array(merged)
    
    def augment_image_data(self, data, targets, patient_id, random_state=None):
        """
        Input:
        ------
        data : numpy.array of type np.float32
            image data with shape (slices, horizontal shape, vertical shape, channels)
        
        target : numpy.array of type np.float32
            target data with shape (1, horizontal shape, vertical shape, channels)
        
        """
        
        randoms = np.random.choice(range(data.shape[0]), int(data.shape[0]*0.35), replace=False)
        ed = preprocessing.ElasticDeformPreprocesser()        
        
        new_randoms = np.random.choice(range(data.shape[0]), int(data.shape[0]*0.35), replace=False)
        flip = preprocessing.FlipImagePreprocessor()
        
        
        deformed_slices, deformed_tars = ed.__call__(data[randoms[0],:,:,:], targets[randoms[0]])
        flipped_slices, flipped_tars = ed.__call__(data[new_randoms[0],:,:,:], targets[new_randoms[0]])
        
        aug_imgs = np.insert(deformed_slices, 0, values=flipped_slices, axis=0)
        aug_tars = np.insert(deformed_tars.reshape(1,236,236), 0, values=flipped_tars.reshape(1,236,236), axis=0)
        totvol = 0
        for r, n in zip(randoms[1:], new_randoms[1:]):
            deformed_slice, deformed_tar = ed.__call__(data[r,:,:,:], targets[r])
            flipped_slice, flipped_tar = flip.__call__(data[n,:,:,:], targets[n])

            totvol += sum(sum(flipped_tar))*3 #each voxel is 1x1x3 mm^3 [mm^3]
            totvol += sum(sum(deformed_tar))*3 #each voxel is 1x1x3 mm^3 [mm^3]
            
            aug_imgs = np.insert(aug_imgs, 0, values=deformed_slice, axis=0)
            aug_imgs = np.insert(aug_imgs, 0, values=flipped_slice, axis=0)
            aug_tars = np.insert(aug_tars, 0, values=deformed_tar, axis=0)
            aug_tars = np.insert(aug_tars, 0, values=flipped_tar, axis=0)
        
        unchanged_slices = [sum(sum(s)) for ind,s in enumerate(targets) if (ind not in randoms[1:]) or (ind not in new_randoms[1:])]
        self.GTV_volumes[patient_id+'_aug'] = totvol + sum(unchanged_slices)
        
        return aug_imgs, aug_tars

    #@jit #(nopython=True, parallel=True)
    def get_data(self, base_path, MRI=False, aug=True, remove_slices=True):
        """

        """
        if not MRI:
            data = np.empty((1,self.max_xyn[0], self.max_xyn[1],3), 
                        dtype=np.ndarray)
        else:
            data = np.empty((1,self.max_xyn[0], self.max_xyn[1],6), 
                        dtype=np.ndarray)
        GTVtarget = np.empty((1,self.max_xyn[0], self.max_xyn[1]), 
                             dtype=np.ndarray)
        
        for i in base_path.glob('M[0-9][0-9][0-9]'):
            patient_id = str(i)[-4:]

            if MRI:
                if (i / 'Base' / 'ADC.mat').exists():

                    img_dpct, img_petct, img_pet, img_GTV = self.not_MRI(i, patient_id)
                    img_adc = self.groom_medimg(i, 'ADC', 'Base')
                    img_perf = self.groom_medimg(i, 'Perf', 'Base')
                    img_t2w = self.groom_medimg(i, 'T2WMR', 'Base')
                    
                    img_adc = self.pad_seq(img_adc)
                    img_perf = self.pad_seq(img_perf)
                    img_t2w = self.pad_seq(img_t2w)
                    
                    self.unwanted_s = []
                    if remove_slices:
                        self.find_unwanted_slices(img_dpct, img_GTV)
                        self.find_unwanted_slices(img_t2w, img_GTV, mod='t2w')
                        self.find_unwanted_slices(img_adc, img_GTV)

                        # By visual inspection, these slices were not qualified:
                        if str(i)[-1] =='7':
                            self.unwanted_s.append(47)
                        elif str(i)[-2] =='31':
                            self.unwanted_s.append(6)
                        
                    merged = self.merge_imgs(len(img_dpct), 
                                         [img_dpct, img_petct, img_pet, 
                                          img_adc, img_perf, img_t2w])

                else:
                    continue
                
            else:
                img_dpct, img_petct, img_pet, img_GTV = self.not_MRI(i, patient_id)
                
                self.unwanted_s = []
                if remove_slices:
                    self.find_unwanted_slices(img_dpct, img_GTV)
                
                merged = self.merge_imgs(len(img_dpct), 
                                         [img_dpct, img_petct, img_pet])
  
            wanted = [s for s in range(len(img_dpct)) if s not in self.unwanted_s]
            merged = merged[wanted,:,:,:]
            img_GTV = [img_GTV[i] for i in wanted]
            
            self.slice_id[patient_id] = (self.slices, 
                         self.slices+merged.shape[0])
            self.slices += (merged.shape[0])
            
            if aug:
                aug_data, aug_targets = self.augment_image_data(merged, 
                                                                img_GTV, 
                                                                patient_id)
                
                self.slice_id[patient_id+'_aug'] = (self.slices, 
                             self.slices+aug_data.shape[0])
                self.slices += (aug_data.shape[0])
                
                merged = np.append(merged, aug_data, axis=0)
                img_GTV = np.append(img_GTV, aug_targets, axis=0)
            
            data = np.append(data, merged, axis=0)
            GTVtarget = np.append(GTVtarget, img_GTV, axis=0)
            print(patient_id, '    ', merged.shape)
        
        #because the first slice is only filled with 'None's:
        return [data[1:], GTVtarget[1:,...,np.newaxis]]
    
    def stratify_split(self):
        """
        Simple stratification of the split of the GTVs into training, 
        validation and test set.
        """
        
        self.train_ids = []
        self.val_ids = []
        self.test_ids = []

        sorted_GTVs = sorted(self.GTV_volumes, key=self.GTV_volumes.get)
        
        augs = [val for val in self.GTV_volumes.keys() if 'aug' in val]
        self.train_ids.extend(augs)
        
        sorted_GTVs = [pat for pat in sorted_GTVs if pat not in augs]
        
        large = sorted_GTVs[:round(len(sorted_GTVs)/2)]
        small = sorted_GTVs[round(len(sorted_GTVs)/2):]
        
        samp_L = sample(large, round(len(large)*0.7))
        samp_S = sample(small, round(len(small)*0.7))
        
        self.train_ids.extend(samp_L)
        self.train_ids.extend(samp_S)
        
        large = [e for e in large if e not in samp_L]
        small = [e for e in small if e not in samp_S]
        
        samp_L = sample(large, round(len(large)*0.5))
        samp_S = sample(small, round(len(small)*0.5))
        
        self.val_ids.extend(sample(large, round(len(large)*0.5)))
        self.val_ids.extend(sample(small, round(len(small)*0.5)))
        
        large = [e for e in large if e not in samp_L]
        small = [e for e in small if e not in samp_S]
        
        self.test_ids.extend(large)
        self.test_ids.extend(small)
        
        del samp_L, samp_S
                
    def split(self):
        """
        Simple split into training, validation and test set (no stratification)
        """        
        patients = list(self.GTV_volumes)
        
        self.train_ids = patients[:round(len(patients)*0.7)]
        self.val_ids = patients[round(len(patients)*0.7):round(len(patients)*0.8)]
        self.test_ids = patients[round(len(patients)*0.8):]
    
    def make_slice_list(self, patient_ids):
        """
        patient_ids (list) : List of the patient ids, for which slices one 
        wants
        
        Output:
        -------
        wanted_slices : extracted slices corresponding to the given patient 
        ids.
        
        """
        slices = []
        for pat in patient_ids:
            slices.extend(list(range(self.slice_id[pat][0], 
                                            self.slice_id[pat][1])))
       
        return slices
    
    def train_val_test_slices(self):
        """
        Divide the slices into train-, validation- and testset, according to
        the GTV volume and corresponding patient_ids
        
        Output:
        -------
        [train_slices, val_slices, test_slices] : list of slices associated to 
        train-, validation- and testset, respectively.
        
        """
        
        self.stratify_split()
        #self.split()

        train_slices = self.make_slice_list(self.train_ids)
        val_slices = self.make_slice_list(self.val_ids)
        test_slices = self.make_slice_list(self.test_ids)        
        
        return [train_slices, val_slices, test_slices]
    
    def make_patient_id_list(self, patients):
        #slices = []
        pats = []
        for pat in patients:
            #slices.extend(list(range(self.slice_id[pat][0], self.slice_id[pat][1])))
            pats.extend([pat]*len(range(self.slice_id[pat][0], self.slice_id[pat][1])))

        # return np.vstack((slices,pats))
        return pats
    
    def train_val_test_patients(self):
        """
        Divide the slices into train-, validation- and testset, according to
        the GTV volume and corresponding patient_ids
        
        Output:
        -------
        [train_slices, val_slices, test_slices] : list of slices associated to 
        train-, validation- and testset, respectively.
        
        """

        train_patients = self.make_patient_id_list(self.train_ids)
        val_patients = self.make_patient_id_list(self.val_ids)
        test_patients = self.make_patient_id_list(self.test_ids)        
        
        return [train_patients, val_patients, test_patients]
    

class HDFMaker(DatasetMaker):
    
    def __init__(self, filename, data, target, 
                 dataset_name='dat', id_name='pat_ids', target_name='mask'):
        self.filename = filename
        self.data = data
        self.target = target
        
        self.dataset_name = dataset_name
        self.id_name = id_name
        self.target_name = target_name
        
    def make_hdf(self, slices, ids):
        """
        Create the groups and datasets of the hdf5-file
        """
        
        [train_slices, val_slices, test_slices] = slices
        [train_ids, val_ids, test_ids] = ids
        
        with h5py.File(self.filename) as hdf:
            hdf.create_group('train')
            hdf.create_group('validation')
            hdf.create_group('test')
            
            # Add the datasets:
            hdf['train'][self.dataset_name] = np.array(self.data[train_slices], 
               dtype=np.int32)
            hdf['validation'][self.dataset_name] = np.array(self.data[val_slices], 
                    dtype=np.int32)
            hdf['test'][self.dataset_name] = np.array(self.data[test_slices], 
               dtype=np.int32)
            
            # Add patient ids:
            hdf['train'][self.id_name] = np.array(train_ids, dtype='S10')
            hdf['validation'][self.id_name] = np.array(val_ids, dtype='S10')
            hdf['test'][self.id_name] = np.array(test_ids, dtype='S10')
            
            # Add slice ids:
            hdf['train']['slice_ids'] = np.array(train_slices, dtype='S10')
            hdf['validation']['slice_ids'] = np.array(val_slices, dtype='S10')
            hdf['test']['slice_ids'] = np.array(test_slices, dtype='S10')
            
            # Add GTV target:
            hdf['train'][self.target_name] = np.array(self.target[train_slices], dtype=np.int32)
            hdf['validation'][self.target_name] = np.array(self.target[val_slices], dtype=np.int32)
            hdf['test'][self.target_name] = np.array(self.target[test_slices], dtype=np.int32)
            
        hdf.close()
    
    
