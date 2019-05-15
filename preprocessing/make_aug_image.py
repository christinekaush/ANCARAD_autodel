# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 21:54:10 2019

@author: christink
"""

import numpy as np
from scipy.io import loadmat
import preprocessing
import matplotlib.pyplot as plt


data = loadmat('..\Data_070119\M070\Base\DPCT.mat', squeeze_me=True)['LVA_images']

ed = preprocessing.ElasticDeformPreprocesser(alpha=80, sigma=25, alpha_affine=15) 
flip = preprocessing.FlipImagePreprocessor()

deformed = ed.elastic_transform(data).astype(np.int32, copy=False)
flipped = flip.flip_image(data)   

plt.imshow(deformed[:,:,25].transpose(), cmap='gray')
plt.imshow(flipped[:,:,25].transpose(), cmap='gray')
plt.imshow(data[:,:,25].transpose(), cmap='gray')
        
plt.imsave('deformed_M007_25', deformed[:,:,25].transpose(), cmap='gray')
plt.imsave('org_M007_25', data[:,:,25].transpose(), cmap='gray')
plt.imsave('flipped_M007_25', flipped[:,:,25].transpose(), cmap='gray')


#============================================================================

from pathlib import Path
import cv2
import make_hdf5
import preprocessing

base_path = Path('.\\Data_070119\\')
max_xyn = (236, 236)
ck = make_hdf5.DatasetMaker(max_xyn)
imgs, tar = ck.get_data(base_path, MRI=True, aug=True, remove_slices=False)

plt.imsave('dpctorg__M027_27.png', imgs[27,:,:,0].astype(np.float32, copy=False), cmap='gray')
plt.imsave('petorg_M027_27.png', imgs[27,:,:,2].astype(np.float32, copy=False), cmap='afmhot')
plt.imsave('t2worg_M027_27.png', imgs[27,:,:,5].astype(np.float32, copy=False), cmap='gray')
plt.imsave('adcorg_M027_27.png', imgs[27,:,:,3].astype(np.float32, copy=False), cmap='gray')
plt.imsave('maskorg_M027_27.png', tar[27,:,:,0].astype(np.float32, copy=False), cmap='gray')


ed = preprocessing.ElasticDeformPreprocesser(alpha=80, sigma=35, alpha_affine=15) 
imgs1 = imgs.astype(np.float32, copy=False)
tar = tar.astype(np.float32, copy=False)
data = cv2.merge((imgs1[27,:,:,:], tar[27]))
deformed = ed.elastic_transform(data).astype(np.float32, copy=False)


plt.imsave('dpctaug_M027_27.png', deformed[:,:,0], cmap='gray')
plt.imsave('petaug__M027_27.png', deformed[:,:,2], cmap='hot')
plt.imsave('t2waug__M027_27.png', deformed[:,:,5], cmap='gray')
plt.imsave('adcaug__M027_27.png', deformed[:,:,3], cmap='gray')
plt.imsave('maskaug_M027_27.png', deformed[:,:,-1], cmap='gray')
