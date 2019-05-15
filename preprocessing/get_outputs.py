# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 22:33:58 2019

@author: CKK1
"""

import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
from skimage import morphology
from matplotlib import cm

def scale_array(dat, out_range=(-1, 1)):
    domain = [np.amin(dat), np.amax(dat)]

    def interp(x):
        return out_range[0] * (1.0 - x) + out_range[1] * x

    def uninterp(x):
        b = 0
        if (domain[1] - domain[0]) != 0:
            b = domain[1] - domain[0]
        else:
            b =  1.0 / domain[1]
        return (x - domain[0]) / b

    return interp(uninterp(dat))

def colour_scale_contour(contour, colour=[33,140,0]):
    rgb_contour = cv2.merge((contour,contour,contour))
    contour_sc = scale_array(rgb_contour, out_range=(0,255))
    contour_sc[np.where((contour_sc==[255,255,255]).all(axis=2))] = colour
    return contour_sc


def get_output_image(ct_sc, pet_sc, mask, true_mask):
    # 1. Get contour positions of mask:
    cont, hierchy = cv2.findContours(mask.astype(np.uint8, copy=False), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    true_cont, true_hierchy = cv2.findContours(true_mask.astype(np.uint8, copy=False), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # 2. Get contours:
    contour = cv2.drawContours(mask, cont, -1, (0,166,33), 2)
    true_contour = cv2.drawContours(true_mask, true_cont, -1, (166,0,50), 2)

    # 3. add color to the contours:
    rgb_contour = colour_scale_contour(contour, colour=[0,160,33])
    rgb_truecontour = colour_scale_contour(true_contour, colour=[166,0,50])
    
    # 4. make the gray-image rgb:
    rgbct = cv2.cvtColor(ct_sc.astype(np.uint8, copy=True), cv2.COLOR_GRAY2BGR)
    rgbpet = cv2.cvtColor(pet_sc.astype(np.uint8, copy=True), cv2.COLOR_GRAY2BGR)
    
    # 5. Overlap mask and image:
    petct = cv2.addWeighted(rgbpet, 1, rgbct, 1, 0.0)
    
#    final_ct = cv2.addWeighted(rgb_contour.astype(np.uint8, copy=False), 1, rgbct, 1, 0.0)
#    final_ct = cv2.addWeighted(rgb_truecontour.astype(np.uint8, copy=False), 1, final_ct, 1, 0.0)
#    
#    final_pet = cv2.addWeighted(rgb_contour.astype(np.uint8, copy=False), 1, rgbpet, 1, 0.0)
#    final_pet = cv2.addWeighted(rgb_truecontour.astype(np.uint8, copy=False), 1, final_pet, 1, 0.0)

    return petct


def get_outline(mask, color, width=2):
    mask = mask.astype(bool).astype(float)
    dilated = mask.copy()
    for _ in range(width):
        dilated = morphology.dilation(dilated)
    outline = (dilated - mask).astype(float)
    c_outline = np.stack([outline]*4, axis=-1)
    c_outline[..., :-1] = color
    c_outline[..., -1] = outline
    return c_outline.squeeze()


def get_fused_output(ct, pet, mask, true_mask):
    pet = pet.astype('float32')
    pet = pet**(1/5)
    pet_img = cm.hot(pet)
    pet_img[..., -1] = pet
    
    mask_outline = get_outline(mask, [1, 0, 0])
    true_mask_outline = get_outline(true_mask, [0, 1, 0])
    
    fig, sub = plt.subplots(figsize=(5, 5))
    sub.axis('off')
    sub.imshow(ct, cmap='gray')
    sub.imshow(pet_img)
    sub.imshow(mask_outline)
    sub.imshow(true_mask_outline)
    
    return fig

def get_fused(petct, mask, true_mask, id, performance, s=' '):
    fig, sub = plt.subplots(figsize=(5, 5))
    sub.axis('off')
    sub.imshow(petct/2)
    sub.imshow(mask)
    sub.imshow(true_mask)
    #sub.set_title(f' {id}, slice {s}, Dice: {performance:.3f}', fontsize=18, fontname='Palatino Linotype')
    sub.text(5,230,f'slice {s}', fontsize=18, fontname='Palatino Linotype', color='white')
    sub.text(155,230,f'Dice: {performance:.3f}', fontsize=18, fontname='Palatino Linotype', color='white')
    return fig
    
def process_results(out_path, pat_ids, names, slice_ids, val=True):
    """
    names (list) : list of the filenames
    slice_ids (list) : list of slice ids
    
    """
    group = 'val' if val else 'test'
    
    with h5py.File(out_path, 'r') as f:
        imgs = f[group]['images'].value
        masks = f[group]['prediction'].value
        targets = f[group]['masks'].value
        dice = f[group]['dice'].value
        
        #imgs[..., 1] = (imgs[..., 1] - imgs[..., 1].min())/(imgs[..., 1].max() - imgs[..., 1].min())
        for ind, mask in enumerate(masks[:,:,:,0]):
            print(ind)
            ct = imgs[ind,:,:,0]
            pet = imgs[ind,:,:,1]
            true_mask = targets[ind]
            
            ct_sc = scale_array(ct, out_range=(0,255))
            pet_sc = scale_array(pet, out_range=(0,255))
            
            #ct = (ct - ct.min())/(ct.max() - ct.min())
            #petct = get_fused_output(ct, pet, mask, true_mask)
       
            #petct = get_output_image(ct_sc, pet_sc, mask, true_mask)
                    
            mask_outline = get_outline(mask, [0, 174/255, 255/255])
            true_mask_outline = get_outline(true_mask, [0, 148/255, 0])
            
            plt.imsave('pet.png', pet_sc, cmap='hot')
            plt.imsave('ct.png', ct_sc, cmap='gray')
            
            pet = plt.imread('pet.png')
            ct = plt.imread('ct.png')
            petct = cv2.addWeighted(pet, 1, ct, 1, 0.0)
            
            data = 'val' if val else 'test'
            if str(f'{data}_fused{ind}.png') in names:
                s = slice_ids[np.where(np.array(names) == f'{data}_fused{ind}.png')[0][0]]
                fig = get_fused(petct, 
                                mask_outline, 
                                true_mask_outline, 
                                pat_ids[ind], 
                                dice[ind],
                                s=s)
                fig.savefig(f'.\\{data}_fused_{pat_ids[ind]}_{s}.pdf')
            else:
                fig = get_fused(petct, 
                                mask_outline, 
                                true_mask_outline, 
                                pat_ids[ind], 
                                dice[ind])
            fig.savefig(f'..\\resulting_images\\{data}_fused{ind}.pdf')
            
            #fig = get_fused_output(ct_sc, pet_sc, mask, true_mask)
            
#%%
from scipy.io import loadmat as spio
    
CT = spio('.\Data_070119\M007\Base\DPCT.mat', squeeze_me=True)['LVA_images']
CTV = spio('.\Data_070119\M007\ROI\CTV.mat', squeeze_me=True)['LVA_images']
GTV = spio('.\Data_070119\M007\ROI\GTV.mat', squeeze_me=True)['LVA_images']
ind=28


ct = CT[:,:,ind].transpose()
ctv = CTV[:,:,ind].transpose()
gtv = GTV[:,:,ind].transpose()

ct_sc = scale_array(ct, out_range=(0,255))
      
gtv_mask_outline = get_outline(gtv, [0, 174/255, 255/255])
ctv_mask_outline = get_outline(ctv, [203/255,101/255,104/255])

fig, sub = plt.subplots(figsize=(5, 5))
sub.axis('off')
sub.imshow(ct, cmap='gray')
sub.imshow(gtv_mask_outline)
sub.imshow(ctv_mask_outline)
plt.show(fig)

fig.savefig(f'CTV_GTV_DPCT.png')



#%%
            
import glob
from natsort import natsorted

val_slice_ids = [3, 7, 11, 12, 4, 9, 14, 16, 20, 22,
                 23, 3, 21, 28, 31, 33, 36, 41, 42, 44,
                 47, 25, 27, 29]
test_slice_ids = [3, 7, 2, 4, 7, 12, 10, 14, 16, 21,
                  29, 22, 27, 29, 10, 19, 23, 28, 34,
                  37, 16, 24, 28, 30]
tests = []
vals = []
for i,j in zip(natsorted(glob.glob('test*.png')), natsorted(glob.glob('val*.png'))):
    tests.append(i)
    vals.append(j)

with h5py.File('../data_070119_MRI_final.h5','r') as o:
    pat_ids_val = (o['validation']['pat_ids'].value).astype(str)
    pat_ids_test = (o['test']['pat_ids'].value).astype(str)

#%%
out_path = '../code\logs\PETCT_petct_windowing_c32_w220_aug_basic_f1_adam_03\outputs_4998.h5'
process_results(out_path, pat_ids_val, vals, val_slice_ids, val=True)
process_results(out_path, pat_ids_test, tests, test_slice_ids, val=False)