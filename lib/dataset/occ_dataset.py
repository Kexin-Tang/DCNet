# --------------------------------------------------------
# P2ORM: Formulation, Inference & Application
# Licensed under The MIT License [see LICENSE for details]
# Written by Xuchong Qiu
# --------------------------------------------------------

import torch.utils.data as data
import os
import os.path
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import h5py
import torchvision.transforms as F
PI = 3.1416


class PIOD_Dataset(data.Dataset):
    """generic dataset loader for occlusion edge/ori/order estimation from image"""
    def __init__(self, config, isTest=False, input_transf=None, target_transf=None, co_transf=None, edge_only=True):
        self.input_transform = input_transf
        self.target_transform = target_transf
        self.co_transform = co_transf
        self.config = config
        self.isTest = isTest

        self.edge_only = edge_only

        self.img_root = os.path.join(config.dataset.train_image_set, 'Augmentation', 'Aug_JPEGImages')
        self.label_root = os.path.join(config.dataset.train_image_set, 'Augmentation', 'Aug_HDF5EdgeOriLabel')
        # training
        if not self.isTest:
            list_file = os.path.join(config.dataset.train_image_set, 'Augmentation', 'train_pair_320x320.lst')
            with open(list_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    lines[i] = (os.path.split(line.rstrip().split()[0])[1])[:-4]
                    
        # eval
        else:
            list_file = os.path.join(config.dataset.val_image_set, 'val_doc_2010.txt')
            with open(list_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    lines[i] = line.rstrip()
        self.img_list = sorted(lines)

    def __getitem__(self, idx):
        # load sample data
        img_path = os.path.join(self.img_root, self.img_list[idx] + '.jpg')


        img = Image.open(img_path, 'r')
        img_org_sz = [img.size[1], img.size[0]]  # H,W

        # transforms and data augmentation
        label_filename = os.path.join(self.label_root, self.img_list[idx] + '.h5')
        h5 = h5py.File(label_filename, 'r')
        label_occ = np.squeeze(h5['label'][...])
        label_occ = np.transpose(label_occ, axes=(1, 2, 0))  ## (H,W,2)   0-edgemap 1-orientmap
        occ_lbl_list = [label_occ[:, :, 0]]

        if self.input_transform is not None: img = self.input_transform(img)
        if self.target_transform is not None: occ_lbl_list = self.target_transform(occ_lbl_list)
        if self.co_transform is not None:
            occ_lbl_list = [Image.fromarray(label) for label in occ_lbl_list]
            input_pair = {'image': img, 'label': occ_lbl_list}
            input_pair = self.co_transform(input_pair)
            img = input_pair['image']
            occ_lbl_list = input_pair['label']
        sample = ((img), tuple(occ_lbl_list), img_path)

        return sample

    def __len__(self):
        return len(self.img_list)


'''
    @ only use in eval
'''
class My_Dataset(data.Dataset):
    def __init__(self, config, isTest=True, input_transf=None, target_transf=None, co_transf=None, edge_only=False):
        self.input_transform = input_transf
        self.target_transform = target_transf
        self.co_transform = co_transf
        self.config = config
        self.isTest = isTest

        self.edge_only = edge_only


        list_file = os.path.join(config.dataset.val_image_set, 'list.txt')
        self.img_root = os.path.join(config.dataset.val_image_set)
        with open(list_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                '''
                    @ if list of files do not contain <.type>
                '''
                # lines[i] = (os.path.split(line.rstrip().split()[0])[1])[:-4]
                '''
                    @ if list of files do contain <.type>
                '''
                lines[i] = line.rstrip()
        self.img_list = sorted(lines)


    def __getitem__(self, idx):
        
        # img_path = os.path.join(self.img_root, self.img_list[idx] + '.jpg')
        img_path = os.path.join(self.img_root, self.img_list[idx])
        

        img = Image.open(img_path, 'r')
        img_org_sz = [img.size[1], img.size[0]]  # H,W

        if self.input_transform is not None: img = self.input_transform(img)
        if self.target_transform is not None: occ_lbl_list = self.target_transform(occ_lbl_list)
        if self.co_transform is not None:
            input_pair = {'image': img, 'label':[[]]}
            input_pair = self.co_transform(input_pair)
            img = input_pair['image']
        sample = ((img), img_path)

        return sample

    def __len__(self):
        return len(self.img_list)




def get_net_loadsz(org_img_sz, minSize):
    """
    for net with input constrain
    :param org_img_sz: [H, W]
    :param minSize: int
    :return: net load size
    """
    H_load = (org_img_sz[0] // minSize) * minSize
    W_load = (org_img_sz[1] // minSize) * minSize
    if H_load < org_img_sz[0]: H_load += minSize
    if W_load < org_img_sz[1]: W_load += minSize

    return H_load, W_load
