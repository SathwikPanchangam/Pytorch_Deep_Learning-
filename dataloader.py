# import the modules
import os
import glob
from turtle import pd
from matplotlib import image
from numpy import imag
from setuptools.namespaces import flatten
import random
from itertools import islice
from typing import Any, List
import cv2 as cv
import pandas as pd
import torch
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset


# Class for loading image datasets.
class ImageDataLoader(VisionDataset):
    '''
    This class returns the images and labels for the image dataset.
    '''
    def __init__(self, root: str, train = False, test=False,valid=False,transform =None):
        super().__init__()
        
        self.root = root
        self.train_size = 0.7
        self.test_size = 0.2
        self.valid = 0.1
        self.transform = transform
        self.imagepaths = []


        # Creating a dictionary for the classes and their labels.
        classes  = []
        for path in glob.glob(self.root + '/*'):
            classes.append(path.split('/')[-1])
        
        idx_to_class = {i:val for i,val in enumerate(classes)}
        class_to_idx = {value:key for key, value in idx_to_class.items()}

        self.class_to_idx = class_to_idx
        
        # train set paths
        if train == True:
            self.imagepaths = self.get_data_paths(self.root)[0]

        # test set paths
        if test == True:
            self.imagepaths = self.get_data_paths(self.root)[1]

        # Validation set paths
        if valid == True:
            self.imagepaths = self.get_data_paths(self.root)[2]

    def get_data_paths(self,root):
        '''
        This function returns the paths for all the images in training, testing and
        validation datasets.
        '''
        for path in glob.glob(root + '/*'):
            self.imagepaths.append(glob.glob(path + '/*'))

        image_paths = list(flatten(self.imagepaths))
        random.shuffle(image_paths)
        total_images = len(image_paths)

        # Splitting the dataset into train, test and validiation sets.     
        splits = [int(self.train_size*total_images),
                  int(self.test_size * total_images), 
                  int(self.valid_size * total_images)]

        output = [list(islice(image_paths,elem)) for elem in splits]

        train_image_paths = output[0]
        test_image_paths = output[1]
        valid_image_paths = output[2]
        
        return train_image_paths, test_image_paths, valid_image_paths

    def __len__(self) -> int:
        return len(self.imagepaths)

    def __getitem__(self, index: int) -> Any:
        
        image_file_path = self.imagepaths[index]
        # Reading the image
        image = cv.imread(image_file_path)

        # Changing the default BGR version of cv2 to RGB channels.
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Creating the labels
        label = image_file_path.split('/')[-2]
        label = self.class_to_idx[label]

        # Applying transform to the image
        if self.transform:
            image = self.transform(image)

        return image, label



class CsvImageDataLoader(VisionDataset):
    '''
    This class returns the images and their corresponding lables from a csv and images dataset.
    '''
    def __init__(self, csv_file: str,root_dir: str, transform = None):
        super().__init__()
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> Any:
        image_path = os.path.join(self.root, self.annotations.iloc[index,0])
        # Reading the image
        image = cv.imread(image_path)
        # Changing the default BGR version of cv2 to RGB channels.
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        label = torch.tensor(int(self.annotations.iloc[index,1]))

        return image, label