import os


import pandas as pd

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn import model_selection

import torch
from torch.utils.data import Dataset, DataLoader


def get_transforms():

    train_transforms = A.Compose(
        [
            A.Resize(256, 256, interpolation=cv2.INTER_CUBIC),
            A.Normalize(),
            ToTensorV2(),
        ]
        )

    val_transforms = A.Compose(
        [   
            A.Resize(256, 256, interpolation=cv2.INTER_CUBIC),
            A.Normalize(),
            ToTensorV2(),
        ]
        )


    test_transforms = A.Compose(
        [   
            A.Resize(256, 256, interpolation=cv2.INTER_CUBIC),
            A.Normalize(),
            ToTensorV2(),
        ]
        )
    
    return {'train_transforms': train_transforms, 
            'val_transforms': val_transforms,
            'test_transforms': test_transforms}


class ActionRecognitionDataset(Dataset):
    
    def __init__(self, data_dir, df_train_test_split, transform):

        self.data_dir = data_dir
        self.transform = transform
        self.df = df_train_test_split
    
    def __len__(self):
        return len(self.df)
    
    def _get_full_path(self, video_folder_path):
        return os.path.join(self.data_dir, video_folder_path)
    
    def _read_image(self, image_path):
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            img = self.transform(image=img)['image']
        
        return img
    
    def __getitem__(self, idx):
        
        video_path, _, label = self.df.iloc[idx]
        
        video_path = self._get_full_path(video_path)

        lst_imgs = sorted(os.listdir(video_path))
        
        imgs = torch.stack([self._read_image(os.path.join(video_path, path)) for path \
                            in lst_imgs],
                           dim=0)
        
        label = torch.tensor(label).long()
        
        return imgs, label

class ActionRecognitionDataWrapper():

    def __init__(self, data_dir, dataset, split, transforms, batch_size, num_workers):

        self.data_dir = data_dir
        self.dataset = dataset
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split

        self._setup()

    def _setup(self):
        
        annotation_path = f'./data/{self.dataset.upper()}/annotation'
        df_anno = pd.read_csv(os.path.join(annotation_path, 'train_test_split.csv'))

        # Train, Val
        df_anno_split = df_anno[['video_folder_path', self.split , 'label_id']]
        df_train_val = df_anno_split[df_anno_split[self.split] == 'train']
        

        df_train, df_val = model_selection.train_test_split(df_train_val, 
                                                            train_size=0.8, 
                                                            stratify=df_train_val['label_id'])


        # Test
        df_test = df_anno_split[df_anno_split[self.split] == 'test']
        

        

        self.train = ActionRecognitionDataset(self.data_dir, 
                                              df_train, 
                                              self.transforms['train_transforms'])
        
        self.val = ActionRecognitionDataset(self.data_dir, 
                                            df_val, 
                                            self.transforms['val_transforms'])

        self.test = ActionRecognitionDataset(self.data_dir, 
                                             df_test, 
                                             self.transforms['test_transforms'])
        
    
    def get_train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def get_val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def get_test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)

