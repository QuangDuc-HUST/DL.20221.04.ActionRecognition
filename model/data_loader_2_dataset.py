import os

import pandas as pd

import cv2
import albumentations as A

from sklearn import model_selection

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class ActionRecognitionDataset(Dataset):
    """
    Dataset class for training and validation stage
    """
    def __init__(self, data_dir, df_train_test_split, is_first_dataset, dataset_name, transform):

        self.data_dir = data_dir
        self.transform = transform
        self.df = df_train_test_split
        self.is_first_dataset = is_first_dataset
        self.dataset_name = dataset_name
    
    def __len__(self):
        return len(self.df)
    
    def _get_full_path(self, video_folder_path):
        return os.path.join(self.data_dir, video_folder_path)
    
    def _read_image(self, image_path, data_replay):
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            img = A.ReplayCompose.replay(data_replay, image=img)['image']
        
        return img
    
    def _read_one_clip(self, folder_path):
        
        lst_imgs = sorted(os.listdir(folder_path))

        data_replay = None

        if self.transform is not None:
            # Read the first img
            first_img = cv2.imread(os.path.join(folder_path, lst_imgs[0]))
            first_img = cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)
            data_replay = self.transform(image=first_img)['replay']
        
        imgs = torch.stack([self._read_image(os.path.join(folder_path, path), data_replay) for path \
                            in lst_imgs],
                           dim=0)

        return imgs

    def __getitem__(self, idx):
        
        clip_path, _, label = self.df.iloc[idx]
        
        full_clip_path = self._get_full_path(clip_path)

        imgs = self._read_one_clip(full_clip_path)
        
        label = torch.tensor(label).long()
        
        return imgs, label, self.is_first_dataset

class ActionRecognitionDatasetTest(ActionRecognitionDataset):
    """
    Dataset class for test stage
    """

    def __init__(self, data_dir, df_train_test_split, transform, clip_per_video):
        super().__init__(data_dir, df_train_test_split, transform)
        self.clip_per_video = clip_per_video

    def __getitem__(self, idx):
        
        video_path, _, label = self.df.iloc[idx]
        
        video_path = self._get_full_path(video_path)

        if self.clip_per_video <= 1:
            clip_path = video_path
            imgs = self._read_one_clip(clip_path)
        
        else:
            clip_paths = os.listdir(video_path)
            imgs = torch.stack([self._read_one_clip(os.path.join(video_path, path)) for path in clip_paths],
                                dim=0)

        label = torch.tensor(label).long()
        return imgs, label


class ActionRecognitionDataWrapper():
    """
    Class for generating dataloader for train, validation, and test set 
    """
    def __init__(self, 
                 data_dir_1, 
                 dataset_1,
                 data_dir_2,
                 dataset_2, 
                 data_split, 
                 transforms, 
                 batch_size, 
                 num_workers,
                 clip_per_video,
                 *args, 
                 **kwargs):

        self.data_dir_1 = data_dir_1
        self.data_dir_2 = data_dir_2
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = data_split
        self.clip_per_video = clip_per_video

        self._setup()


    def _setup(self):
        
        df_anno_train_val_1, df_anno_test_1, df_anno_train_val_2, df_anno_test_2 = self._get_annotation_pandas()

        # Train, Val
        df_train_val_1 = df_anno_train_val_1[['video_folder_path', self.split , 'label_id']]
        df_train_val_2 = df_anno_train_val_2[['video_folder_path', self.split , 'label_id']]

        df_train_1, df_val_1 = model_selection.train_test_split(df_train_val_1, 
                                                            train_size=0.8, 
                                                            stratify=df_train_val_1['label_id'])

        df_train_2, df_val_2 = model_selection.train_test_split(df_train_val_2, 
                                                            train_size=0.8, 
                                                            stratify=df_train_val_2['label_id'])


        # Test
        df_test_1 = df_anno_test_1[['video_folder_path', self.split , 'label_id']]
        df_test_2 = df_anno_test_2[['video_folder_path', self.split , 'label_id']]


        self.train_1 = ActionRecognitionDataset(self.data_dir_1, 
                                              df_train_1, 
                                              self.transforms['train_transforms'])
                    
        self.train_2 = ActionRecognitionDataset(self.data_dir_2, 
                                              df_train_2, 
                                              self.transforms['train_transforms'])


        self.train = ConcatDataset([self.train_1, self.train_2])
        

        self.val_1 = ActionRecognitionDataset(self.data_dir_1, 
                                             df_val_1,
                                             self.transforms['val_transforms'])

        self.val_2 = ActionRecognitionDataset(self.data_dir_2, 
                                            df_val_2, 
                                            self.transforms['val_transforms'])

        self.test_1 = ActionRecognitionDatasetTest(self.data_dir_1, 
                                             df_test_1, 
                                             self.transforms['test_transforms'],
                                             self.clip_per_video)

        self.test_2 = ActionRecognitionDatasetTest(self.data_dir_2, 
                                             df_test_2, 
                                             self.transforms['test_transforms'],
                                             self.clip_per_video)

    def _get_annotation_pandas(self):
        """
        Get df based on number of clips per video
        start from clip_1, clip_2, .. (if self.clip_per_video > 1)
        or default

        """
        def _get_name_clip(name_clip):
            """
            Support function for pandas explore
            
            """
            return [os.path.join(name_clip, f'clip_{x+1}') for x in range(self.clip_per_video)]


        annotation_folder_1 = f'./data/{self.dataset_1.upper()}/annotation/'
        default_df_1 = pd.read_csv(os.path.join(annotation_folder_1, 'train_test_split.csv'))

        train_val_df_1 = default_df_1[default_df_1[self.split] == 'train'].copy(deep=True)
        test_df_1 = default_df_1[default_df_1[self.split] == 'test'].copy(deep=True)

        if self.clip_per_video > 1:
            train_val_df_1['video_folder_path'] = train_val_df_1['video_folder_path'].apply(_get_name_clip)
            train_val_df_1 = train_val_df_1.explode('video_folder_path').reset_index()
            del train_val_df_1['index']  #reset index


        annotation_folder_2 = f'./data/{self.dataset_2.upper()}/annotation/'
        default_df_2 = pd.read_csv(os.path.join(annotation_folder_2, 'train_test_split.csv'))

        train_val_df_2 = default_df_2[default_df_2[self.split] == 'train'].copy(deep=True)
        test_df_2 = default_df_2[default_df_2[self.split] == 'test'].copy(deep=True)

        if self.clip_per_video > 1:
            train_val_df_2['video_folder_path'] = train_val_df_2['video_folder_path'].apply(_get_name_clip)
            train_val_df_2 = train_val_df_2.explode('video_folder_path').reset_index()
            del train_val_df_2['index']  #reset index

        return train_val_df_1, test_df_1, train_val_df_2, test_df_2
    
    def get_train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def get_val_1_dataloader(self):
        return DataLoader(self.val_1, batch_size=self.batch_size, num_workers=self.num_workers)

    def get_val_2_dataloader(self):
        return DataLoader(self.val_2, batch_size=self.batch_size, num_workers=self.num_workers)

    def get_test_1_dataloader(self):
        return DataLoader(self.test_1, batch_size=self.batch_size, num_workers=self.num_workers)

    def get_test_2_dataloader(self):
        return DataLoader(self.test_2, batch_size=self.batch_size, num_workers=self.num_workers)


