import os

import pandas as pd

import cv2
import albumentations as A

from sklearn import model_selection

import torch
from torch.utils.data import Dataset, DataLoader


class ActionRecognitionDataset(Dataset):
    """
    Dataset class for training and validation stage
    """
    def __init__(self, data_dir, df_train_test_split, transform):

        self.data_dir = data_dir
        self.transform = transform
        self.df = df_train_test_split

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

        imgs = torch.stack([self._read_image(os.path.join(folder_path, path), data_replay) for path
                            in lst_imgs],
                           dim=0)

        return imgs

    def __getitem__(self, idx):

        clip_path, _, label = self.df.iloc[idx]

        full_clip_path = self._get_full_path(clip_path)

        imgs = self._read_one_clip(full_clip_path)

        label = torch.tensor(label).long()

        return imgs, label


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
                 data_dir,
                 dataset,
                 data_split,
                 transforms,
                 batch_size,
                 num_workers,
                 clip_per_video,
                 *args,
                 **kwargs):

        self.data_dir = data_dir
        self.dataset = dataset
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = data_split
        self.clip_per_video = clip_per_video

        self._setup()

    def _setup(self):

        df_anno_train_val, df_anno_test = self._get_annotation_pandas()

        # Train, Val
        df_train_val = df_anno_train_val[['video_folder_path', self.split, 'label_id']]

        df_train, df_val = model_selection.train_test_split(df_train_val,
                                                            train_size=0.8,
                                                            stratify=df_train_val['label_id'])

        # Test
        df_test = df_anno_test[['video_folder_path', self.split, 'label_id']]

        self.train = ActionRecognitionDataset(self.data_dir,
                                              df_train,
                                              self.transforms['train_transforms'])

        self.val = ActionRecognitionDataset(self.data_dir,
                                            df_val,
                                            self.transforms['val_transforms'])

        self.test = ActionRecognitionDatasetTest(self.data_dir,
                                                 df_test,
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

        annotation_folder = f'./data/{self.dataset.upper()}/annotation/'
        default_df = pd.read_csv(os.path.join(annotation_folder, 'train_test_split.csv'))

        train_val_df = default_df[default_df[self.split] == 'train'].copy(deep=True)
        test_df = default_df[default_df[self.split] == 'test'].copy(deep=True)

        if self.clip_per_video > 1:
            train_val_df['video_folder_path'] = train_val_df['video_folder_path'].apply(_get_name_clip)
            train_val_df = train_val_df.explode('video_folder_path').reset_index()
            del train_val_df['index']  # reset index

        return train_val_df, test_df

    def get_train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def get_val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def get_test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)
