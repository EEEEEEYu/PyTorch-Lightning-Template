# Copyright 2024 Haowen Yu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import os
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

from torch.nn.functional import one_hot

# This user-defined dataset inherits from torch.utils.data.Dataset which defines how data is loaded and transformed.
# This version assumes the task is classification and dataset has the following format:
"""
Format 1: label is defined in the name of training instances. The name is split by a `_`
dataset_dir/
    instances/
        instance0_cat.png
        instance1_dog.png
        instance2_cat.png
        ...
"""

"""
Format 2: label is a separate file(commonly used in segmentation, pose estimation tasks)
dataset_dir/
    instances/
        instance0.png
        instance1.png
        instance2.png
        ...
    labels/
        instance0.npz
        instance1.npz
        instance2.npz
        ...
"""


# For maximum generalization capability, this class is using a 'map' style dataset. The file paths will be loaded into
# a dictionary with KV defined as: <instance_path, label/label_path>. The user's need variates greatly.
# Make appropriate changes to this class to fit in your needs.
class ImageDataset(data.Dataset):
    def __init__(
            self,
            dataset_dir='.',
            purpose='train',
            split_ratios=(0.8, 0.1, 0.1),
            aug_prob=0.5,
            use_augmentation=True,
            label_is_obj=False
    ):
        self.purpose = purpose
        assert self.purpose in ('train', 'validation', 'test'), f"Undefined purpose: {self.purpose}"
        self.split_ratios = split_ratios
        self.dataset_dir = dataset_dir
        self.use_augmentation = use_augmentation
        self.aug_prob = aug_prob
        self.label_is_obj = label_is_obj

        self.instance_list, self.label_list, self.label_info = self.__init_file_list()
        self.augmentation = self.__configure_augmentation()
        self.idx_range = self.__train_val_test_split()

    def __init_file_list(self):
        if not os.path.exists(os.path.join(self.dataset_dir, 'instances')):
            raise FileNotFoundError('Instance directory not found!')
        instance_list = os.listdir(os.path.join(self.dataset_dir, 'instances'))

        # If label is object type, load label directory. Assume label file has the same name as instance file.
        label_info = None
        if self.label_is_obj:
            if not os.path.exists(os.path.join(self.dataset_dir, 'labels')):
                raise FileNotFoundError('Label directory not found!')
            label_list = os.listdir(os.path.join(self.dataset_dir, 'labels'))

        # Extract label from instance file name if label is not object type. Infer discrete label's property
        # (e.g. num_of_class) from the complete label list and convert to a dict: <label_name, label_rank>.
        # Label names are sorted in lexicographical order.
        else:
            label_list = [path.split('_')[-1].split('.')[0] for path in instance_list]
            label_list_unique = sorted(list(set(label_list)))
            label_info = dict([(name, i) for i, name in enumerate(label_list_unique)])

        return instance_list, label_list, label_info

    def __configure_augmentation(self):
        augmentation = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(self.aug_prob),
                transforms.RandomVerticalFlip(self.aug_prob),
                transforms.RandomRotation(10),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]
        ) if self.purpose == 'train' else transforms.Compose(
            [
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]
        )

        return augmentation

    def __train_val_test_split(self):
        train_len = len(self.instance_list) * self.split_ratios[0]
        val_len = len(self.instance_list) * self.split_ratios[1]
        # test_len = len(self.instance_list) - train_len - val_len

        if self.purpose == 'train':
            return 0, train_len
        elif self.purpose == 'validation':
            return train_len, train_len + val_len
        else:
            return train_len + val_len, len(self.instance_list)

    # Modify this if your task is not using one-hot encoding labels.
    def __transform_label_to_tensor(self, label):
        return one_hot(torch.tensor(self.label_info[label]), len(self.label_info)).float()

    def __len__(self):
        return self.idx_range[1] - self.idx_range[0]

    def __getitem__(self, idx):
        instance_path = os.path.join(self.dataset_dir, 'instances', self.instance_list[idx])
        img = np.load(instance_path)

        # If your image is store as (H, W, C), uncomment this line to transpose the axis to (C, H, W)
        # img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).float()

        if self.label_is_obj:
            label_tensor = self.__transform_label_to_tensor(self.label_list[idx])
        else:
            label_path = os.path.join(self.dataset_dir, 'labels', self.label_list[idx])
            label = np.load(label_path)
            label_tensor = torch.from_numpy(label).float()

        if self.use_augmentation:
            img_tensor = self.augmentation(img_tensor)

        return img_tensor, label_tensor
