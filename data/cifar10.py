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
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F


class Cifar10(data.Dataset):
    def __init__(
            self,
            root=None,
            purpose='train',
            image_height=32,
            image_width=32,
            num_classes=10,
            augmentation_cfg=None
    ):
        super().__init__()
        self.purpose = purpose
        self.aug_prob = augmentation_cfg.get("probability")
        self.use_augmentation = augmentation_cfg.get("enabled")
        self.augmentation = self.__configure_augmentation()

        if purpose == 'train':
            self.dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
        elif purpose == 'validation':
            self.dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
        else:
            self.dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True)

    def __configure_augmentation(self):
        if self.purpose == 'train' and self.use_augmentation:
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(self.aug_prob),
                    transforms.RandomRotation(10),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ]
            )

        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_tensor, label = self.dataset[idx]
        if self.use_augmentation:
            img_tensor = self.augmentation(img_tensor)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor
