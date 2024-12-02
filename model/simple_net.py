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

import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(
            self,
            in_channels=3,
            block1_hidden_channel1=32,
            block1_hidden_channel2=64,
            block2_hidden_channel1=128,
            block2_hidden_channel2=256,
            block3_hidden_channel1=256,
            num_classes=10
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=block1_hidden_channel1, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(block1_hidden_channel1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=block1_hidden_channel1, out_channels=block1_hidden_channel2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(block1_hidden_channel2),
            nn.ReLU(inplace=True)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=block1_hidden_channel2, out_channels=block2_hidden_channel1, kernel_size=3, padding=1),
            nn.BatchNorm2d(block2_hidden_channel1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=block2_hidden_channel1, out_channels=block2_hidden_channel2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(block2_hidden_channel2),
            nn.ReLU(inplace=True)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=block2_hidden_channel2, out_channels=block3_hidden_channel1, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(block3_hidden_channel1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(16 * block3_hidden_channel1, 4 * block3_hidden_channel1),
            nn.ReLU(inplace=True),
            nn.Linear(4 * block3_hidden_channel1, block3_hidden_channel1),
            nn.ReLU(inplace=True),
            nn.Linear(block3_hidden_channel1, self.num_classes),
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        return out
