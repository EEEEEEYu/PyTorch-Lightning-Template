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

from typing import Dict, List, Optional, Tuple

import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self, in_channels: int = 3, blocks: Optional[Dict[str, Dict[str, List[int]]]] = None, num_classes: int = 10, input_meta = None):
        super().__init__()
        if blocks is None:
            raise ValueError("SimpleNet expects a 'blocks' configuration section.")
        
        print(f"Input meta for model: {input_meta}")

        block1_c1, block1_c2 = self._extract_channels(blocks, "block1", expected=2, default=[32, 64])
        block2_c1, block2_c2 = self._extract_channels(blocks, "block2", expected=2, default=[128, 256])
        block3_c1, = self._extract_channels(blocks, "block3", expected=1, default=[256])

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=block1_c1, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(block1_c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=block1_c1, out_channels=block1_c2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(block1_c2),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=block1_c2, out_channels=block2_c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(block2_c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=block2_c1, out_channels=block2_c2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(block2_c2),
            nn.ReLU(inplace=True),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=block2_c2, out_channels=block3_c1, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(block3_c1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(16 * block3_c1, 4 * block3_c1),
            nn.ReLU(inplace=True),
            nn.Linear(4 * block3_c1, block3_c1),
            nn.ReLU(inplace=True),
            nn.Linear(block3_c1, self.num_classes),
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        return out

    @staticmethod
    def _extract_channels(blocks: Dict[str, Dict[str, List[int]]], block_name: str, expected: int, default: List[int]) -> Tuple[int, ...]:
        block_cfg = blocks.get(block_name, {})
        channels = block_cfg.get("hidden_channels", default)
        if len(channels) < expected:
            raise ValueError(f"Block '{block_name}' expects at least {expected} hidden_channels entries.")
        return tuple(channels[:expected])
