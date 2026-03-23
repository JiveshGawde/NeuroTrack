from collections import defaultdict
from typing import Optional, Tuple
import torch
import re
from torchvision import transforms
import torch.nn as nn
import glob
import os
import cv2
from torch.utils.data import Dataset
from src.config import FULL_DATA_PATH


MILD_DEMENTED_PATH: str = os.path.abspath(os.path.join(FULL_DATA_PATH, "MildDemented/MildDemented"))
MODERATE_DEMENTED_PATH: str = os.path.abspath(os.path.join(FULL_DATA_PATH, "ModerateDemented/ModerateDemented"))
NON_DEMENTED_PATH: str = os.path.abspath(os.path.join(FULL_DATA_PATH, "NonDemented (2)/NonDemented"))
VERYMILD_DEMENTED_PATH: str = os.path.abspath(os.path.join(FULL_DATA_PATH, "VeryMildDemented/VeryMildDemented"))

_FILENAME_RE = re.compile(
    r"OAS1_(?P<patient>\d+)_(?P<scan>MR\d+_\d+)\.nii_slice_(?P<slice>\d+)\.png"
)

LABEL_FOLDERS = [(0, MILD_DEMENTED_PATH), 
                 (1, MODERATE_DEMENTED_PATH), 
                 (2, NON_DEMENTED_PATH), 
                 (3, VERYMILD_DEMENTED_PATH)]

class AlzheimersDataset(Dataset):
    def __init__(self, n_images: int = 3, random_state: int = 0, transform: Optional[transforms.Compose] = None):
        self.n_images = n_images 
        self.random_state = random_state
        self.transform: Optional[transforms.Compose] = transform

        self.window = []
        group_images = defaultdict(list)

        for label, folder in LABEL_FOLDERS:
            for path in glob.glob(os.path.join(folder, "*.png")):
                parsed_data = self.parse_filename(path)
                if parsed_data is None:
                    continue
                patient, scan, slice = parsed_data
                group_images[(patient, scan, label)].append([slice, path])


        for (patient, scan, label), slices in group_images.items():
            slices.sort(key=lambda x: x[0])
            paths = [p for _, p in slices]

            for i in range(len(paths) - n_images + 1):
                self.window.append((paths[i: i + n_images], label))

    def __len__(self) -> int:
        return len(self.window)

    def __getitem__(self, index):
        paths, label = self.window[index]

        frames = []

        for p in paths:
            img = cv2.imread(p,cv2.IMREAD_GRAYSCALE)
            if self.transform is not None:
                img = self.transform(img)

            img = transforms.ToTensor()(img[..., None])
            frames.append(img)

        return torch.stack(frames, dim=0), torch.tensor(label, dtype=torch.long)

    def parse_filename(self, path) -> Optional[Tuple[str, str, int]]:
        m = _FILENAME_RE.match(os.path.basename(path))
        if m is None:
            return None

        return m.group("patient"), m.group("scan"), int(m.group("slice"))
