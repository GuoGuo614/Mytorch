from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip, struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        with gzip.open(image_filename, 'rb') as f:
            image_data = f.read()
        magic, num_images, rows, cols = struct.unpack('>IIII', image_data[:16])

        if magic != 2051:
            return None

        images = np.frombuffer(image_data, dtype=np.uint8, offset=16)
        X = images.reshape(num_images, rows * cols).astype(np.float32) / 255.0
        self.images = X

        with gzip.open(label_filename, 'rb') as f:
            label_data = f.read()
        magic, num_labels = struct.unpack('>II', label_data[:8])

        if magic != 2049:
            return None

        y = np.frombuffer(label_data, dtype=np.uint8, offset=8)
        self.labels = y

        self.transforms = transforms

    def __getitem__(self, index) -> object:
        img = self.images[index]
        return self.apply_transforms(img), self.labels[index]

    def __len__(self) -> int:
        return self.images.shape[0]
