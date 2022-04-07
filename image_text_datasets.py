import io
import json
import webdataset as wds
import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset

DATASET_SIZE = 100000

def load_data(
    data_dir,
    batch_size,
    image_size,
    random_crop=False,
    random_flip=True,
    use_webdataset=False,
):
    if use_webdataset:
        ds = load_webdataset(
            resolution=image_size,
            file_paths=data_dir,
            random_crop=random_crop,
            random_flip=random_flip,
        )
        dl = DataLoader(ds, batch_size=batch_size)
    else:
        ds = ImageDataset(
            resolution=image_size,
            file_paths=data_dir,
            random_crop=random_crop,
            random_flip=random_flip,
        )
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    while True:
        yield from dl


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        entry = entry.split(".")
        ext = entry[-1].strip()
        filename = entry[0]
        if ext and ext.lower() in ["jpg", "jpeg", "png", "gif", "webp"]:
            text_path = bf.join(data_dir, filename+'.txt')
            if bf.exists(text_path):
                results.append((full_path, text_path))
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        file_paths,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_files = file_paths
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_files)

    def __getitem__(self, idx):
        path = self.local_files[idx]
        with bf.BlobFile(path[0], "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        with bf.BlobFile(path[1], "r") as f:
            text = f.read().strip()

        return np.transpose(arr, [2, 0, 1]), out_dict, text


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(
        min_smaller_dim_size, max_smaller_dim_size + 1)

    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def load_webdataset(
    resolution,
    file_paths,
    random_crop=False,
    random_flip=True,
):
    dataset = wds.WebDataset(
        file_paths,
        cache_dir=None,
        cache_size=10**10,
        handler=wds.handlers.warn_and_stop,
    )

    def filter_dataset_laion(item):
        if "txt" not in item:
            return False
        if "jpg" not in item:
            return False
        return True

    filtered_dataset = dataset.select(filter_dataset_laion)

    def preprocess_dataset(item):
        image_data = item["jpg"]
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        if random_crop:
            arr = random_crop_arr(pil_image, resolution)
        else:
            arr = center_crop_arr(pil_image, resolution)
        if random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]
        arr = arr.astype(np.float32) / 127.5 - 1
        caption = item["txt"].decode("utf-8").strip()
        return np.transpose(arr, [2, 0, 1]), {}, caption

    transformed_dataset = filtered_dataset.map(
        preprocess_dataset, handler=wds.handlers.warn_and_stop)
    return transformed_dataset
