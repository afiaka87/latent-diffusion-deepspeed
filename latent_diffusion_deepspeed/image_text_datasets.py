import io
import math
import os
from posixpath import expanduser
import random
from pathlib import Path

import blobfile as bf
import numpy as np
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def create_dataloader(
    distr_backend,
    data_dir,
    batch_size,
    image_size,
    dataset_length,
    random_crop=False,
    random_flip=False,
    use_webdataset=False,
    num_workers=0,
):
    if use_webdataset:
        wds_urls = parse_data_dir(data_dir)
        ds = load_webdataset(distr_backend, resolution=image_size,
                             file_paths=wds_urls, batch_size=batch_size, random_crop=random_crop, random_flip=random_flip)
        dl = wds.WebLoader(ds, batch_size=None,
                           shuffle=False, num_workers=num_workers)
        number_of_batches = dataset_length // (
            batch_size * distr_backend.get_world_size())
        dl = dl.slice(number_of_batches)
        dl.length = number_of_batches
        print(f"Loaded webdataset with {number_of_batches} batches on {distr_backend.get_world_size()} gpus")
    else:
        ds = ImageDataset(resolution=image_size, file_paths=data_dir,
                          random_crop=random_crop, random_flip=random_flip)
        dl = DataLoader(ds, batch_size=batch_size,
                        shuffle=True, drop_last=True)
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


def clean_caption(caption):
    caption = caption.decode("utf-8")
    caption = caption.replace("\n", " ").replace(
        "\t", " ").replace("\r", " ").replace("  ", " ")
    caption = caption.strip()
    return caption


def load_webdataset(
    distr_backend,
    resolution,
    file_paths,
    batch_size,
    random_crop=False,
    random_flip=False,
):
    def filter_by_item(item):
        if mycap not in item: return False
        if myimg not in item: return False
        return True

    def pil_transform_to_np(arr):
        if random_crop:
            arr = random_crop_arr(arr, resolution)
        else:
            arr = center_crop_arr(arr, resolution)
        if random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]
        arr = arr.astype(np.float32) / 127.5 - 1
        return np.transpose(arr, [2, 0, 1])

    def bytes_to_pil_image(item): 
        pil_image = Image.open(io.BytesIO(item)).convert("RGB")
        pil_image.load()
        return pil_image

    myimg, mycap = "jpg", "txt"
    image_text_mapping = {
        myimg: bytes_to_pil_image,
        mycap: clean_caption
    }
    image_mapping = {myimg: pil_transform_to_np}
    dataset = wds.WebDataset(urls=file_paths,
                             handler=wds.warn_and_continue,
                             cache_dir=expanduser(
                                 "~/.cache/latent-diffusion-webdataset"),
                             cache_size=10**10)
    filtered_dataset = dataset.select(filter_by_item)
    dataset = filtered_dataset.map_dict(**image_text_mapping).map_dict(**image_mapping).to_tuple(
        mycap, myimg).batched(batch_size / distr_backend.get_world_size(), partial=True)
    return dataset


def parse_data_dir(data_dir):
    # quit early if no tar files were found
    if Path(data_dir).is_dir():
        wds_uris = [str(p) for p in Path(data_dir).glob(
            "**/*") if ".tar" in str(p).lower()]  # .name
        assert len(
            wds_uris) > 0, 'The directory ({}) does not contain any WebDataset/.tar files.'.format(data_dir)
        print('Found {} WebDataset .tar(.gz) file(s) under given path {}!'.format(
            len(wds_uris), data_dir))
    elif ('http://' in data_dir.lower()) | ('https://' in data_dir.lower()):
        wds_uris = f"pipe:curl -L -s {data_dir} || true"
        print('Found {} http(s) link under given path!'.format(
            len(wds_uris), data_dir))
    elif 'gs://' in data_dir.lower():
        wds_uris = f"pipe:gsutil cat {data_dir} || true"
        print('Found {} GCS link under given path!'.format(
            len(wds_uris), data_dir))
    elif '.tar' in data_dir:
        wds_uris = data_dir
        print('Found WebDataset .tar(.gz) file under given path {}!'.format(data_dir))
    else:
        raise Exception(
            'No folder, no .tar(.gz) and no url pointing to tar files provided under {}.'.format(data_dir))
    return wds_uris
