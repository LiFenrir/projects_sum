import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from PIL import Image
import torchvision.transforms as T


class SAM3TextDataset(Dataset):

    def __init__(
        self,
        root,
        annotation_file,
        image_size=1024
    ):

        self.root = Path(root)

        with open(
            annotation_file,
            "r",
            encoding="utf-8"
        ) as f:
            self.annotations = json.load(f)


        self.image_transform = T.Compose(
            [
                T.Resize(
                    (image_size, image_size)
                ),

                T.ToTensor(),

                T.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406
                    ],

                    std=[
                        0.229,
                        0.224,
                        0.225
                    ]
                )
            ]
        )


        self.mask_transform = T.Compose(
            [
                T.Resize(
                    (image_size, image_size),
                    interpolation=T.InterpolationMode.NEAREST
                ),

                T.PILToTensor()
            ]
        )


    def __len__(self):

        return len(self.annotations)



    def __getitem__(self, idx):

        item = self.annotations[idx]


        image_path = (
            self.root /
            "images" /
            item["image"]
        )


        mask_path = (
            self.root /
            item["mask"]
        )


        image = Image.open(
            image_path
        ).convert("RGB")


        mask = Image.open(
            mask_path
        ).convert("L")


        image = self.image_transform(
            image
        )


        mask = self.mask_transform(
            mask
        )


        mask = (
            mask.float()
            /
            255.0
        )


        mask = mask.squeeze(0)


        return {

            "image":
                image,

            "text":
                item["text"],

            "mask":
                mask
        }