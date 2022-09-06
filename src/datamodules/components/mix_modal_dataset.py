from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

# from data.transforms.randaug import RandAugment


class GaussianBlur:
    """blur a single image on CPU."""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(
            3,
            3,
            kernel_size=(kernel_size, 1),
            stride=1,
            padding=0,
            bias=False,
            groups=3,
        )
        self.blur_v = nn.Conv2d(
            3,
            3,
            kernel_size=(1, kernel_size),
            stride=1,
            padding=0,
            bias=False,
            groups=3,
        )
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(nn.ReflectionPad2d(radias), self.blur_h, self.blur_v)

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


class MixModalDataset:
    def __init__(
        self,
        file_path,
        img_root_dir,
        stage="fit",
        image_size=224,
        multimodal=False,
        mlm=False,
        whole_word_mask=False,
        eda=False,
        eda_prob=0.5,
        only_img=False,
    ):
        file_path = Path(file_path)
        self.img_root_dir = Path(img_root_dir) if isinstance(img_root_dir, str) else None
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found!")
        self.data = pd.read_json(file_path, lines=True)
        if only_img:
            self.data = self.data[self.data.img.apply(lambda x: x is not None and len(x) > 0)]
        self.stage = stage
        if multimodal:
            # self.transformer = transforms.Compose(
            #     [
            #         transforms.RandomHorizontalFlip(p=0.2),
            #         transforms.RandomPerspective(p=0.2),
            #         transforms.RandomVerticalFlip(p=0.2),
            #     ]
            # )
            self.transformer = self.get_simclr_pipeline_transform(image_size)
            # self.transformer = RandAugment(2, 9)
        self.multimodal = multimodal
        self.mlm = mlm
        self.whole_word_mask = whole_word_mask
        self.eda = eda
        self.eda_prob = eda_prob
        print("total data:", len(self.data))

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(kernel_size=int(0.1 * size)),
                transforms.ToTensor(),
            ]
        )
        return data_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        text = sample["text"]
        # if use text agumentation
        if self.eda and self.stage == "fit":
            p = np.random.random()
            if p < self.eda_prob:
                text = np.random.choice(sample["eda_text"])
        if not self.mlm:
            label = sample["label"]
            # if use multimodal data
            if self.multimodal:
                img_name = sample["img"]
                if img_name:
                    img = Image.open(self.img_root_dir / img_name)
                    valid_img = True
                else:
                    # dummy image, will be masked during training
                    img = Image.new("RGB", (224, 224))
                    valid_img = False
                return self.transformer(img), text, valid_img, label
            else:
                return text, label
        # if use masked language model
        else:
            if not self.whole_word_mask:
                return text
            else:
                cn_ref = sample["cn_ref"]
                return text, cn_ref
