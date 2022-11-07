from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoFeatureExtractor, AutoTokenizer

from src.datamodules.components.dm_base import DatamoduleBase
from src.models.components.blip_base import init_tokenizer
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


DATASETS = [
    "weibo",
    "twitter",
    "twitter_with_event",
    "weibo_with_event",
]

VISUAL_MODEL_TYPES = ["vgg", "blip", "other"]


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


class WeiboDataset(Dataset):
    def __init__(
        self,
        img_path: Union[str, Path],
        data_path: Union[str, Path],
        transforms: Optional[transforms.Compose],
    ) -> None:
        assert img_path and data_path, "img_path and data_path must be provided"
        self.data = pd.read_json(data_path, lines=True)
        self.img_path = Path(img_path)
        # apply image transforms
        self.transforms = transforms
        print("data size:", len(self.data))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        text = self.data.iloc[idx]["text"]
        img_name = self.data.iloc[idx]["imgs"]
        img_path = self.img_path / img_name
        img = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx]["label"]
        return (
            text,
            self.transforms(img).unsqueeze(0) if self.transforms else img,
            torch.tensor([label], dtype=torch.long),
        )


class TwitterDataset(WeiboDataset):
    pass


class TwitterDatasetWithEvent(TwitterDataset):
    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        text = self.data.iloc[idx]["text"]
        img_name = self.data.iloc[idx]["imgs"]
        img_path = self.img_path / img_name
        img = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx]["label"]
        event_label = self.data.iloc[idx]["event"]
        return (
            text,
            self.transforms(img).unsqueeze(0) if self.transforms else img,
            torch.tensor([label], dtype=torch.long),
            torch.tensor([event_label], dtype=torch.long),
        )


class Collector:
    def __init__(self, tokenizer: Any, processor: Optional[Any], max_length: int = 200) -> None:
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length

    def __call__(self, data: List) -> Tuple:
        texts, imgs, labels = zip(*data)
        text_encodeds = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        img_encodeds = (
            self.processor(imgs, return_tensors="pt")
            if self.processor is not None
            else torch.cat(imgs, dim=0)
        )
        # image agumentation here, todo...
        labels = torch.cat(labels)
        return text_encodeds, img_encodeds, labels


class CollectorWithEvent(Collector):
    def __call__(self, data: List) -> Tuple:
        texts, imgs, labels, event_labels = zip(*data)
        text_encodeds = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        img_encodeds = (
            self.processor(imgs, return_tensors="pt")
            if self.processor is not None
            else torch.cat(imgs, dim=0)
        )
        # image agumentation here, todo...
        labels = torch.cat(labels)
        event_labels = torch.cat(event_labels)
        return text_encodeds, img_encodeds, labels, event_labels


class MultiModalData(DatamoduleBase):
    def __init__(
        self,
        img_path: str,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        val_set_ratio: float = 0.2,
        batch_size: int = 8,
        num_workers: int = 0,
        tokenizer_name: str = "bert-base-chinese",
        processor_name: Optional[str] = None,
        max_length: int = 200,
        dataset_name: str = "weibo",
        use_test_as_val: bool = False,
        simclr_trans: bool = False,
        vis_model_type: str = "vgg",
    ):
        assert dataset_name in DATASETS
        assert vis_model_type in VISUAL_MODEL_TYPES
        super().__init__(val_set_ratio, batch_size, num_workers)
        self.img_path = img_path
        self.train_path = train_path
        self.test_path = test_path
        if vis_model_type == "blip":
            self.tokenizer = init_tokenizer()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, cache_dir=Path.home() / ".cache"
            )
        self.processor = (
            AutoFeatureExtractor.from_pretrained(processor_name, cache_dir=Path.home() / ".cache")
            if processor_name
            else None
        )
        self.max_length = max_length
        self.dataset_name = dataset_name
        self.use_test_as_val = use_test_as_val
        self.simclr_trans = simclr_trans
        self.vis_model_type = vis_model_type
        if dataset_name == "weibo":
            self.dataset_cls = WeiboDataset
        elif dataset_name == "twitter":
            self.dataset_cls = TwitterDataset
        elif dataset_name == "twitter_with_event":
            self.dataset_cls = TwitterDatasetWithEvent

    def setup(self, stage: Optional[str] = None) -> None:
        self.collector = self._get_collector()
        if stage == "fit" or stage is None:
            dataset = self._get_dataset(stage)
            if not self.use_test_as_val:
                val_size = max(int(len(dataset) * self.val_ratio), 1)
                train_size = len(dataset) - val_size
                assert (
                    train_size >= 1 and val_size >= 1
                ), "Train size or val size is smaller than 1!"
                print("Train size", train_size, "Val size", val_size)
                self.train_data, self.val_data = random_split(dataset, [train_size, val_size])
            else:
                self.train_data = dataset
                self.val_data = self._get_dataset("test")
                print("Train size", len(self.train_data), "Val size", len(self.val_data))

        if stage == "test" or stage is None:
            self.test_data = self._get_dataset(stage)
            print("Test size", len(self.test_data))

    def _get_collector(self) -> Any:
        params = dict(
            tokenizer=self.tokenizer,
            processor=self.processor,
            max_length=self.max_length,
        )
        if "event" not in self.dataset_name:
            log.debug(f"Using Collector for {self.dataset_name}")
            return Collector(**params)
        else:
            log.debug(f"Using CollectorWithEvent for {self.dataset_name}")
            return CollectorWithEvent(**params)

    def _get_dataset(self, stage: Optional[str] = "fit") -> Dataset:
        params = dict(
            img_path=self.img_path,
            data_path=self.train_path if stage == "fit" or stage is None else self.test_path,
            transforms=self._get_transforms(self.vis_model_type),
        )
        if self.simclr_trans:
            log.debug(f"Using simclr transform in {self.dataset_name}")
            params["transforms"] = self._get_simclr_pipeline_transform(224)
        else:
            log.debug(f"Not using simclr transform in {self.dataset_name}")
        return self.dataset_cls(**params)

    @staticmethod
    def _get_transforms(vis_model_type: str = "vgg", image_size: int = 224) -> transforms.Compose:
        trans = None
        if vis_model_type == "vgg":
            log.debug("Using vgg transform")
            trans = transforms.Compose(
                [
                    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        elif vis_model_type == "blip":
            log.debug("Using blip transform")
            trans = transforms.Compose(
                [
                    transforms.Resize(
                        (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                    ),
                ]
            )
        elif vis_model_type == "other":
            log.debug("Using default transform")
        return trans

    @staticmethod
    def _get_simclr_pipeline_transform(size, s=1):
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
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return data_transforms


# debug
if __name__ == "__main__":
    data = MultiModalData(
        img_path="/data/fake_news/datasets/MM17-WeiboRumorSet/images",
        train_path="/data/fake_news/datasets/MM17-WeiboRumorSet/train_data.json",
        test_path="/data/fake_news/datasets/MM17-WeiboRumorSet/test_data.json",
    )
    data.setup("fit")
    it = iter(data.train_dataloader())
    while True:
        item = next(it)
