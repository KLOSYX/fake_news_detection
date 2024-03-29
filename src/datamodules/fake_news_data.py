from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    BatchEncoding,
    BatchFeature,
    BertTokenizer,
)

from src.datamodules.components.dm_base import DatamoduleBase
from src.models.components.blip_base import init_tokenizer
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

DATASETS = [
    "weibo",
    "weibo21",
    "twitter",
    "twitter_with_event",
    "weibo_with_event",
    "weibo_kb",
    "twitter_kb",
]

VISUAL_MODEL_TYPES = ["vgg", "blip", "other"]


@dataclass(frozen=True)
class RawData:
    """Raw data class.

    Args:
        text: text data
        image: image data, could be None, Image.Image, or Tensor
        label: label data, could be None, int, or List[int]
        event: event data, could be None, int, or List[int]
        kb_annotations: knowledge base annotations, could be None or List[Dict]]

    Returns:
        RawData: RawData object
    """

    text: str | None = None
    image: Image.Image | torch.Tensor | None = None
    label: torch.Tensor | None = None
    event: torch.Tensor | None = None
    kb_annotations: list[dict] | None = None


@dataclass
class FakeNewsItem:
    """Fake news item class.

    Args:
        text_encoded: encoded text data
        image_encoded: encoded image data
        label: label data
        event_label: event label data

    Returns:
        FakeNewsItem: FakeNewsItem object
    """

    text_encoded: BatchEncoding | None = None
    image_encoded: BatchFeature | torch.Tensor | None = None
    label: torch.Tensor | None = None
    event_label: torch.Tensor | None = None

    def to(self, device: str | torch.device) -> "FakeNewsItem":
        """Move tensors to device."""
        if self.text_encoded is not None:
            self.text_encoded = self.text_encoded.to(device)
        if self.image_encoded is not None:
            self.image_encoded = self.image_encoded.to(device)
        if self.label is not None:
            self.label = self.label.to(device)
        if self.event_label is not None:
            self.event_label = self.event_label.to(device)
        return self


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
        img_path: str | Path,
        data_path: str | Path,
        transforms: transforms.Compose | None,
    ) -> None:
        assert img_path and data_path, "img_path and data_path must be provided"
        self.data = pd.read_json(data_path, lines=True)
        self.img_path = Path(img_path)
        # apply image transforms
        self.transforms = transforms
        log.info("Data size: %d" % len(self.data))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> RawData:
        text = self.data.iloc[idx]["text"]
        img_name = self.data.iloc[idx]["imgs"]
        img_path = self.img_path / img_name
        img = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx]["label"]
        return RawData(
            text,
            self.transforms(img).unsqueeze(0) if self.transforms else img,
            torch.tensor([label], dtype=torch.long),
        )


class Weibo21Dataset(WeiboDataset):
    pass


class TwitterDataset(WeiboDataset):
    pass


class TwitterDatasetWithEvent(TwitterDataset):
    def __getitem__(self, idx: int) -> RawData:
        text = self.data.iloc[idx]["text"]
        img_name = self.data.iloc[idx]["imgs"]
        img_path = self.img_path / img_name
        img = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx]["label"]
        event_label = self.data.iloc[idx]["event"]
        return RawData(
            text,
            self.transforms(img).unsqueeze(0) if self.transforms else img,
            torch.tensor([label], dtype=torch.long),
            torch.tensor([event_label], dtype=torch.long),
        )


class WeiboDatasetWithEvent(WeiboDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data["event_label"] = np.argmax(
            pd.get_dummies(self.data.event_label).to_numpy(), axis=1
        )

    def __getitem__(self, idx: int) -> RawData:
        text = self.data.iloc[idx]["text"]
        img_name = self.data.iloc[idx]["imgs"]
        img_path = self.img_path / img_name
        img = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx]["label"]
        event_label = self.data.iloc[idx]["event_label"]
        return RawData(
            text,
            self.transforms(img).unsqueeze(0) if self.transforms else img,
            torch.tensor([label], dtype=torch.long),
            torch.tensor([event_label], dtype=torch.long),
        )


class WeiboDatasetKB(WeiboDataset):
    def __init__(
        self,
        img_path: str | Path,
        data_path: str | Path,
        w2v_path: str | Path,
        transforms: transforms.Compose | None,
    ) -> None:
        super().__init__(img_path, data_path, transforms)
        self.w2v = np.load(w2v_path)
        # self.w2v = torch.from_numpy(self.w2v).float()

    def __getitem__(self, idx) -> RawData:
        text = self.data.iloc[idx]["text"]
        img_name = self.data.iloc[idx]["imgs"]
        img_path = self.img_path / img_name
        img = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx]["label"]
        annotations: list[dict] = self.data.iloc[idx]["wiki_annotations"]
        for a in annotations:
            a["vecs"] = torch.from_numpy(self.w2v[a["ids"]])
            a.pop("ids")
        return RawData(
            text,
            self.transforms(img).unsqueeze(0) if self.transforms else img,
            torch.tensor([label], dtype=torch.long),
            None,
            annotations,
        )


class TwitterDatasetKB(WeiboDatasetKB):
    pass


class Collector:
    def __init__(self, tokenizer: Any, processor: Any | None, max_length: int = 200) -> None:
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length

    def __call__(self, data: list[RawData]) -> FakeNewsItem:
        texts, imgs, labels = [], [], []
        for d in data:
            texts.append(d.text)
            imgs.append(d.image)
            labels.append(d.label)
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
        return FakeNewsItem(text_encodeds, img_encodeds, labels)


class CollectorKB(Collector):
    def __call__(self, data: list[RawData]) -> FakeNewsItem:
        texts, imgs, labels, annotations = [], [], [], []
        for d in data:
            texts.append(d.text)
            imgs.append(d.image)
            labels.append(d.label)
            annotations.append(d.kb_annotations)
        text_encodeds = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        img_encodeds = (
            self.processor(imgs, return_tensors="pt")
            if self.processor is not None
            else torch.cat(imgs, dim=0)
        )
        labels = torch.cat(labels)
        true_annotation_nums = [len(a) for a in annotations]
        embeddings = [x["vecs"] for x in chain.from_iterable(annotations) if x]
        true_desc_lengths: torch.Tensor = torch.Tensor([e.shape[0] for e in embeddings]).to(
            torch.long
        )
        embeddings = pad_sequence(
            embeddings, batch_first=True, padding_value=0
        )  # (total_entities, max_desc_length, 300)
        # manually set max description length
        embeddings = embeddings[:, :128, :]
        true_desc_lengths = torch.clamp(true_desc_lengths, max=128)
        assert embeddings.shape[0] == sum(true_annotation_nums) == true_desc_lengths.size(0)

        cross_atts = self._get_cross_attention_mask(text_encodeds, annotations)

        text_encodeds.update(
            {
                "kb_embeddings": embeddings,
                "kb_true_annotation_nums": torch.Tensor(true_annotation_nums).to(torch.long),
                "kb_cross_atts": cross_atts.to(torch.long),
                "kb_true_desc_lengths": true_desc_lengths,
            }
        )

        return FakeNewsItem(text_encodeds, img_encodeds, labels)

    @staticmethod
    def _get_cross_attention_mask(text_encodeds, annotations) -> torch.Tensor:
        atts = torch.zeros(text_encodeds.input_ids.size(0), text_encodeds.input_ids.size(1), 32)
        entity_map_list: list[dict[int, int]] = []
        for ann in annotations:
            entity_map = {}
            for i, a in enumerate(ann):
                for n in range(a["start"], a["end"]):
                    entity_map[n] = i
            entity_map_list.append(entity_map)
        for b, m in enumerate(text_encodeds.offset_mapping.tolist()):
            for t, n in enumerate(m):
                if n[0] == 0 and n[1] == 0:
                    continue
                entity_id = entity_map_list[b].get(n[0], -1)
                if entity_id != -1:
                    atts[b, t, entity_id] = 1
        return atts


class CollectorWithEvent(Collector):
    def __call__(self, data: list[RawData]) -> FakeNewsItem:
        texts, imgs, labels, event_labels = [], [], [], []
        for d in data:
            texts.append(d.text)
            imgs.append(d.image)
            labels.append(d.label)
            event_labels.append(d.event)
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
        return FakeNewsItem(text_encodeds, img_encodeds, labels, event_labels)


class MultiModalData(DatamoduleBase):
    def __init__(
        self,
        img_path: str,
        train_path: str | None = None,
        val_path: str | None = None,
        test_path: str | None = None,
        w2v_path: str | None = None,
        val_set_ratio: float = 0.2,
        batch_size: int = 8,
        num_workers: int = 0,
        tokenizer_name: str = "bert-base-chinese",
        processor_name: str | None = None,
        max_length: int = 200,
        dataset_name: str = "weibo",
        use_test_as_val: bool = False,
        simclr_trans: bool = False,
        vis_model_type: str = "vgg",
    ):
        """
        Args:
            img_path: path to image folder
            train_path: path to train jsonl file
            val_path: path to val jsonl file. If None, val_set_ratio will be used to split train set
            test_path: path to test jsonl file
            w2v_path: path to w2v file. Now only used when dataset contains knowledge
            val_set_ratio: ratio of val set in train set
            batch_size: batch size
            num_workers: num of workers for dataloader
            tokenizer_name: name of tokenizer
            processor_name: name of processor. Used by some ViT based models
            max_length: max length of text
            dataset_name: name of dataset. Used to determine which dataset to use
            use_test_as_val: whether to use test set as val set. If True, val_path will be ignored
            simclr_trans: whether to use simclr transformation
            vis_model_type: type of visual model. Used to determine which image processor to use
        """
        assert vis_model_type in VISUAL_MODEL_TYPES
        super().__init__(val_set_ratio, batch_size, num_workers)
        self.dataset_cls = self._get_dataset_class(dataset_name)
        self.img_path = img_path
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.w2v_path = w2v_path
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

    def setup(self, stage: str | None = None) -> None:
        self.collector = self._get_collector()
        if stage == "fit" or stage is None:
            dataset = self._get_dataset(stage)
            if not self.use_test_as_val:
                # val set is either from train data or val data
                if self.val_path is None:
                    # val set is from train data
                    val_size = max(int(len(dataset) * self.val_ratio), 1)
                    train_size = len(dataset) - val_size
                    assert (
                        train_size >= 1 and val_size >= 1
                    ), "Train size or val size is smaller than 1!"
                    self.train_data, self.val_data = random_split(dataset, [train_size, val_size])
                else:
                    # val set is from val data
                    self.train_data = self._get_dataset("fit")
                    self.val_data = self._get_dataset("val")
            else:
                # val set is from test data
                self.train_data = dataset
                self.val_data = self._get_dataset("test")

            log.info("Train size: %d, Val size: %d", len(self.train_data), len(self.val_data))

        if stage == "test" or stage is None:
            # test set is not ready until stage is test
            self.test_data = self._get_dataset(stage)
            log.info("Test size: %d" % len(self.test_data))

    def _get_collector(self) -> Any:
        """Get collector for dataset.

        Adjust to different dataset name.
        """
        params = dict(
            tokenizer=self.tokenizer,
            processor=self.processor,
            max_length=self.max_length,
        )
        if "event" in self.dataset_name:
            log.debug(f"Using CollectorWithEvent for {self.dataset_name}")
            return CollectorWithEvent(**params)
        elif "kb" in self.dataset_name:
            log.debug(f"Using CollectorKB for {self.dataset_name}")
            return CollectorKB(**params)
        else:
            log.debug(f"Using Collector for {self.dataset_name}")
            return Collector(**params)

    @staticmethod
    def _get_dataset_class(dataset_name: str) -> Any:
        assert dataset_name in DATASETS
        dataset_cls = WeiboDataset
        if dataset_name == "weibo21":
            dataset_cls = Weibo21Dataset
        elif dataset_name == "twitter":
            dataset_cls = TwitterDataset
        elif dataset_name == "weibo_with_event":
            dataset_cls = WeiboDatasetWithEvent
        elif dataset_name == "twitter_with_event":
            dataset_cls = TwitterDatasetWithEvent
        elif dataset_name == "weibo_kb":
            dataset_cls = WeiboDatasetKB
        elif dataset_name == "twitter_kb":
            dataset_cls = TwitterDatasetKB
        return dataset_cls

    def _get_dataset(self, stage: str | None = "fit") -> Dataset:
        # get dataset path
        if stage == "fit":
            data_path = self.train_path
        elif stage == "val":
            data_path = self.val_path
        else:
            data_path = self.test_path

        # set up initial params
        params = dict(
            img_path=self.img_path,
            data_path=data_path,
            transforms=self._get_transforms(self.vis_model_type),
        )

        # add params for different dataset
        if self.simclr_trans:
            log.debug(f"Using simclr transform in {self.dataset_name}")
            params["transforms"] = self._get_simclr_pipeline_transform(224)
        else:
            log.debug(f"Not using simclr transform in {self.dataset_name}")

        if "kb" in self.dataset_name:
            params["w2v_path"] = self.w2v_path

        return self.dataset_cls(**params)

    @staticmethod
    def _get_transforms(vis_model_type: str = "vgg", image_size: int = 224) -> transforms.Compose:
        """Get transforms for different visual model type."""
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
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "weibo.yaml")
    with omegaconf.open_dict(cfg):
        cfg[
            "train_path"
        ] = "/home/anbinx/develop/notebooks/wiki_entity_link/weibo_train_data_wiki.json"
        cfg["w2v_path"] = "/home/anbinx/develop/notebooks/wiki_entity_link/wiki_desc_vec.npy"
        cfg["img_path"] = root / "data" / "MM17-WeiboRumorSet" / "images_filtered"
        cfg[
            "test_path"
        ] = "/home/anbinx/develop/notebooks/wiki_entity_link/weibo_test_data_wiki.json"
        cfg["dataset_name"] = "weibo_kb"
    # cfg.data_dir = str(root / "data")
    dm = hydra.utils.instantiate(cfg)
    dm.setup("fit")
    dataloader = dm.train_dataloader()
    item = next(iter(dataloader))

    pass
