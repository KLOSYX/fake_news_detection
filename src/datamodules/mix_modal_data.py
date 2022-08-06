import random
from typing import Any, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoFeatureExtractor, AutoTokenizer

from datamodules.components.mix_modal_dataset import MixModalDataset


class Processor:
    def __init__(
        self,
        img_processor,
        text_tokenizer,
        max_length=256,
        mlm=False,
        whole_word_mask=False,
        mlm_probability=0.15,
    ) -> None:
        self.img_processor = img_processor
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length
        self.mlm = mlm
        self.whole_word_mask = whole_word_mask
        self.mlm_probability = mlm_probability

    def __call__(self, data: list) -> Any:
        if not self.mlm:
            if self.img_processor is not None:  # multimodal
                imgs, texts, valid_imgs, labels = zip(*data)
                pixels = self.img_processor(imgs, return_tensors="pt")
                tokens = self.text_tokenizer(
                    list(texts),
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                )
                valid_imgs = torch.tensor(valid_imgs, dtype=torch.float)
                labels = torch.argmax(torch.tensor(labels, dtype=torch.long), dim=1)
                return pixels, tokens, valid_imgs, labels
            else:
                texts, labels = zip(*data)
                tokens = self.text_tokenizer(
                    list(texts),
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                )
                labels = torch.argmax(torch.tensor(labels, dtype=torch.long), dim=1)
                return tokens, labels
        else:
            if not self.whole_word_mask:
                tokens = self.text_tokenizer(
                    data,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                )
                tokens["input_ids"], tokens["labels"] = self._torch_mask_tokens(tokens.input_ids)
                return tokens
            else:
                # whole word mask
                text, cn_refs = zip(*data)
                tokens = self.text_tokenizer(
                    list(text),
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                )
                mask_labels = [
                    self._whole_word_mask(self._handle_chinese_ref(tokens.tokens(i), cn_refs[i]))
                    for i in range(len(text))
                ]
                mask_labels = torch.tensor(mask_labels, dtype=torch.bool)
                labels = tokens.input_ids.masked_fill(~mask_labels, -100)
                tokens["input_ids"], tokens["labels"] = self._torch_mask_tokens(
                    tokens.input_ids, None, labels
                )
                return tokens

    def _handle_chinese_ref(self, tokens: List[str], cn_ref: List[int]):
        # For Chinese tokens, we need extra inf to mark sub-word, e.g [喜,欢]-> [喜，##欢]
        for i in range(len(tokens)):
            if i in cn_ref:
                tokens[i] = "##" + tokens[i]
        return tokens

    def _torch_mask_tokens(
        self, inputs: Any, special_tokens_mask: Optional[Any] = None, labels: torch.Tensor = None
    ) -> Tuple[Any, Any]:
        """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random,
        10% original."""
        if labels is None:
            labels = inputs.clone()
            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.text_tokenizer.get_special_tokens_mask(
                        val, already_has_special_tokens=True
                    )
                    for val in labels.tolist()
                ]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            else:
                special_tokens_mask = special_tokens_mask.bool()

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            # We only compute loss on masked tokens
            labels[~masked_indices] = -100

        else:
            masked_indices = labels != -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.text_tokenizer.convert_tokens_to_ids(
            self.text_tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.text_tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """Get 0/1 labels for masked tokens with whole word mask proxy."""
        cand_indexes = []
        num_valid_token = 0
        for i, token in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]" or token == "[PAD]":
                continue
            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
                num_valid_token += 1
            else:
                cand_indexes.append([i])
                num_valid_token += 1

        random.shuffle(cand_indexes)
        # num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        num_to_predict = min(
            max_predictions, max(1, int(round(num_valid_token * self.mlm_probability)))
        )
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels


class MixModalData(pl.LightningDataModule):
    def __init__(
        self,
        visual_encoder="openai/clip-vit-base-patch32",
        tokenizer_name="hfl/chinese-roberta-wwm-ext",
        batch_size_per_gpu=4,
        num_workers=0,
        max_length=256,
        multimodal=False,
        mlm=False,
        whole_word_mask=False,
        val_ratio=0.1,
        eda=False,
        eda_prob=0.5,
        only_img=False,
        **args
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.visual_processor = (
            AutoFeatureExtractor.from_pretrained(
                self.hparams.visual_encoder, cache_dir="/data/.cache"
            )
            if multimodal
            else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.tokenizer_name, cache_dir="/data/.cache"
        )
        self.processor = Processor(
            self.visual_processor,
            self.tokenizer,
            max_length=max_length,
            mlm=mlm,
            whole_word_mask=whole_word_mask,
        )

    def setup(self, stage: str = None) -> None:
        if stage is None or stage == "fit":
            data = MixModalDataset(
                self.hparams.train_path,
                stage=stage,
                multimodal=self.visual_processor is not None,
                mlm=self.hparams.mlm,
                whole_word_mask=self.hparams.whole_word_mask,
                eda=self.hparams.eda,
                eda_prob=self.hparams.eda_prob,
                only_img=self.hparams.only_img,
            )
            data_size = len(data)
            val_size = int(self.hparams.val_ratio * data_size)
            train_size = data_size - val_size
            print("train_size:", train_size, "\nval_size:", val_size)
            self.train_dataset, self.valid_dataset = random_split(data, [train_size, val_size])
            # change stage to 'val', avoid applying eda on val set
            self.valid_dataset.stage = "val"

        if stage is None or stage == "test":
            self.test_dataset = MixModalDataset(
                self.hparams.test_path,
                stage=stage,
                multimodal=self.visual_processor is not None,
                mlm=self.hparams.mlm,
                whole_word_mask=self.hparams.whole_word_mask,
                eda=self.hparams.eda,
                eda_prob=self.hparams.eda_prob,
                only_img=self.hparams.only_img,
            )

    def train_dataloader(self) -> Any:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size_per_gpu,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=self.processor,
            drop_last=True,
        )

    def val_dataloader(self) -> Any:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.batch_size_per_gpu,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=self.processor,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size_per_gpu,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=self.processor,
        )


if __name__ == "__main__":
    # debug
    dm = MixModalData(
        batch_size_per_gpu=4,
        num_workers=0,
        train_path="/data/clean_raw_text/street_labeled_data.json",
        test_path="/data/clean_raw_text/street_labeled_data.json",
        tokenizer_name="IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese",
        multimodal=True,
        mlm=False,
        whole_word_mask=False,
        only_img=True,
    )
    dm.setup("fit")
    dataloader = dm.train_dataloader()
    it = iter(dataloader)
    while True:
        item = next(it)
