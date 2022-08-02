from pathlib import Path

import torch
from transformers import BertTokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordPiece


class Processor(object):
    def __init__(self, tokenizer_name, local=False):
        super().__init__()
        self.local = local
        if not local:
            self.tokenizer = BertTokenizer.from_pretrained(
                tokenizer_name, cache_dir=str(Path.home() / ".cache")
            )
        else:
            tokenizer = Tokenizer(WordPiece())
            self.tokenizer = tokenizer.from_file(tokenizer_name)

    def __call__(self, features: torch.Tensor, texts: str):
        texts = list(texts) if isinstance(texts, tuple) else texts
        if self.local:
            batch_encoding = self.tokenizer.encode_batch(texts)
            tokens = torch.tensor(
                [encodings.ids for encodings in batch_encoding], dtype=torch.long
            )
            attention_mask = torch.tensor(
                [encodings.attention_mask for encodings in batch_encoding],
                dtype=torch.float,
            )
            token_type_ids = torch.tensor(
                [encodings.type_ids for encodings in batch_encoding], dtype=torch.long
            )
        else:
            encodings = self.tokenizer(
                texts,
                truncation=True,
                max_length=64,
                padding="max_length",
                return_tensors="pt",
            )
            tokens = encodings.input_ids
            attention_mask = encodings.attention_mask  # [N, L]
            token_type_ids = encodings.token_type_ids  # [N, L]
        # img_mask = torch.ones(batch_size, 1, dtype=torch.float)
        # img_id = torch.ones(batch_size, 1, dtype=torch.long)
        # attention_mask = torch.cat([attention_mask, img_mask], dim=-1)  # [N, L+1]
        # token_type_ids = torch.cat([token_type_ids, img_id], dim=-1)  # [N, L+1]
        return [tokens, features, attention_mask, token_type_ids]

    def get_vocab_size(self):
        return (
            self.tokenizer.vocab_size
            if not self.local
            else self.tokenizer.get_vocab_size()
        )

    def get_padding_idx(self):
        return (
            self.tokenizer.pad_token_id
            if not self.local
            else self.tokenizer.token_to_id("[PAD]")
        )
        
    def get_tokenizer(self):
        return self.tokenizer


if __name__ == "__main__":
    texts = [
        "耐磨2021年冬季运动帆布鞋低帮白绿系带",
        "高领修身型加厚女装大红色针织衫纯色",
        "长袖绿色男装圆领纯色套头针织衫",
        "2021年冬季长袖打底衫套头咖色女装",
        "纯色精品男包2021年冬季斜挎包",
        "拉链直筒裤黑色蓝色毛衣微弹加绒裤",
        "圆领短袖咖啡色宽松型女装纯色上衣短款",
        "九分袖女装连衣裙2020年夏季",
    ]
    processor = Processor("tokenizer/my_tokenizer_8000.json", local=True)
    batch_encoding = processor.tokenizer.encode_batch(texts)
    tokens = torch.tensor(
        [encodings.ids for encodings in batch_encoding], dtype=torch.long
    )
    attention_mask = torch.tensor(
        [encodings.attention_mask for encodings in batch_encoding], dtype=torch.float,
    )
    token_type_ids = torch.tensor(
        [encodings.type_ids for encodings in batch_encoding], dtype=torch.long
    )
