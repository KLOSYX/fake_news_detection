import pickle
from pathlib import Path
from typing import Dict, List

import pandas
import pandas as pd
import pyrootutils
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

root = pyrootutils.setup_root(".", pythonpath=True)

from scripts.genre.trie import Trie


def main(args):
    with open(root / "data" / "kilt_titles_trie_dict.pkl", "rb") as f:
        trie = Trie.load_from_dict(pickle.load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    dfs: list[pd.DataFrame] = []
    for f_name in args.in_files:
        df = pandas.read_json(f_name, lines=True)
        dfs.append(df)
    data = pandas.concat(dfs)

    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/genre-kilt", cache_dir=Path.home() / ".cache"
    )
    model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/genre-kilt", cache_dir=Path.home() / ".cache"
        )
        .eval()
        .to(device)
    )

    outputs: list[dict[str, str]] = []

    for i, row in tqdm(data.iterrows(), desc="Generating titles", total=len(data)):
        docs = model.generate(
            **tokenizer(row.text, return_tensors="pt").to(device),
            num_beams=args.n_beams,
            num_return_sequences=args.n_seqs,
            # OPTIONAL: use constrained beam search
            prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
        )
        outputs.append(
            {
                "text": row.text,
                "docs": tokenizer.batch_decode(docs, skip_special_tokens=True),
            }
        )

    pandas.DataFrame(outputs).to_json(
        args.out_file, orient="records", lines=True, force_ascii=False
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_files", "-i", nargs="+", help="Input files")
    parser.add_argument("--out_file", "-o", help="Output file")
    parser.add_argument("--n_beams", "-n", type=int, default=5, help="Number of beams")
    parser.add_argument("--n_seqs", "-s", type=int, default=5, help="Number of return sequences")
    args = parser.parse_args()
    main(args)
