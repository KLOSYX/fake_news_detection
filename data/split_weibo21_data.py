import pandas as pd
import pyrootutils

root = pyrootutils.setup_root(".", pythonpath=True)

from src.utils.data_util import clean_text


def main(args):
    # read data
    all_data = pd.read_json(args.input_file, lines=True)
    # keep sample with images
    all_data = all_data[all_data.imgs.apply(len) > 0]
    all_data["text"] = all_data.content.astype(str).apply(clean_text)
    all_data = all_data[["text", "imgs", "label", "split_comments"]]
    # split data
    if not args.time_order:
        all_data = all_data.sample(frac=1, random_state=args.seed)
    # split with ratio 0.7, 0.1, 0.2
    train_data = all_data[: len(all_data) // 10 * 7]
    val_data = all_data[len(all_data) // 10 * 7 : len(all_data) // 10 * 8]
    test_data = all_data[len(all_data) // 10 * 8 :]

    print("train data size: ", len(train_data))
    print("val data size: ", len(val_data))
    print("test data size: ", len(test_data))

    train_data.to_json(
        root / "data" / "weibo21/train_data.json", lines=True, orient="records", force_ascii=False
    )
    val_data.to_json(
        root / "data" / "weibo21/val_data.json", lines=True, orient="records", force_ascii=False
    )
    test_data.to_json(
        root / "data" / "weibo21/test_data.json", lines=True, orient="records", force_ascii=False
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/weibo21/all_data_cleaned.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time_order", action="store_true")

    args = parser.parse_args()
    main(args)
