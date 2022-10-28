import json
from pathlib import Path

import pandas as pd
import pyrootutils
from PIL import Image
from tqdm import tqdm

root = pyrootutils.setup_root(".", pythonpath=True)

from src.utils.data_util import clean_text

pd.options.mode.chained_assignment = None  # default='warn'


def refine_images(root_dir: Path, path: Path):
    if not root_dir.exists():
        root_dir.mkdir(parents=True)
    try:
        # image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = Image.open(str(path)).convert("RGB")
        if min(image.size) < 100:
            return ""
        else:
            save_path = root_dir / (path.stem + path.suffix)
            with save_path.open("wb") as f:
                #     pickle.dump(image, f)
                image.save(f)
    except Exception as e:
        print(e)
        save_path = ""
    return str(save_path)


def read_data(path):
    data = []
    with open(path, encoding="utf-8") as infile:
        for i, line in enumerate(infile):
            data.append(list(line.strip().split("|")))
    return data


def check_valid_image(img_root_dir: Path, img_path_list: list):
    valid_img_list = []
    for url in img_path_list:
        img_name = url.split("/")[-1]
        real_path = img_root_dir / img_name
        if real_path.exists():
            valid_img_list.append(img_name)
        else:
            print(f"{img_name} not exists")
    return valid_img_list


def main(lang: str = "cn", is_refine_imgs: bool = True, is_format_posts: bool = True):
    assert lang in ["cn", "en"], "lang must be cn or en."
    save_path = root / "data/MM17-WeiboRumorSet/images_filtered"
    # refine images
    if is_refine_imgs:
        target_directories = [
            root / "data/MM17-WeiboRumorSet/nonrumor_images",
            root / "data/MM17-WeiboRumorSet/rumor_images",
        ]
        old2new = {}
        for directory in target_directories:
            directory = Path(directory)
            for path in tqdm(directory.glob("*"), desc=f"refine images in {directory}"):
                # if path.suffix in [".gif"]:
                #     continue
                old2new[str(path)] = refine_images(save_path, path)
    # with open("old2new.txt", "w") as f:
    #     for old, new in old2new.items():
    #         f.write(f"{old} {new}\n")

    # ===== format posts =====

    if is_format_posts:
        file_list = [
            root / "data/MM17-WeiboRumorSet/tweets/test_nonrumor.txt",
            root / "data/MM17-WeiboRumorSet/tweets/test_rumor.txt",
            root / "data/MM17-WeiboRumorSet/tweets/train_nonrumor.txt",
            root / "data/MM17-WeiboRumorSet/tweets/train_rumor.txt",
        ]

        data_list = [read_data(file) for file in file_list]

        format_data = {}
        for split_name, data in zip(file_list, data_list):
            name = str(split_name).split("/")[-1].split(".")[0]
            format_data[name] = []
            for i, d in enumerate(data):
                if i % 3 == 0:
                    format_data[name].append({})
                    format_data[name][-1]["meta_data"] = d
                if i % 3 == 1:
                    format_data[name][-1]["imgs"] = [img for img in d if img != "null"]
                else:
                    format_data[name][-1]["title"] = d
                format_data[name][-1]["label"] = 0 if "nonrumor" in name else 1

        for k, v in format_data.items():
            with open(
                root / "data" / "MM17-WeiboRumorSet" / f"{k}_format.json", "w", encoding="utf-8"
            ) as outfile:
                for d in v:
                    line = json.dumps(d, ensure_ascii=False)
                    outfile.write(line + "\n")

    # ===== preprocess data =====

    if lang == "cn":
        json_file_list = [
            root / "data/MM17-WeiboRumorSet/test_nonrumor_format.json",
            root / "data/MM17-WeiboRumorSet/test_rumor_format.json",
            root / "data/MM17-WeiboRumorSet/train_nonrumor_format.json",
            root / "data/MM17-WeiboRumorSet/train_rumor_format.json",
        ]

        dfs = [pd.read_json(file, lines=True) for file in json_file_list]

        for df in dfs:
            df["imgs"] = df.imgs.apply(lambda x: check_valid_image(save_path, x))

        for df in dfs:
            df["title"] = df.title.apply(lambda x: clean_text("".join(x)))

        test_data = pd.concat(dfs[:2])
        train_data = pd.concat(dfs[2:])

    else:
        train_data = pd.read_json(
            root / "data" / "MM17-WeiboRumorSet" / "train_data_tencent_translated_cleaned.json",
            lines=True,
        )
        test_data = pd.read_json(
            root / "data" / "MM17-WeiboRumorSet" / "test_data_tencent_translated_cleaned.json",
            lines=True,
        )
        train_data["title"] = train_data.translated_text.apply(clean_text)
        test_data["title"] = test_data.translated_text.apply(clean_text)

        train_data["imgs"] = train_data.imgs.apply(lambda x: check_valid_image(save_path, x))
        test_data["imgs"] = test_data.imgs.apply(lambda x: check_valid_image(save_path, x))

    print("train length:", len(train_data), "test length:", len(test_data))

    train_data_with_img = train_data[train_data.imgs.apply(len) > 0]
    train_data_with_img["imgs"] = train_data_with_img.imgs.apply(lambda x: x[0])
    train_data_with_img = train_data_with_img.drop_duplicates(subset=["title", "imgs"])

    test_data_with_img = test_data[test_data.imgs.apply(len) > 0]
    test_data_with_img["imgs"] = test_data_with_img.imgs.apply(lambda x: x[0])
    # test_data_with_img = test_data_with_img.drop_duplicates(subset=["title", "imgs"])

    print(
        "train length with image:",
        len(train_data_with_img),
        "test length with image:",
        len(test_data_with_img),
    )

    train_data_with_img.to_json(
        root / "data/MM17-WeiboRumorSet/train_data.json",
        lines=True,
        orient="records",
        force_ascii=False,
    )
    test_data_with_img.to_json(
        root / "data/MM17-WeiboRumorSet/test_data.json",
        lines=True,
        orient="records",
        force_ascii=False,
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--lang", type=str, default="cn")
    parser.add_argument("--is_format_posts", action="store_true")
    parser.add_argument("--is_refine_imgs", action="store_true")

    args = parser.parse_args()

    main(**vars(args))
