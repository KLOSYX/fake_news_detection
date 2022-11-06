import pickle
import re
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import pyrootutils
from langdetect import DetectorFactory, detect
from PIL import Image
from tqdm import tqdm

root = pyrootutils.setup_root(".", pythonpath=True)

from src.utils.data_util import clean_text
from src.utils.data_util.english_preprocessor import EnglishProcessor
from src.utils.google_trans_new.google_trans_new import google_translator

DetectorFactory.seed = 0
pd.options.mode.chained_assignment = None  # default='warn'


def translate_text(text: str, lang_tgt: str, retries: int = 3):
    fail_cnt = 0
    while fail_cnt < retries:
        try:
            result = translator.translate(text, lang_tgt=lang_tgt)
        except Exception as e:
            tqdm.write(f"text: {text}")
            tqdm.write(str(e))
            result = ""
            fail_cnt += 1
            time.sleep(1)
    return result


def get_translated_text(translated_dict, text_id, text):
    result = translated_dict.get(text_id, "")
    if not result:
        tqdm.write(text_id)
        result = translate_text(text, "en")
    return result


def refine_images(root_dir: Path, path: Path):
    if not root_dir.exists():
        root_dir.mkdir(parents=True)
    try:
        image = Image.open(str(path)).convert("RGB")
        # if min(image.size) < 100:
        #     save_path = ""
        # else:
        save_path = root_dir / (path.stem + path.suffix)
        with save_path.open("wb") as f:
            image.save(f)
    except Exception as e:
        tqdm.write(str(e))
        save_path = ""
    return str(save_path)


def check_valid_image(
    img_dict: Dict,
    img_ids: List[str],
    post_id: str,
    invalid_img_set: Set[str],
    mediaeval2015_data: pd.DataFrame,
) -> List[str]:
    """Filter out invalid images According to https://github.com/MKLab-ITI/image-verification-
    corpus/issues/4#issuecomment-1123350335, the images in the dataset could also present in the
    MediaEval 2015 dataset."""
    valid_img_list = []
    img_ids_old = mediaeval2015_data.loc[
        mediaeval2015_data["tweetId"] == post_id, "imageId(s)"
    ].values.tolist()
    if img_ids_old:
        img_ids_old = img_ids_old[0].split(",")
    img_ids_set = set(img_ids + img_ids_old)
    for img_id in img_ids_set:
        img_id = img_id.strip()
        img_name = img_dict.get(img_id, "")
        if img_name:
            valid_img_list.append(img_name.split("/")[-1])
        else:
            # print(f"{img_id} not exists")
            invalid_img_set.add(img_id)
    return valid_img_list


def detection_lang(text: str):
    try:
        lang = detect(text)
    except Exception as e:
        tqdm.write(f"text: {text}")
        tqdm.write(str(e))
        lang = "unk"
    return lang


def get_event_name(image_id: str) -> str:
    event_name = re.sub("fake", "", image_id)
    event_name = re.sub("real", "", event_name)
    event_name = re.sub("[0-9_]", "", event_name)
    event_name = event_name.split(".")[0]
    event_name = re.sub("[A-Z]+", "", event_name)
    return event_name


def main(args):
    dev_data = pd.read_csv(
        root / "data/image-verification-corpus-master/mediaeval2016/devset/posts.txt",
        delimiter="\t",
    )

    # %%
    test_data = pd.read_csv(
        root / "data/image-verification-corpus-master/mediaeval2016/testset/posts_groundtruth.txt",
        delimiter="\t",
    )

    mediaeval_2015_data = pd.concat(
        [
            pd.read_csv(
                root / "data/image-verification-corpus-master/mediaeval2015/devset/tweets.txt",
                delimiter="\t",
            ),
            pd.read_csv(
                root / "data/image-verification-corpus-master/mediaeval2015/testset/tweets.txt",
                delimiter="\t",
            ),
        ],
        axis=0,
    )

    # %%
    dev_data["label"] = dev_data.label.apply(lambda x: 0 if x == "real" else 1)
    test_data["label"] = test_data.label.apply(lambda x: 0 if x == "real" else 1)

    # %%
    img_path_list = [
        root
        / "data/image-verification-corpus-master/mediaeval2015/devset/Medieval2015_DevSet_Images",
        root / "data/image-verification-corpus-master/mediaeval2015/testset/TestSetImages",
        root
        / "data/image-verification-corpus-master/mediaeval2016/testset/Mediaeval2016_TestSet_Images",
    ]

    # %%
    img_save_path = Path(root / "data/image-verification-corpus-master/images_filtered")
    img_list = []
    if not img_save_path.exists():
        img_save_path.mkdir(parents=True)

    for img_path in img_path_list:
        cnt = 0
        for f in tqdm(img_path.glob("**/*"), desc=f"Refining images in {str(img_path)}"):
            if f.suffix in [".txt"] or f.is_dir():
                continue
            else:
                img = refine_images(img_save_path, f)
                if img:
                    img_list.append(img)
                    cnt += 1
        print(f"===== Valid images in {str(img_path)}: {cnt} in total ======")

    # %%
    img_dict = {x.split("/")[-1].split(".")[0]: x for x in img_list}

    # %%
    invalid_img_set = set()
    dev_data["imgs"] = dev_data.apply(
        lambda x: check_valid_image(
            img_dict,
            x["image_id(s)"].split(","),
            x["post_id"],
            invalid_img_set,
            mediaeval_2015_data,
        ),
        axis=1,
    )
    test_data["imgs"] = test_data.apply(
        lambda x: check_valid_image(
            img_dict, x["image_id"].split(","), x["post_id"], invalid_img_set, mediaeval_2015_data
        ),
        axis=1,
    )
    print(f"===== Invalid images: {len(invalid_img_set)} in total ======")
    print(invalid_img_set)

    translated_train_data = pickle.load(
        open(root / "data/image-verification-corpus-master/cleaned_train_text.pkl", "rb")
    )
    translated_test_data = pickle.load(
        open(root / "data/image-verification-corpus-master/cleaned_test_text.pkl", "rb")
    )

    dev_data["text"] = dev_data.post_text.apply(clean_text)
    dev_data = dev_data[dev_data.text.apply(lambda x: len(x) > 10)]
    tqdm.pandas(desc="Detecting language")
    dev_data["lang"] = dev_data.text.progress_apply(lambda x: detection_lang(x))

    test_data["text"] = test_data.post_text.apply(clean_text)
    test_data = test_data[test_data.text.apply(lambda x: len(x) > 10)]
    tqdm.pandas(desc="Detecting language")
    test_data["lang"] = test_data.text.progress_apply(lambda x: detection_lang(x))

    tqdm.pandas(desc="Translating text")
    dev_data["text"] = dev_data.progress_apply(
        lambda x: get_translated_text(translated_train_data, str(x["post_id"]), x["text"])
        if x["lang"] != "en"
        else x["text"],
        axis=1,
    )

    tqdm.pandas(desc="Translating text")
    test_data["text"] = test_data.progress_apply(
        lambda x: get_translated_text(translated_test_data, str(x["post_id"]), x["text"])
        if x["lang"] != "en"
        else x["text"],
        axis=1,
    )

    dev_data_valid = dev_data[dev_data.imgs.apply(len) > 0]
    dev_data_valid["imgs"] = dev_data_valid.imgs.apply(lambda x: x[0])
    dev_data_valid = dev_data_valid.drop_duplicates(subset=["text", "imgs"])

    test_data_valid = test_data[test_data.imgs.apply(len) > 0]
    test_data_valid["imgs"] = test_data_valid.imgs.apply(lambda x: x[0])
    # test_data_valid = test_data_valid.drop_duplicates(subset=["text", "imgs"])
    if args.use_strict_preprocessor:
        preprocessor = EnglishProcessor(min_len=0, stopwords_path=root / "data" / "stopwords.txt")
        test_data_valid["text"] = test_data_valid.text.apply(lambda x: preprocessor(x))
        dev_data_valid["text"] = dev_data_valid.text.apply(lambda x: preprocessor(x))

    # filter text that is shorter than 10
    if args.min_text_length > 0:
        min_text_length = args.min_text_length
        dev_data_valid = dev_data_valid[dev_data_valid.text.apply(lambda x: len(x.split()) > min_text_length)]
        test_data_valid = test_data_valid[test_data_valid.text.apply(lambda x: len(x.split()) > min_text_length)]

    # all_data = pd.concat([dev_data_valid, test_data_valid], axis=0)
    # all_data["event"] = all_data.imgs.apply(get_event_name)
    # all_data["event"] = np.argmax(pd.get_dummies(all_data.event).to_numpy(), axis=1)
    # dev_data_valid["event"] = all_data.event.iloc[: dev_data_valid.shape[0]]
    # test_data_valid["event"] = all_data.event.iloc[dev_data_valid.shape[0] :]

    # ===== Update: remove event in test dataset =====
    dev_data_valid["event"] = dev_data_valid.imgs.apply(get_event_name)
    dev_data_valid["event"] = np.argmax(pd.get_dummies(dev_data_valid.event).to_numpy(), axis=1)
    test_data_valid["event"] = -1
    # ===== end =====

    print(f"===== Training event number: {len(set(dev_data_valid.event))} ======")
    # print(f"===== Testing event number: {len(set(test_data_valid.event))} ======")

    print(
        "===== Saving data =====\n",
        "Valid training data size",
        dev_data_valid.shape[0],
        "\n",
        "Valid testing data size",
        test_data_valid.shape[0],
        "\n",
    )

    dev_data_valid[["text", "imgs", "event"]].to_json(
        root / "data/image-verification-corpus-master/train_posts.json",
        lines=True,
        orient="records",
        force_ascii=False,
    )
    test_data_valid[["text", "imgs", "event"]].to_json(
        root / "data/image-verification-corpus-master/test_posts.json",
        lines=True,
        orient="records",
        force_ascii=False,
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--use_strict_preprocessor", action="store_true")
    parser.add_argument("--min_text_length", type=int, default=0)

    args = parser.parse_args()

    main(args)
