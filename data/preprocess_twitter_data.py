import json
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

from src.utils.google_trans_new.google_trans_new import google_translator

DetectorFactory.seed = 0
pd.options.mode.chained_assignment = None  # default='warn'


def clean_str_sst(string):
    """Tokenization/string cleaning for the SST dataset."""
    string = re.sub("[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "", string)
    return string.strip().lower()


# %%
def clean_text(text):
    try:
        text = text.decode("utf-8").lower()
    except Exception as ex:
        text = text.encode("utf-8").decode("utf-8").lower()
    text = re.sub("\u2019|\u2018", "'", text)
    text = re.sub("\u201c|\u201d", '"', text)
    text = re.sub("[\u2000-\u206F]", " ", text)
    text = re.sub("[\u20A0-\u20CF]", " ", text)
    text = re.sub("[\u2100-\u214F]", " ", text)
    text = re.sub(r"http:\ ", "http:", text)
    text = re.sub(r"http[s]?:[^\ ]+", " ", text)
    text = re.sub(r"&gt;", " ", text)
    text = re.sub(r"&lt;", " ", text)
    text = re.sub(r"&quot;", " ", text)
    text = re.sub(r"\"", " ", text)
    text = re.sub(r"#\ ", "#", text)
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"\\", " ", text)
    text = re.sub(r"[\(\)\[\]\{\}]", r" ", text)
    text = re.sub(
        "[" "\U0001F300-\U0001F64F" "\U0001F680-\U0001F6FF" "\u2600-\u26FF\u2700-\u27BF]+",
        r" ",
        text,
    )
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " had ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"#", " #", text)
    text = re.sub(r"@", " @", text)
    text = re.sub(r"[\!\?\.\,\+\-\$\%\^\>\<\=\:\;\*\(\)\{\}\[\]\/\~\&\'\|]", " ", text)
    text = text.strip()
    text = " ".join(text.split())

    return text


def translate_text(translator, text: str, lang_tgt: str, retries: int = 3):
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


def get_translated_text(translator, translated_text_map, text_id, text):
    result = translated_text_map.get(text_id, "")
    if not result:
        tqdm.write(f"Failed to get translated text: {text_id}")
        result = translate_text(translator, text, "en")
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
    invalid_img_set: Set[str],
) -> List[str]:
    """Filter out invalid images According to https://github.com/MKLab-ITI/image-verification-
    corpus/issues/4#issuecomment-1123350335, the images in the dataset could also present in the
    MediaEval 2015 dataset."""
    valid_img_list = []
    img_ids_set = set(img_ids)
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
    # event_name = re.sub("[A-Z]+", "", event_name)
    return event_name


if __name__ == "__main__":
    dev_data = pd.read_csv(
        root / "data/image-verification-corpus-master/mediaeval2015/devset/tweets.txt",
        delimiter="\t",
    )

    # %%
    test_data = pd.read_csv(
        root / "data/image-verification-corpus-master/mediaeval2015/testset/tweets.txt",
        delimiter="\t",
    )

    # mediaeval_2015_data = pd.concat(
    #     [
    #         pd.read_csv(
    #             root / "data/image-verification-corpus-master/mediaeval2015/devset/tweets.txt",
    #             delimiter="\t",
    #         ),
    #         pd.read_csv(
    #             root / "data/image-verification-corpus-master/mediaeval2015/testset/tweets.txt",
    #             delimiter="\t",
    #         ),
    #     ],
    #     axis=0,
    # )

    # %%
    dev_data["label"] = dev_data.label.apply(lambda x: 0 if x == "real" else 1)
    test_data["label"] = test_data.label.apply(lambda x: 0 if x == "real" else 1)

    # %%
    img_path_list = [
        root
        / "data/image-verification-corpus-master/mediaeval2015/devset/Medieval2015_DevSet_Images",
        root / "data/image-verification-corpus-master/mediaeval2015/testset/TestSetImages",
        # root
        # / "data/image-verification-corpus-master/mediaeval2016/testset/Mediaeval2016_TestSet_Images",
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
            x["imageId(s)"].split(","),
            invalid_img_set,
        ),
        axis=1,
    )
    test_data["imgs"] = test_data.apply(
        lambda x: check_valid_image(img_dict, x["imageId(s)"].split(","), invalid_img_set),
        axis=1,
    )
    print(f"===== Invalid images: {len(invalid_img_set)} in total ======")
    print(invalid_img_set)

    # translated_train_data = pickle.load(
    #     open(root / "data/image-verification-corpus-master/cleaned_train_text.pkl", "rb")
    # )
    # translated_test_data = pickle.load(
    #     open(root / "data/image-verification-corpus-master/cleaned_test_text.pkl", "rb")
    # )
    translated_text_map = json.load(
        (root / "data/image-verification-corpus-master/translated_text_map.json").open("r")
    )

    dev_data["text"] = dev_data.tweetText.apply(lambda x: clean_text(x))
    dev_data = dev_data[dev_data.text.apply(lambda x: len(x) > 0)]
    tqdm.pandas(desc="Detecting language")
    dev_data["lang"] = dev_data.text.progress_apply(lambda x: detection_lang(x))

    test_data["text"] = test_data.tweetText.apply(lambda x: clean_text(x))
    test_data = test_data[test_data.text.apply(lambda x: len(x) > 0)]
    tqdm.pandas(desc="Detecting language")
    test_data["lang"] = test_data.text.progress_apply(lambda x: detection_lang(x))

    translator = google_translator(
        proxies={"https": "172.22.112.1:7890"},
        timeout=5,
    )

    tqdm.pandas(desc="Translating text")
    dev_data["text"] = dev_data.progress_apply(
        lambda x: get_translated_text(
            translator, translated_text_map, str(x["tweetId"]), x["text"]
        )
        if x["lang"] != "en"
        else x["text"],
        axis=1,
    )

    tqdm.pandas(desc="Translating text")
    test_data["text"] = test_data.progress_apply(
        lambda x: get_translated_text(
            translator, translated_text_map, str(x["tweetId"]), x["text"]
        )
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

    all_data = pd.concat([dev_data_valid, test_data_valid], axis=0)
    all_data["event"] = all_data.imgs.apply(get_event_name)
    all_data["event"] = np.argmax(pd.get_dummies(all_data.event).to_numpy(), axis=1)
    dev_data_valid["event"] = all_data.event.iloc[: dev_data_valid.shape[0]]
    test_data_valid["event"] = all_data.event.iloc[dev_data_valid.shape[0] :]
    print("===== Training events: ", dev_data_valid.event.unique().shape[0], " =====")
    print("===== Testing events: ", test_data_valid.event.unique().shape[0], " =====")
    print("===== Total events: ", all_data.event.unique().shape[0], " ======")

    print(
        "===== Saving data =====\n",
        "Valid training data size",
        dev_data_valid.shape[0],
        "\n",
        "Valid testing data size",
        test_data_valid.shape[0],
        "\n",
    )

    dev_data_valid.to_json(
        root / "data/image-verification-corpus-master/train_posts.json",
        lines=True,
        orient="records",
        force_ascii=False,
    )
    test_data_valid.to_json(
        root / "data/image-verification-corpus-master/test_posts.json",
        lines=True,
        orient="records",
        force_ascii=False,
    )
