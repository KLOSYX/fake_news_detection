import pickle
import re
import time
from pathlib import Path
from typing import Dict, List

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
        if min(image.size) < 100:
            save_path = ""
        else:
            save_path = root_dir / (path.stem + path.suffix)
            with save_path.open("wb") as f:
                image.save(f)
    except Exception as e:
        tqdm.write(str(e))
        save_path = ""
    return str(save_path)


def check_valid_image(img_dict: Dict, img_id_list: list) -> List[str]:
    valid_img_list = []
    for img_id in img_id_list:
        img_id = img_id.strip()
        img_name = img_dict.get(img_id, "")
        if img_name:
            valid_img_list.append(img_name.split("/")[-1])
        else:
            print(f"{img_id} not exists")
    return valid_img_list


def detection_lang(text: str):
    try:
        lang = detect(text)
    except Exception as e:
        tqdm.write(f"text: {text}")
        tqdm.write(str(e))
        lang = "unk"
    return lang


if __name__ == "__main__":
    dev_data = pd.read_csv(
        root / "data/image-verification-corpus-master/mediaeval2016/devset/posts.txt",
        delimiter="\t",
    )

    # %%
    test_data = pd.read_csv(
        root / "data/image-verification-corpus-master/mediaeval2016/testset/posts_groundtruth.txt",
        delimiter="\t",
    )

    # %%
    dev_data["label"] = dev_data.label.apply(lambda x: 0 if x == "real" else 1)
    test_data["label"] = test_data.label.apply(lambda x: 0 if x == "real" else 1)

    # %%
    img_path_list = [
        root
        / "data/image-verification-corpus-master/mediaeval2015/devset/MediaEval2015_DevSet_Images",
        root
        / "data/image-verification-corpus-master/mediaeval2015/testset/MediaEval2015_TestSetImages",
        root
        / "data/image-verification-corpus-master/mediaeval2016/testset/Mediaeval2016_TestSet_Images/Mediaeval2016_TestSet_Images",
    ]

    # %%
    img_save_path = Path(root / "data/image-verification-corpus-master/images_filtered")
    img_list = []
    if not img_save_path.exists():
        img_save_path.mkdir(parents=True)

    for img_path in img_path_list:
        cnt = 0
        for f in tqdm(img_path.glob("*"), desc=f"Refining images in {str(img_path)}"):
            if f.suffix in [".txt"]:
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
    dev_data["imgs"] = dev_data["image_id(s)"].apply(
        lambda x: check_valid_image(img_dict, x.split(","))
    )

    translated_train_data = pickle.load(
        open(root / "data/image-verification-corpus-master/cleaned_train_text.pkl", "rb")
    )
    translated_test_data = pickle.load(
        open(root / "data/image-verification-corpus-master/cleaned_test_text.pkl", "rb")
    )

    dev_data["text"] = dev_data.post_text.apply(lambda x: clean_text(x))
    dev_data = dev_data[dev_data.text.apply(lambda x: len(x) > 0)]
    tqdm.pandas(desc="Detecting language")
    dev_data["lang"] = dev_data.text.progress_apply(lambda x: detection_lang(x))

    test_data["imgs"] = test_data["image_id"].apply(
        lambda x: check_valid_image(img_dict, x.split(","))
    )
    test_data["text"] = test_data.post_text.apply(lambda x: clean_text(x))
    test_data = test_data[test_data.text.apply(lambda x: len(x) > 0)]
    tqdm.pandas(desc="Detecting language")
    test_data["lang"] = test_data.text.progress_apply(lambda x: detection_lang(x))

    translator = google_translator(
        proxies={"http": "172.22.112.1:7890", "https": "172.22.112.1:7890"},
        timeout=5,
    )

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
    # dev_data_valid = dev_data_valid.drop_duplicates(subset=["text", "imgs"])

    test_data_valid = test_data[test_data.imgs.apply(len) > 0]
    test_data_valid["imgs"] = test_data_valid.imgs.apply(lambda x: x[0])
    # test_data_valid = test_data_valid.drop_duplicates(subset=["text", "imgs"])

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
