import re


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
    text = re.sub(
        r"(?<=[^.])((?:(?:https?|ftp|file)://|(?<![a-zA-Z\-\.])www\.)"
        r"[\-A-Za-z0-9\+&@\(\)#/%\?=\~_|!:\,\.\;]+[\-A-Za-z0-9\+&@#/%=\~_\|])"
        r"(?=[<\u4E00-\u9FA5￥，。；！？、“”‘’>（）—《》…● \t\n])",
        " ",
        text,
    )  # remove URL
    text = text.strip()
    text = " ".join(text.split())

    return text
