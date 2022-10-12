import requests
import random
from hashlib import md5


class BaiDuFanyi:
    def __init__(self, appKey, appSecret, fromLang="auto", toLang="en"):
        self.url = "https://fanyi-api.baidu.com/api/trans/vip/translate"
        self.appid = appKey
        self.secretKey = appSecret
        self.fromLang = fromLang
        self.toLang = toLang
        self.salt = random.randint(32768, 65536)
        self.header = {"Content-Type": "application/x-www-form-urlencoded"}

    def translate(self, text):
        sign = self.appid + text + str(self.salt) + self.secretKey
        md = md5()
        md.update(sign.encode(encoding="utf-8"))
        sign = md.hexdigest()
        data = {
            "appid": self.appid,
            "q": text,
            "from": self.fromLang,
            "to": self.toLang,
            "salt": self.salt,
            "sign": sign,
        }
        response = requests.post(self.url, params=data, headers=self.header)  # 发送post请求
        text = response.json()  # 返回的为json格式用json接收数据
        # print(text)
        try:
            results = text["trans_result"][0]["dst"]
            return results
        except Exception as ex:
            print(ex)
            return ""


if __name__ == "__main__":
    appKey = "20221012001387816"  # 你在第一步申请的APP ID
    appSecret = "ThXoAK3TTPMmnaOKX0yF"  # 公钥
    BaiduTranslate_test = BaiDuFanyi(appKey, appSecret)
    Results = BaiduTranslate_test.translate("Hello, World!")  # 要翻译的词组
    print(Results)
