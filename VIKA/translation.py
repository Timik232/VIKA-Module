import requests
import os
import json
import subprocess
from neuro_defs import cwd, slh
from private_api import oauth_token, FOLD_ID


def generate_iam_token():
    command = 'yc iam create-token'

    try:
        output = subprocess.check_output(command, shell=True)
        token = output.decode('utf-8').strip()
        return token
    except subprocess.CalledProcessError as e:
        print(f"Failed to generate IAM token: {e}")
        return None


def translate_to_en_no_yandex(intent, text, data):
    if not os.path.isfile(f'jsons{slh()}translated.json'):
        translated = {}
        with open(f'jsons{slh()}translated.json', 'w', encoding='UTF-8') as f:
            json.dump(translated, f, ensure_ascii=False, indent=4)
    else:
        with open(f'jsons{slh()}intents_dataset.json', 'r', encoding='UTF-8') as f:
            translated = json.load(f)
    if intent in translated:
        return data[intent]["responses"][0]
    else:
        try:
            url = "https://microsoft-translator-text.p.rapidapi.com/translate"

            querystring = {"to[0]":"en","api-version":"3.0","profanityAction":"NoAction","textType":"plain"}

            payload = [{ "Text": text }]
            headers = {
                "content-type": "application/json",
                "X-RapidAPI-Key": "fe03215d45msh627865176ed4684p19531ajsn8091f3a6ca92",
                "X-RapidAPI-Host": "microsoft-translator-text.p.rapidapi.com"
            }

            response = requests.post(url, json=payload, headers=headers, params=querystring)
            print(response.json())
        except Exception:
            return "Can't translate to English, change language to Russian."
        translated[intent] = response.json()[0]["translations"][0]["text"]
        with open(f'jsons{slh()}translated.json', 'w', encoding='UTF-8') as f:
            json.dump(translated, f, ensure_ascii=False, indent=4)
        return response.json()[0]["translations"][0]["text"]


def translate_to_en(intent, text, data):
    if not os.path.isfile(f'jsons{slh()}translated.json'):
        translated = {}
        with open(f'jsons{slh()}translated.json', 'w', encoding='UTF-8') as f:
            json.dump(translated, f, ensure_ascii=False, indent=4)
    else:
        with open(f'jsons{slh()}translated.json', 'r', encoding='UTF-8') as f:
            translated = json.load(f)
    if intent in translated:
        return translated[intent]
    else:
        print(translated)
        try:
            url = "https://iam.api.cloud.yandex.net/iam/v1/tokens"
            headers = {
                "Content-Type": "application/json",
            }

            data = {
                "yandexPassportOauthToken": oauth_token
            }

            response = requests.post(url, json=data, headers=headers)
            IAM_TOKEN = response.json()["iamToken"]
            # IAM_TOKEN = "t1.9euelZrHiZ7KzZebmMbMipKTlseYie3rnpWazJeby5CeipSUicqZno_Plpzl9PdDZVhb-e99NmaM3fT3AxRWW_nvfTZmjM3n9euelZrJnJ6alJqTnonHzs2Ol4mVlO_8xeuelZrJnJ6alJqTnonHzs2Ol4mVlA.i0WCeHEUOpBDH7xRsbgMi0GYKxlmGnzxnFWxQigE3zGtgfAENqMiNA3uWECrAVkPDwjt9oS0ZhyFLLOtV9bCCw"
            folder_id = FOLD_ID
            target_language = "en"
            texts = [text]
            body = {
                "targetLanguageCode": target_language,
                "texts": texts,
                "folderId": folder_id,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer {0}".format(IAM_TOKEN)
            }
            response = requests.post('https://translate.api.cloud.yandex.net/translate/v2/translate',
                                     json=body,
                                     headers=headers
                                     )
            print(response.json())
        except Exception:
            return "Can't translate to English, change language to Russian."
        translated[intent] = response.json()["translations"][0]["text"]
        with open(f'jsons{slh()}translated.json', 'w', encoding='UTF-8') as f:
            json.dump(translated, f, ensure_ascii=False, indent=4)
        return response.json()["translations"][0]["text"]
