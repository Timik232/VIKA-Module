import requests


def translate_to_en(text):
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
    return response.json()[0]["translations"][0]["text"]

