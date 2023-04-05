import os
import pickle
from torch.utils.data import DataLoader
from vk_api.utils import get_random_id
import vk_api
from private_api import token_api  # токен который не должен быть у всех, поэтому вынес в отдельный файл.
import nltk
import requests
import re
import random
import json
import tokenization
from transformers import logging
import tensorflow as tf
import numpy as np
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import pipeline
from huggingface_hub import notebook_login
from UserClass import UserInfo

# notebook_login()

logging.set_verbosity_error()

vk_session = vk_api.VkApi(token=token_api)
vk = vk_session.get_api()


def send_message(id, msg, stiker=None, attach=None):
    try:
        vk.messages.send(
            user_id=id,
            random_id=get_random_id(),
            message=msg,
            sticker_id=stiker,
            attachment=attach
        )
    except BaseException as ex:
        print(ex)
        return


def send_document(user_id, doc_req, message=None):
    upload = vk.VkUpload(vk_session)
    document = upload.document_message(doc_req)[0]
    print(document)
    owner_id = document['owner_id']
    doc_id = document['id']
    attachment = f'doc{owner_id}_{doc_id}'
    post = {'user_id': user_id, 'random_id': 0, "attachment": attachment}
    if message is not None:
        post['message'] = message
    try:
        vk_session.method('messages.send', post)
    except BaseException:
        send_message(id, "Не удалось отправить документ")
        return


def send_photo(user_id, img_req, message=None):
    upload = vk_api.VkUpload(vk_session)
    photo = upload.photo_messages(img_req)[0]
    owner_id = photo['owner_id']
    photo_id = photo['id']
    attachment = f'photo{owner_id}_{photo_id}'
    post = {'user_id': user_id, 'random_id': 0, "attachment": attachment}
    if message is not None:
        post['message'] = message
    try:
        vk_session.method('messages.send', post)
    except BaseException:
        send_message(id, "Не удалось отправить картинку")
        return


def check_room(text):
    text = text.lower().replace("a", "а").replace("b", "в")
    words = text.split()
    if len(words) > 10:
        return ""
    re_78 = r"^(([а-ди])|(ивц))-?[1-4][0-9]{2}$"
    re_78_lection = r"^а-?(([1-9])|(1[1-8]))$"
    re_universal = r"^[а-яa-z]{1,3}-*[0-9]+$"
    re_86 = r'^[лтсрон]-?[1-9][0-9]{0,2}$'
    re_78_without = r"[1-2][08]-?[0-5]"
    re_20_kpk = r"^[1-4][0-9]{0,2}$"
    for word in words:
        if re.search(re_78, word) or re.search(re_78_lection, word) or re.search(re_78_without, word):
            return "78"
        elif re.search(re_86, word):
            return "86"
        elif re.search(re_20_kpk, word):
            return "kpk_20"
        elif re.search(re_universal, word):
            return "not_found"
    return ""


def clean_up(text):
    text = text.lower()
    # описываем текстовый шаблон для удаления: "все, что НЕ является буквой \w или пробелом \s"
    re_not_word = r'[^\w\s]'
    text = re.sub(re_not_word, '', text)
    return text


def text_match(user_text, example):
    user_text = clean_up(user_text)
    example = clean_up(example)
    if user_text.find(example) != -1:
        return True
    if example.find(user_text) != -1:
        return True
    example_len = len(example)
    difference = nltk.edit_distance(user_text, example)
    return (difference / example_len) < 0.3


users = {}


def get_intent(text, model_mlp, vectorizer, dictionary):
    corrected_text = ""
    for word in text.split():
        new_word = str(dictionary.correction(word))
        if new_word == "None":
            corrected_text += word + ' '
        else:
            corrected_text += new_word + ' '
    # corrected_text = dictionary.correction(text)
    text_vec = vectorizer.transform([corrected_text.strip()])
    return model_mlp.predict(text_vec)[0]


def get_intent_bert(text, model_mlp, vectorizer, dictionary):
    corrected_text = ""
    for word in text.split():
        new_word = str(dictionary.correction(word))
        if new_word == "None":
            corrected_text += word + ' '
        else:
            corrected_text += new_word + ' '
    # corrected_text = dictionary.correction(text)
    text_vec = vectorizer.encode([corrected_text])
    import pandas as pd
    proba = model_mlp.predict_proba(text_vec)[0]
    print(max(proba), corrected_text)
    # print(pd.DataFrame(columns=model_mlp.classes_, data=proba), sep="\n")
    return model_mlp.predict(text_vec)[0]


def get_response(intent, data):
    return random.choice(data[intent]['responses'])


def answering(text, model_mlp, data, vectorizer, dictionary):
    text = clean_up(text)
    if text.strip() == "" or text == " " or len(text) < 2:
        intent = "flood"
    else:
        # intent = get_intent(text, model_mlp, vectorizer, dictionary)
        intent = get_intent_bert(text, model_mlp, vectorizer, dictionary)
        # intent = cosine_sim(text, vectorizer)
    answer = get_response(intent, data)
    full_answer = [answer, intent]
    return full_answer


def add_answer(users):
    with open('jsons\\intents_dataset.json', 'r', encoding='UTF-8') as f:
        data = json.load(f)
    while True:
        print(
            "Выберите пункт меню:\n1.Вывести количество тем\n2.Вывести все темы\n3.Добавить тему\n4.Удалить тему\n5.Вывести всю информацию по теме\n6.Добавить ответ к теме\n7.Добавить вопрос к теме\n8.Вывести количество пользователей\n9.Вывести рейтинг")
        choice = input()
        if choice == "1":
            print("Количество тем: " + str(len(data)))
        elif choice == "2":
            for i in data:
                print(i)
        elif choice == "3":
            print("Введите название темы")
            intent = input()
            if intent in data:
                print("Такая тема уже существует")
            else:
                print("Вводите вопросы, чтобы закончить, введите 0")
                flag = False
                while True:
                    question = input()
                    if question == "0":
                        break
                    if not flag:
                        data[intent] = {}
                        data[intent]['examples'] = []
                        data[intent]['responses'] = []
                        flag = True
                    data[intent]['examples'].append(question)
                print("Вводите ответы, чтобы закончить, введите 0")
                while True:
                    answer = input()
                    if answer == "0":
                        break
                    data[intent]['responses'].append(answer)
                with open('jsons\\intents_dataset.json', 'w', encoding='UTF-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                print("Ответ был записан в файл. Ввести еще ответ? (y/n)")
                end = input()
                end = end.lower()
                if end == "n" or end == "no" or end == "нет":
                    break
                else:
                    print("Продолжайте вводить")
        elif choice == "4":
            print("Введите название темы")
            intent = input()
            if intent in data:
                del data[intent]
                with open('intents_dataset.json', 'w', encoding='UTF-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                print("Тема была удалена")
            else:
                print("Тема не найдена")
        elif choice == "5":
            print("Введите название темы")
            intent = input()
            if intent in data:
                print(data[intent])
            else:
                print("Тема не найдена")
        elif choice == "6":
            print("Введите название темы")
            intent = input()
            if intent in data:
                print("Вводите ответы, чтобы закончить, введите 0")
                while True:
                    answer = input()
                    if answer == "0":
                        break
                    data[intent]['responses'].append(answer)
                with open('jsons\\intents_dataset.json', 'w', encoding='UTF-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                print("Ответ был записан в файл.")
            else:
                print("Тема не найдена")
        elif choice == "7":
            print("Введите название темы")
            intent = input()
            if intent in data:
                print("Вводите вопросы, чтобы закончить, введите 0")
                while True:
                    question = input()
                    if question == "0":
                        break
                    data[intent]['examples'].append(question)
                with open('jsons\\intents_dataset.json', 'w', encoding='UTF-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                print("Вопросы были записаны в файл.")
            else:
                print("Тема не найдена")
        elif choice == "8":
            print("Количество пользователей бота: " + str(len(users)))
        elif choice == "9":
            rate = 0
            for i in users:
                rate += users[i].like
            print("Рейтинг бота (количество лайков минус количество дизлайков): " + str(rate))
        else:
            print("Неверный пункт меню")


