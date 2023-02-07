from vk_api.keyboard import VkKeyboard, VkKeyboardColor
from vk_api.utils import get_random_id
import vk_api
from private_api import token_api  # токен который не должен быть у всех,поэтому вынес в отдельную функцию
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
import random
import json


vk_session = vk_api.VkApi(token=token_api)
vk = vk_session.get_api()


# класс с информацией о пользователе, чтобы можно было запоминать информацию о пользователе, в том числе о рассылках.
# не знаю, как это реализовано в основном боте, поэтому пусть будет так
class UserInfo:
    def __init__(self):
        # предложение запоминать группу пользователя, чтобы не вводить её каждый раз
        self.name = ""  # заглушка чтобы отправлять сообщения создателю
        self.group = ""
        self.state = ""  # состояние пользователя, чтобы понимать, что ему нужно сделать
        self.sending = []
        # условно изначальные рассылки присвоить False, чтобы не рассылал, сделано скорее как заглушка
        for i in range(11):
            self.sending.append(False)


def send_message(id, msg, stiker=None, attach=None):
        try:
            vk.messages.send(
                user_id=id,
                random_id=get_random_id(),
                message=msg,
                sticker_id=stiker,
                attachment=attach
            )
        except BaseException:
            print("ошибка, возможно человек добавил в чс")
            return


# на данный момент только одна клавиатура, но есть возможность создавать другие
def create_keyboard(id, text, response="start"):
    try:
        keyboard = VkKeyboard(one_time=True)
        if response == "start":
            keyboard = VkKeyboard(one_time=False)
            keyboard.add_button('Расписание', color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button('Карта Университета', color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button('Рассылка', color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button('Расписание пересдач', color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button('Что ты умеешь?', color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button('Пожелания по улучшению', color=VkKeyboardColor.PRIMARY)
        elif response == "yaro":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Ссылка на ВК', "https://vk.com/bramind002")
        elif response == "grifon":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Стикеры с грифоном в ТГ', "https://t.me/addstickers/rtumirea")
        elif response == "psychology":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Психологическая служба', "https://student.mirea.ru/psychological_service/staff/")
        elif response == "map":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Навигатор', "https://ischemes.ru/group/rtu-mirea/vern78")
        elif response == "rules":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Устав', "https://www.mirea.ru/upload/medialibrary/d0e/Ustav-Novyy.pdf")
            keyboard.add_openlink_button('Правила внутреннего распорядка', "https://www.mirea.ru/docs/125641/")
            keyboard.add_openlink_button('Этический кодекс', "https://student.mirea.ru/regulatory_documents/file/3f9468db49ffd14fe96c0d28d8c056bf.pdf")
        elif response == "museums":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Подробнее о музеях', "https://www.mirea.ru/about/history-of-the-university/the-museum-mirea/")
        elif response == "obhodnoy":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Про физкультуру', "https://student.mirea.ru/help/section/physical_education/")
        elif response == "work":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Центр карьеры', "https://career.mirea.ru/")
        elif response == "website":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Сайт МИРЭА', "https://www.mirea.ru/")
        elif response == "military":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Памятка военнообязанному', "https://student.mirea.ru/help/section/conscript/")
        elif response == "rectorate":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Ректорат', "https://www.mirea.ru/about/administration/rektorat/")
        elif response == "library":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Сайт библиотеки', "https://library.mirea.ru/")
        elif response == "office":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('СтудОфис', "https://student.mirea.ru/services/")
        elif response == "scholarship":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Размер стипендии', "https://student.mirea.ru/scholaship_support/scholarships/state_academic_support/")
        vk.messages.send(
            user_id=id,
            random_id=get_random_id(),
            message=text, keyboard=keyboard.get_keyboard())
    except BaseException as Exception:
        print(Exception)
        return


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
    return (difference / example_len) < 0.4


users = {}


def get_intent(text, model_mlp, vectorizer):
    text_vec = vectorizer.transform([text])
    return model_mlp.predict(text_vec)[0]


def get_response(intent, data):
    return random.choice(data[intent]['responses'])


def answering(text, model_mlp, data, vectorizer):
    intent = get_intent(text, model_mlp, vectorizer)
    answer = get_response(intent, data)
    full_answer = []
    full_answer.append(answer)
    full_answer.append(intent)
    return full_answer


def add_answer():
    with open('intents_dataset.json', 'r', encoding='UTF-8') as f:
        data = json.load(f)
    while True:
        print("Введите название интента")
        intent = input()
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
        with open('intents_dataset.json', 'w', encoding='UTF-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print("Ответ был записан в файл. Продолжайте вводить")




