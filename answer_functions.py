from keyboard import create_keyboard
from neuro_defs import text_match, send_message, vk
from threading import Thread
from learning_functions import make_bertnetwork, learn_spell
import json


def is_canceled(id, msg):
    if text_match(msg, 'отмена') or text_match(msg, 'отменить') or text_match(msg, 'назад'):
        send_message(id, "Отменено")
        return True
    else:
        return False


def is_intent(id, intent, data, is_new = False):
    if intent in data:
        return True
    else:
        if not is_new:
            send_message(id, "Тема не найдена")
        return False


def admin_answer(id, users, message, event, data):
    if text_match(message, "выход") or text_match(message, ".Выход"):
        users[id].state = "waiting"
        create_keyboard(id, "Выход выполнен, можете снова пользоваться ботом")
    elif event.text == "1.Вывести все темы":
        smg = ""
        count = 0
        for i in data:
            smg += i + ", "
            count += 1
            if count == 200:
                send_message(id, smg)
                smg = ""
                count = 0
        send_message(id, smg)
        create_keyboard(id, "Выберите пункт меню", "admin")
    elif event.text == "2.Добавить тему":
        send_message(id, "Введите название темы")
        users[id].state = "add_intent"
    elif event.text == "3.Удалить тему":
        send_message(id, "Введите название темы")
        users[id].state = "delete"
    elif event.text == "4.Вывести всю информацию по теме":
        send_message(id, "Введите название темы")
        users[id].state = "info"
    elif event.text == "5.Добавить ответ к теме":
        send_message(id, "Введите название темы")
        users[id].state = "check_intent add_answer"
    elif event.text == "6.Добавить вопрос к теме":
        send_message(id, "Введите название темы")
        users[id].state = "check_intent add_question"
    elif event.text == "7.Найти тему по вопросу" or text_match(message, "тема по вопросу"):
        users[id].state = "get_topic"
        send_message(id, "Отправьте вопрос, по которому будет выдана тема")
    elif event.text == "8.Статистика и рейтинг":
        users[id].state = "statistic"
        create_keyboard(id, "Выберите пункт меню", "statistic")
    elif text_match(message, "Переобучить модель"):
        create_keyboard(id, "Вы уверены? Это может занять значительное время (да/нет)", "yesno")
        users[id].state = "retrain"
    else:
        send_message(id, "Неверный пункт меню")
        create_keyboard(id, "Выберите пункт меню", "admin")


def delete_answer(id, users, data, message, event):
    users[id].state = "admin"
    if is_canceled(id, message):
        create_keyboard(id, "Выберите пункт меню", "admin")
    elif is_intent(id, event.text, data):
        del data[event.text]
        with open('jsons\\intents_dataset.json', 'w', encoding='UTF-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        send_message(id, "Тема была удалена")
        create_keyboard(id, "Выберите пункт меню", "admin")
    else:
        create_keyboard(id, "Выберите пункт меню", "admin")


def info_answer(id, users, message, event, data):
    users[id].state = "admin"
    if is_canceled(id, message):
        create_keyboard(id, "Выберите пункт меню", "admin")
    elif is_intent(id, event.text, data):
        send_message(id, str(data[event.text]))
        create_keyboard(id, "Выберите пункт меню", "admin")


def feedback_answer(id, users, message, event):
    if text_match(message, "отмена"):
        users[id].state = "waiting"
        send_message(id, "Пожелание отменено")
    else:
        users[id].state = "waiting"
        user_get = vk.users.get(user_ids=id)
        first_name = user_get[0]['first_name']
        last_name = user_get[0]['last_name']
        send_message(286288768,
                     "Пользователь " + first_name + " " + last_name + " хочет сказать: " + event.text + " (id: " + str(
                         id) + ")")
        send_message(id, "Спасибо за ваше пожелание!")\


def check_intent_answer(id, users, message, event, data):
    if is_canceled(id, message):
        users[id].state = "admin"
        create_keyboard(id, "Выберите пункт меню", "admin")
    elif is_intent(id, event.text, data):
        if users[id].state.split()[1] == "add_answer":
            send_message(id, "Вводите ответы, чтобы закончить, введите 0")
            users[id].state = "add_answer2" + " " + event.text
        elif users[id].state.split()[1] == "add_question":
            send_message(id, "Вводите вопросы, чтобы закончить, введите 0")
            users[id].state = "add_question2" + " " + event.text
    else:
        users[id].state = "admin"
        create_keyboard(id, "Выберите пункт меню", "admin")


def add_intent_answer(id, users, message, event, data):
    if is_canceled(id, message):
        users[id].state = "admin"
        create_keyboard(id, "Выберите пункт меню", "admin")
    else:
        intent = event.text.replace(" ", "-")
        if is_intent(id, intent, data, True):
            send_message(id, "Такая тема уже есть")
            users[id].state = "admin"
        else:
            data[intent] = {}
            data[intent]['examples'] = []
            data[intent]['responses'] = []
            send_message(id, "Вводите вопросы, чтобы закончить, введите 0")
        users[id].state = "add_question2" + " " + intent + " end"


def is_end_answer(id, users, message):
    if text_match(message, "да") or text_match(message, "yes"):
        send_message(id, "Вводите тему")
        users[id].state = "add_intent"
    elif text_match(message, "нет") or text_match(message, "no"):
        users[id].state = "admin"
        create_keyboard(id, "Выберите пункт меню", "admin")


def retrain_answer(id, users, message,data):
    if text_match(message, "да") or text_match(message, "yes"):
        users[id].state = "admin"
        create_keyboard(id, "Выберите пункт меню", "admin")
        Thread(target=learn_spell, args=(data,)).start()
        # neuro = make_neuronetwork()
        neuro = make_bertnetwork()
        model_mlp = neuro[0]
        vectorizer = neuro[1]
        send_message(id, "Нейросеть переобучена")
    else:
        send_message(id, "Отменено")
        users[id].state = "admin"
        create_keyboard(id, "Выберите пункт меню", "admin")


def add_answer_answer(id, users, event, data):
    intent = users[id].state.split()[1]
    answer = event.text
    answer = answer.replace("&quot;", '"')
    if answer == "0":
        with open('jsons\\intents_dataset.json', 'w', encoding='UTF-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        send_message(id, "Ответ был записан в файл.")
        if len(users[id].state.split()) == 2:
            users[id].state = "admin"
            create_keyboard(id, "Выберите пункт меню", "admin")
        else:
            create_keyboard(id, "Тема закончена. Ввести еще одну тему? (да/нет)", "yesno")
            users[id].state = "is_end"
    else:
        data[intent]['responses'].append(answer)
        send_message(id, "Дальше")


def add_question_answer(id, users, event, data):
    intent = users[id].state.split()[1]
    question = event.text
    question = question.replace("&quot;", '"')
    if question == "0":
        with open('jsons\\intents_dataset.json', 'w', encoding='UTF-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        send_message(id, "Вопросы были записаны в файл.")
        if len(users[id].state.split()) == 2:
            users[id].state = "admin"
            create_keyboard(id, "Выберите пункт меню", "admin")
        else:
            users[id].state = "add_answer2" + " " + intent + " end"
            send_message(id, "Вводите ответы, чтобы закончить, введите 0")
    else:
        data[intent]['examples'].append(question)
        send_message(id, "Дальше")


def statistic_answer(id, users, message, event, data):
    if event.text == "1.Вывести количество тем":
        send_message(id, "Количество тем: " + str(len(data)))
        create_keyboard(id, "Выберите пункт меню", "statistic")
    elif event.text == "2.Рейтинг бота":
        rate = 0
        for i in users:
            rate += users[i].like
        send_message(id, "Рейтинг бота (количество лайков минус количество дизлайков): " + str(rate))
        create_keyboard(id, "Выберите пункт меню", "statistic")
    elif event.text == "3.Количество вопросов":
        count = 0
        for i in data.keys():
            for j in data[i]["examples"]:
                count += 1
        count -= 300  # вычитаем вопросы приветствия/прощания и т.п.
        send_message(id, "Количество вопросов, внесённых в бота: " + str(count))
        create_keyboard(id, "Выберите пункт меню", "statistic")
    elif event.text == "4.Вывести количество пользователей":
        send_message(id, "Количество пользователей бота: " + str(len(users)))
        create_keyboard(id, "Выберите пункт меню", "statistic")
    elif text_match(message, "5.Вернуться"):
        users[id].state = "admin"
        create_keyboard(id, "Выберите пункт меню", "admin")
    elif text_match(message, "выход"):
        users[id].state = "waiting"
        create_keyboard(id, "Выход выполнен, можете снова пользоваться ботом")
    else:
        send_message(id, "Неверный пункт меню")
        create_keyboard(id, "Выберите пункт меню", "statistic")