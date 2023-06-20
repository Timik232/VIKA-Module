import os

from keyboard import create_keyboard
from neuro_defs import text_match, send_message, vk, get_intent_bert, get_starting_date
from threading import Thread
from learning_functions import make_bertnetwork, learn_spell, fine_tuning
import datetime
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


def create_topic_answer(event, data, intent = None):
    if intent is None:
        answer = "Название темы: " + event.text + "\n\n"
        answer += "Вопросы:\n" + "\n".join(data[event.text]["examples"]) + "\n\n"
        answer += "Ответ(ы):\n" + "\n".join(data[event.text]["responses"])
    else:
        answer = "Название темы: " + intent + "\n\n"
        answer += "Вопросы:\n" + "\n".join(data[intent]["examples"]) + "\n\n"
        answer += "Ответ(ы):\n" + "\n".join(data[intent]["responses"])
    return answer


def create_answer_question_lists(event, data, is_question=True):
    if is_question:
        return data[event.text]["examples"]
    else:
        return data[event.text]["responses"]


def get_full_topic_answer(id, users, message, event, data, model_mlp, vectorizer, dictionary, objects):
    intent = get_intent_bert(message, model_mlp, vectorizer, dictionary, objects)
    answer = create_topic_answer(event, data, intent)
    send_message(id, answer)
    users[id].state = "admin"
    create_keyboard(id, "Выберите пункт меню", "admin")


def edit_answer(id, users, message, event):
    if text_match(message, "выход") or text_match(message, ".Выход"):
        users[id].state = "waiting"
        create_keyboard(id, "Выход выполнен, можете снова пользоваться ботом")
    elif event.text == "1.Добавить тему":
        send_message(id, "Введите название темы")
        users[id].state = "add_intent"
    elif event.text == "2.Удалить тему":
        send_message(id, "Введите название темы")
        users[id].state = "delete"
    elif event.text == "3.Добавить ответ к теме":
        send_message(id, "Введите название темы")
        users[id].state = "check_intent add_answer"
    elif event.text == "4.Добавить вопрос к теме":
        send_message(id, "Введите название темы")
        users[id].state = "check_intent add_question"
    elif event.text == "5.Удалить вопрос у темы":
        send_message(id, "Введите название темы")
        users[id].state = "delete_question"
    elif event.text == "6.Удалить ответ у темы":
        send_message(id, "Введите название темы")
        users[id].state = "delete_answer"
    elif event.text == "7.Вернуться":
        users[id].state = "admin"
        create_keyboard(id, "Выберите пункт меню", "admin")
    elif text_match(message, "выход"):
        users[id].state = "waiting"
        create_keyboard(id, "Выход выполнен, можете снова пользоваться ботом")
    else:
        send_message(id, "Неверный пункт меню")
        create_keyboard(id, "Выберите пункт меню", "statistic")


def get_list_dates(starting_dates):
    current_dates = ""
    dates = []
    for i in range(4):
        dates.append(starting_dates[i].strftime("%d %B"))
    current_dates += f"Начало осеннего семестра: {dates[0]}\n"
    current_dates += f"Конец осеннего семестра: {dates[1]}\n"
    current_dates += f"Начало весеннего семестра: {dates[2]}\n"
    current_dates += f"Конец весеннего семестра: {dates[3]}\n"
    return current_dates


def admin_answer(id, users, message, event, data, starting_dates):
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
    elif event.text == "2.Вывести всю информацию по теме":
        send_message(id, "Введите название темы")
        users[id].state = "info"
    elif event.text == "3.Найти тему по вопросу" or text_match(message, "тема по вопросу"):
        users[id].state = "get_topic"
        send_message(id, "Отправьте вопрос, по которому будет выдана тема")
    elif event.text == "4.Вывести всю тему по вопросу":
        users[id].state = "get_full_topic"
        send_message(id, "Отправьте вопрос, по которому будет выдана тема")
    elif event.text == "5.Статистика и рейтинг":
        users[id].state = "statistic"
        create_keyboard(id, "Выберите пункт меню", "statistic")
    elif event.text == "6.Управление темами":
        users[id].state = "edit"
        create_keyboard(id, "Выберите пункт меню", "edit")
    elif event.text == "7.Переобучить модель":
        create_keyboard(id, "Вы уверены? Это может занять значительное время (да/нет)", "yesno")
        users[id].state = "retrain"
    elif event.text == "8.Изменить даты":
        current_dates = get_list_dates(starting_dates)
        create_keyboard(id, current_dates, "dates")
        users[id].state = "dates"
    else:
        send_message(id, "Неверный пункт меню")
        create_keyboard(id, "Выберите пункт меню", "admin")


def delete_question_answer(id, users, message, event, data):
    if is_canceled(id, message):
        create_keyboard(id, "Выберите пункт меню", "edit")
    elif is_intent(id, event.text, data):
        questions = create_answer_question_lists(event, data)
        formatted_answers = ""
        for i in range(len(questions)):
            formatted_answers += f"{i + 1}) {questions[i]}\n"
        send_message(id, f"Чтобы удалить ответ, выберите номер\n{formatted_answers}")
        users[id].state = f"choose_delete_question {event.text}"
    else:
        send_message(id, "Неверный пункт меню")
        create_keyboard(id, "Выберите пункт меню", "edit")


def delete_answer_answer(id, users, message, event, data):
    if is_canceled(id, message):
        create_keyboard(id, "Выберите пункт меню", "edit")
    elif is_intent(id, event.text, data):
        answers = create_answer_question_lists(event, data, False)
        formatted_answers = ""
        for i in range(len(answers)):
            formatted_answers += f"{i+1}) {answers[i]}\n"
        send_message(id, f"Чтобы удалить ответ, выберите номер\n{formatted_answers}")
        users[id].state = f"choose_delete_answer {event.text}"
    else:
        send_message(id, "Неверный пункт меню")
        create_keyboard(id, "Выберите пункт меню", "edit")


def delete_choosed_question(id, users, event, data):
    intent = users[id].state.split()[1]
    success = True
    if not event.text.isdigit():
        send_message(id, "То, что вы ввели, не является номером.")
        success = False
    else:
        number = int(event.text)
        if number == 0:
            number = 1
        number -= 1
        if number < 0:
            send_message(id, "Номер не может быть отрицательным.")
            success = False
        if len(data[intent]["examples"]) <= number:
            send_message(id, "В теме меньше вопросов, чем номер, который вы ввели.")
            success = False
        if success:
            deleted = data[intent]["examples"][number]
            users[id].state = f"confirm_question {intent} {number}"
            create_keyboard(id, f"Вопрос '{deleted}' будет удалён. Подтвердить действие?", "yesno")
            return
    if not success:
        users[id].state = "edit"
        create_keyboard(id, "Выберите пункт меню", "edit")


def delete_choosed_answer(id, users, event, data):
    intent = users[id].state.split()[1]
    success = True
    if not event.text.isdigit():
        send_message(id, "То, что вы ввели, не является номером.")
        success = False
    else:
        number = int(event.text)
        if number == 0:
            number = 1
        number -= 1
        if number < 0:
            send_message(id, "Номер не может быть отрицательным.")
            success = False
        if len(data[intent]["responses"]) <= number:
            send_message(id, "В теме меньше ответов, чем номер, который вы ввели.")
            success = False
        if success:
            deleted = data[intent]["responses"][number]
            users[id].state = f"confirm_answer {intent} {number}"
            create_keyboard(id, f"Ответ '{deleted}' будет удалён. Подтвердить действие?", "yesno")
            return
    if not success:
        users[id].state = "edit"
        create_keyboard(id, "Выберите пункт меню", "edit")


def confirmed_del_question(id, users, event, data):
    if event.text.lower() == "да":
        del data[users[id].state.split()[1]]["examples"][int(users[id].state.split()[2])]
        with open('jsons\\intents_dataset.json', 'w', encoding='UTF-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        send_message(id, "Вопрос был удалён.")
    else:
        send_message(id, "Действие было отменено.")
    users[id].state = "edit"
    create_keyboard(id, "Выберите пункт меню", "edit")


def confirmed_del_answer(id, users, event, data):
    if event.text.lower() == "да":
        del data[users[id].state.split()[1]]["responses"][int(users[id].state.split()[2])]
        with open('jsons\\intents_dataset.json', 'w', encoding='UTF-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        send_message(id, "Ответ был удалён.")
    else:
        send_message(id, "Действие было отменено.")
    users[id].state = "edit"
    create_keyboard(id, "Выберите пункт меню", "edit")


def delete_answer(id, users, data, message, event):
    users[id].state = "admin"
    if is_canceled(id, message):
        create_keyboard(id, "Выберите пункт меню", "edit")
    elif is_intent(id, event.text, data):
        del data[event.text]
        with open('jsons\\intents_dataset.json', 'w', encoding='UTF-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        send_message(id, "Тема была удалена")
        create_keyboard(id, "Выберите пункт меню", "edit")
    else:
        create_keyboard(id, "Выберите пункт меню", "edit")


def info_answer(id, users, message, event, data):
    users[id].state = "admin"
    if is_canceled(id, message):
        create_keyboard(id, "Выберите пункт меню", "admin")
    elif is_intent(id, event.text, data):
        answer = create_topic_answer(event, data)
        send_message(id, answer)
        create_keyboard(id, "Выберите пункт меню", "admin")
    else:
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
        users[id].state = "edit"
        create_keyboard(id, "Выберите пункт меню", "edit")
    elif is_intent(id, event.text, data):
        if users[id].state.split()[1] == "add_answer":
            send_message(id, "Вводите ответы, чтобы закончить, введите 0")
            users[id].state = "add_answer2" + " " + event.text
        elif users[id].state.split()[1] == "add_question":
            send_message(id, "Вводите вопросы, чтобы закончить, введите 0")
            users[id].state = "add_question2" + " " + event.text
    else:
        users[id].state = "edit"
        create_keyboard(id, "Выберите пункт меню", "edit")


def add_intent_answer(id, users, message, event, data):
    if is_canceled(id, message):
        users[id].state = "edit"
        create_keyboard(id, "Выберите пункт меню", "admin")
    else:
        intent = event.text.replace(" ", "-").lower()
        if is_intent(id, intent, data, True):
            send_message(id, "Такая тема уже есть")
            users[id].state = "edit"
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
        users[id].state = "edit"
        create_keyboard(id, "Выберите пункт меню", "edit")


def retrain_answer(id, users, message, data, model_mlp, vectorizer):
    if text_match(message, "да") or text_match(message, "yes"):
        users[id].state = "admin"
        vk.messages.setActivity(user_id=id, type='typing')
        Thread(target=learn_spell, args=(data,)).start()
        # neuro = make_neuronetwork()
        neuro = make_bertnetwork()
        model_mlp = neuro[0]
        vectorizer = neuro[1]
        fine_tuning(data, vectorizer)
        send_message(id, "Нейросеть переобучена")
        create_keyboard(id, "Выберите пункт меню", "admin")
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
            users[id].state = "edit"
            create_keyboard(id, "Выберите пункт меню", "edit")
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
            users[id].state = "edit"
            create_keyboard(id, "Выберите пункт меню", "edit")
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


def dates_answer(id, users, event):
    text = "Введите дату в формате '01.09', где первое число – дата, второе – месяц."
    if event.text == "1.Начало осеннего семестра":
        users[id].state = "start_autumn"
        send_message(id, text)
    elif event.text == "2.Конец осеннего семестра":
        users[id].state = "end_autumn"
        send_message(id, text)
    elif event.text == "3.Начало весеннего семестра":
        users[id].state = "start_spring"
        send_message(id, text)
    elif event.text == "4.Конец весеннего семестра":
        users[id].state = "end_spring"
        send_message(id, text)
    elif text_match(event.text, "Вернуться"):
        users[id].state = "admin"
        create_keyboard(id, "Выберите пункт меню", "admin")
    else:
        send_message(id, "Неверный пункт меню")
        create_keyboard(id, "Выберите пункт меню", "dates")


def change_date_answer(id, users, event, objects, starting_dates):
    flag = False
    try:
        parts = event.text.split(".")
        if not (parts[0].isdigit() and parts[1].isdigit()):
            flag = True
            raise ValueError("Ввод не является числом")
        number1 = int(parts[0])
        number2 = int(parts[1])
        if number1 <= 0 or number2 <= 0:
            flag = True
            raise ValueError("Дата должна быть > 0")
        if number1 > 31 or number2 > 12:
            flag = True
            raise ValueError("Недопустимые значения")
    except Exception as E:
        send_message(id, str(E))
        flag = True
    if not flag:
        number1 = str(int(parts[0]))
        number2 = str(int(parts[1]))
        if users[id].state == "start_autumn":
            objects["start-dates"][0] = f"{number2} {number1}"
            send_message(id, f"Начало осеннего семестра установлено на {number1}.{number2}")
        elif users[id].state == "end_autumn":
            objects["start-dates"][1] = f"{number2} {number1}"
            send_message(id, f"Конец осеннего семестра установлено на {number1}.{number2}")
        elif users[id].state == "start_spring":
            objects["start-dates"][2] = f"{number2} {number1}"
            send_message(id, f"Начало весеннего семестра установлено на {number1}.{number2}")
        elif users[id].state == "end_spring":
            objects["start-dates"][3] = f"{number2} {number1}"
            send_message(id, f"Конец весеннего семестра установлено на {number1}.{number2}")
        with open(os.path.join("jsons", "objects.json"), "w", encoding='UTF-8') as f:
            json.dump(objects, f, ensure_ascii=False, indent=4)
    users[id].state = "dates"
    starting_dates = get_starting_date(objects)
    create_keyboard(id, get_list_dates(starting_dates), "dates")
