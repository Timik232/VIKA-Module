import pickle
import random
import urllib.request
import vk_api
import sys
from neuro_defs import *
from learning_functions import *
from threading import Thread
from vk_api.longpoll import VkLongPoll, VkEventType
from keyboard import create_keyboard
import time


def is_intent(id, intent, data, is_new = False):
    if intent in data:
        return True
    else:
        if not is_new:
            send_message(id, "Тема не найдена")
        return False


def is_canceled(id, msg):
    if text_match(msg, 'отмена') or text_match(msg, 'отменить') or text_match(msg, 'назад'):
        send_message(id, "Отменено")
        return True
    else:
        return False


def main(model_mlp, data, vectorizer, dictionary, objects):
    answering("start",model_mlp,data,vectorizer, dictionary)
    print("Started")
    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me:
            if event.from_chat:
                continue
            id = event.user_id
            if not (id in users):  # если нет в базе данных
                users[id] = UserInfo()
                with open(f'{os.path.dirname(os.getcwd())}\\VIKA_pickle\\mirea_users.pickle', 'wb') as f:
                    pickle.dump(users, f)
            message = clean_up(event.text)  # очищенный текст
            if users[id].state == "Пожелания":
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
                    send_message(id, "Спасибо за ваше пожелание!")
            elif users[id].state == "waiting":
                users[id].state = ""
            elif users[id].state == "admin":
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
                    send_message(id,"Введите название темы")
                    users[id].state = "info"
                elif event.text == "5.Добавить ответ к теме":
                    send_message(id, "Введите название темы")
                    users[id].state = "check_intent add_answer"
                elif event.text == "6.Добавить вопрос к теме":
                    send_message(id, "Введите название темы")
                    users[id].state = "check_intent add_question"
                elif event.text == "7.Найти тему по вопросу" or text_match(message,"тема по вопросу"):
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

            elif users[id].state == "delete":
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
            elif users[id].state == "info":
                users[id].state = "admin"
                if is_canceled(id, message):
                    create_keyboard(id, "Выберите пункт меню", "admin")
                elif is_intent(id, event.text, data):
                    send_message(id, str(data[event.text]))
                    create_keyboard(id, "Выберите пункт меню", "admin")
            elif "check_intent" in users[id].state:
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
            elif "add_answer2" in users[id].state:
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
            elif "add_question2" in users[id].state:
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
            elif users[id].state == "add_intent":
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
            elif users[id].state == "is_end":
                if text_match(message, "да") or text_match(message, "yes"):
                    send_message(id, "Вводите тему")
                    users[id].state = "add_intent"
                elif text_match(message, "нет") or text_match(message, "no"):
                    users[id].state = "admin"
                    create_keyboard(id, "Выберите пункт меню", "admin")
            elif users[id].state == "retrain":
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
            elif users[id].state == "get_topic":
                send_message(id,get_intent(message,model_mlp, vectorizer,dictionary))
                users[id].state = "admin"
                create_keyboard(id, "Выберите пункт меню", "admin")
            elif users[id].state == "statistic":
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

            if users[id].state == "":
                room = check_room(event.text)
                is_important_room = False
                for r in objects["rooms"]:
                    if r in message:
                        is_important_room = True
                        break
                if not is_important_room:
                    if room == "78":
                        create_keyboard(id, "Используйте навигатор по Университету", "map")
                        continue
                    elif room == "86":
                        create_keyboard(id, "Аудитория располагается в кампусе на Проспекте Вернадского, д.86.", "maps")
                        continue
                    elif room == "kpk_20":
                        create_keyboard(id, "Вероятно, аудитория располагается на Стромынке 20 или "
                                            "в колледже РТУ МИРЭА.", "maps")
                        continue
                    elif room == "not_found":
                        send_message(id, "Такая аудитория не была обнаружена")
                        continue
                if message == "админпанель" or message == "админ панель":
                    if id == 286288768:
                        send_message(id, "Доступ получен, чтобы выйти из администраторского режима напишите: 'выход'")
                        users[id].state = "admin"
                        create_keyboard(id, "Выберите пункт меню", "admin")
                    else:
                        send_message(id, "You are not supposed to be here")
                elif clean_up(message) == "расписание":
                    send_message(id, "В данном боте тестируется только система ответов на вопросы, расписание в основной "
                                     "версии ВИКА")
                elif clean_up(message) == "карта университета":
                    create_keyboard(id, "Используйте навигатор по Университету", "map")
                elif clean_up(message) == "рассылка":
                    send_message(id, "В данном боте тестируется только система ответов на вопросы, рассылка в основной "
                                     "версии ВИКА")
                elif clean_up(message) == "расписание пересдач":
                    send_message(id, "В данном боте тестируется только система ответов на вопросы, расписание пересдач в"
                                     " основной версии ВИКА")
                elif clean_up(message) == "что ты умеешь":
                    send_message(id, "Напишите мне любой вопрос, связанный с нашим университетом, и я постараюсь найти "
                                     "ответ на него. Учтите, что я не живой "
                                     "человек и могу ошибаться, однако в этой версии база ответов значительно расширена, а система "
                                     "распознавания вопросов улучшена.")
                elif clean_up(message) == "обратная связь":
                    send_message(id, "Введите в следующем сообщении свои пожелания по улучшению бота(в том числе можно "
                                     "указать вопрос, на который вы хотели бы, чтобы бот мог отвечать. Если знаете, "
                                     "то ещё и сразу ответ). Они будут переданы разработчику. Если хотите отменить "
                                     "отправку, напишите 'Отмена'")
                    users[id].state = "Пожелания"
                elif clean_up(message) == "оценить бота":
                    create_keyboard(id, "Вы можете поставить лайк или дизлайк боту", "rating")
                else:
                    answer = answering(message, model_mlp, data, vectorizer, dictionary)
                    if answer[1] == "feedback":
                        send_message(id,
                                     "Введите в следующем сообщении свои пожелания по улучшению бота. Они будут "
                                     "переданы разработчику. Если хотите отменить отправку, напишите 'Отмена'")
                        users[id].state = "Пожелания"
                    elif answer[1] == "panda":
                        send_photo(id, "файлы/panda.jpg", answer[0])
                    elif answer[1] == "like":
                        users[id].like = 1
                        with open(f'{os.path.dirname(os.getcwd())}\\VIKA_pickle\\mirea_users.pickle', 'wb') as f:
                            pickle.dump(users, f)
                        create_keyboard(id, answer[0])
                    elif answer[1] == "dislike":
                        users[id].like = -1
                        with open(f'{os.path.dirname(os.getcwd())}\\VIKA_pickle\\mirea_users.pickle', 'wb') as f:
                            pickle.dump(users, f)
                        create_keyboard(id, answer[0])
                    elif answer[1] == "none":
                        users[id].like = 0
                        with open(f'{os.path.dirname(os.getcwd())}\\VIKA_pickle\\mirea_users.pickle', 'wb') as f:
                            pickle.dump(users, f)
                        create_keyboard(id, answer[0])
                    elif answer[1] == "f-bot":
                        send_photo(id, "файлы/f-bot.jpg", answer[0])
                    else:
                        create_keyboard(id, answer[0], answer[1])


if __name__ == "__main__":
    # parsing()
    if not os.path.exists(f'{os.path.dirname(os.getcwd())}\\VIKA_pickle'):
        os.mkdir(f"{os.path.dirname(os.getcwd())}\\VIKA_pickle")
    with open('jsons\\intents_dataset.json', 'r', encoding='UTF-8') as f:
        data = json.load(f)
    with open('jsons\\objects.json', 'r', encoding='UTF-8') as f:
        objects = json.load(f)
    if not os.path.isfile(f'{os.path.dirname(os.getcwd())}\\VIKA_pickle\\model.pkl'):
        # neuro = make_neuronetwork()
        neuro = make_bertnetwork()
        model_mlp = neuro[0]
        vectorizer = neuro[1]
    else:
        with open(f'{os.path.dirname(os.getcwd())}\\VIKA_pickle\\model.pkl', 'rb') as f:
            model_mlp = pickle.load(f)
        with open(f'{os.path.dirname(os.getcwd())}\\VIKA_pickle\\vector.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("Обученная модель загружена")

    if os.path.isfile(f'{os.path.dirname(os.getcwd())}\\VIKA_pickle\\mirea_users.pickle'):
        with open(f'{os.path.dirname(os.getcwd())}\\VIKA_pickle\\mirea_users.pickle', 'rb') as f:
            users = pickle.load(f)
            print("Пользователи загружены")
    if os.path.isfile(f'{os.path.dirname(os.getcwd())}\\VIKA_pickle\\dictionary.pickle'):
        with open(f'{os.path.dirname(os.getcwd())}\\VIKA_pickle\\dictionary.pickle', 'rb') as f:
            dictionary = pickle.load(f)
        print("Словарь загружен")
    else:
        print("Загрузка словаря...")
        Thread(target=learn_spell, args=(data,)).start()
    Thread(target=add_answer, args=(users,)).start()
    # fine_tuning(data, vectorizer, model_mlp, dictionary)
    longpoll = VkLongPoll(vk_session)
    while True:
        try:
            main(model_mlp, data, vectorizer, dictionary, objects)
        except requests.exceptions.ReadTimeout:
            print("read-timeout")
            time.sleep(600)
        except Exception as ex:
            print(ex)
