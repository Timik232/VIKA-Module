import os
import pickle
import random
import requests
import urllib.request
import vk_api
from neuro_defs import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from threading import Thread
from vk_api.keyboard import VkKeyboard, VkKeyboardColor
from vk_api.longpoll import VkLongPoll, VkEventType
from vk_api.utils import get_random_id
from vk_api.bot_longpoll import VkBotEventType
from private_api import password  # пароль для админ панели

def is_intent(id, intent, data):
    if intent in data:
        return True
    else:
        send_message(id, "Тема не найдена")
        return False


def is_canceled(id, msg):
    if text_match(msg, 'отмена'):
        send_message(id, "Отменено")
        return True
    else:
        return False


if __name__ == "__main__":
    if not os.path.isfile('model.pkl'):
        with open('intents_dataset.json', 'r', encoding='UTF-8') as f:
            data = json.load(f)
        X = []
        y = []

        for name in data:
            for question in data[name]['examples']:
                X.append(question)
                y.append(name)
            for phrase in data[name]['responses']:
                X.append(phrase)
                y.append(name)

        # векторизируем файлы и обучаем модель

        vectorizer = CountVectorizer()
        X_vec = vectorizer.fit_transform(X)
        model_mlp = MLPClassifier(hidden_layer_sizes=300, activation='relu', solver='adam', learning_rate='adaptive',
                                  max_iter=1500)
        model_mlp.fit(X_vec, y)
        y_pred = model_mlp.predict(X_vec)
        print("точность " + str(accuracy_score(y, y_pred)))
        print("f1 " + str(f1_score(y, y_pred, average='macro')))
        with open('model.pkl', 'wb') as f:
            pickle.dump(model_mlp, f)
        with open('vector.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)

        print("Обучено")
    else:
        with open('intents_dataset.json', 'r', encoding='UTF-8') as f:
            data = json.load(f)
        with open('model.pkl', 'rb') as f:
            model_mlp = pickle.load(f)
        with open('vector.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("Обученная модель загружена")
    if os.path.isfile('mirea_users.pickle'):
        with open('mirea_users.pickle', 'rb') as f:
            users = pickle.load(f)
    Thread(target=add_answer, args=(users,)).start()

    longpoll = VkLongPoll(vk_session)
    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me:
            if event.from_chat:
                continue
            id = event.user_id
            if not (id in users):  # если нет в базе данных
                users[id] = UserInfo()
                with open('mirea_users.pickle', 'wb') as f:
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
            elif users[id].state == "password":
                if text_match(message, "отмена"):
                    users[id].state = "waiting"
                    send_message(id, "Ввод отменён")
                elif event.text == password:
                    send_message(id, "Доступ получен, чтобы выйти из администраторского режима напишите 'выход'")
                    users[id].state = "admin"
                    create_keyboard(id, "Выберите пункт меню:\n\n\n\n4.Удалить тему\n5.Вывести всю информацию по теме\n6.Добавить ответ к теме\n7.Добавить вопрос к теме\n8.Вывести количество пользователей\n9.Вывести рейтинг", "admin")
            elif users[id].state == "admin":
                if text_match(message, "выход"):
                    users[id].state = "waiting"
                    create_keyboard(id, "Выход выполнен, можете снова пользоваться ботом")
                if text_match(message, "1.Вывести количество тем"):
                    send_message(id, "Количество тем: " + str(len(data)))
                elif text_match(message,"2.Вывести все темы"):
                    smg = ""
                    for i in data:
                        smg += i + "\n"
                    send_message(id, smg)
                elif text_match(message, "3.Добавить тему"):
                    send_message(id, "Введите название темы")
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
                    print("Ответ был записан в файл. Ввести еще ответ? (y/n)")
                    end = input()
                    end = end.lower()
                    if end == "n" or end == "no" or end == "нет":
                        break
                    else:
                        print("Продолжайте вводить")
                elif text_match(message, "4.Удалить тему"):
                    send_message(id, "Введите название темы")
                    users[id].state = "delete"
                elif text_match(message, "5.Вывести всю информацию по теме"):
                    send_message(id,"Введите название темы")
                    users[id].state = "info"
                elif text_match(message, "6.Добавить ответ к теме"):
                    send_message(id, "Введите название темы")
                    users[id].state = "add_answer"
                elif text_match(message, "7.Добавить вопрос к теме"):
                    print("Введите название темы")
                    intent = input()
                    print("Вводите вопросы, чтобы закончить, введите 0")
                    while True:
                        question = input()
                        if question == "0":
                            break
                        data[intent]['examples'].append(question)
                    with open('intents_dataset.json', 'w', encoding='UTF-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)
                    print("Вопросы были записаны в файл.")
                elif text_match(message, "8.Количество пользователей бота"):
                    send_message(id, "Количество пользователей бота: " + str(len(users)))
                elif text_match(message, "9.Вывести рейтинг бота"):
                    rate = 0
                    for i in users:
                        rate += users[i].like
                    send_message(id, "Рейтинг бота (количество лайков минус количество дизлайков): " + str(rate))
                else:
                    send_message(id, "Неверный пункт меню")
            elif users[id].state == "delete":
                users[id].state = "admin"
                if is_canceled(id, message):
                    continue
                elif is_intent(id, event.text, data):
                    del data[event.text]
                    with open('intents_dataset.json', 'w', encoding='UTF-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)
                    send_message(id, "Тема была удалена")
            elif users[id].state == "info":
                users[id].state = "admin"
                if is_canceled(id, message):
                    continue
                elif is_intent(id, event.text, data):
                    send_message(id, data[event.text])
            elif users[id].state == "add_answer":
                users[id].state = "admin"
                if is_canceled(id, message):
                    continue
                elif is_intent(id, event.text, data):
                    send_message(id, "Вводите ответы, чтобы закончить, введите 0")
                    users[id].state = "add_answer2" + " " + event.text
            elif "add_answer2" in users[id].state:
                intent = users[id].state.split()[1]
                answer = event.text
                if answer == "0":
                    users[id].state = "finish_answer"
                data[intent]['responses'].append(answer)
            elif users[id].state == "finish_answer":
                with open('intents_dataset.json', 'w', encoding='UTF-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                send_message(id, "Ответ был записан в файл.")
                users[id].state = "admin"


            if message == "админпанель" or message == "админ панель":
                send_message(id, "Введите пароль")
                users[id].state = "password"
            elif text_match(message, "расписание"):
                send_message(id, "В данном боте тестируется только система ответов на вопросы, расписание в основной "
                                 "версии ВИКА")
            elif text_match(message, "карта Университета"):
                create_keyboard(id, "Используйте навигатор по Университету", "map")
            elif text_match(message, "рассылка"):
                send_message(id, "В данном боте тестируется только система ответов на вопросы, рассылка в основной "
                                 "версии ВИКА")
            elif text_match(message, "Расписание пересдач"):
                send_message(id, "В данном боте тестируется только система ответов на вопросы, расписание пересдач в "
                                 "основной версии ВИКА")
            elif text_match(message, "Что ты умеешь"):
                send_message(id, "Напишите мне любой вопрос, связанный с нашим университетом, и я постараюсь найти "
                                 "ответ на него. Учтите, что я не живой "
                                 "человек и могу ошибаться, однако в этой версии база ответов значительно расширена, а система "
                                 "распознавания вопросов улучшена.")
            elif text_match(message, "Пожелания по улучшению"):
                send_message(id, "Введите в следующем сообщении свои пожелания по улучшению бота(в том числе можно "
                                 "указать вопрос, на который вы хотели бы, чтобы бот мог отвечать. Если знаете, "
                                 "то ещё и сразу ответ). Они будут переданы разработчику. Если хотите отменить "
                                 "отправку, напишите 'Отмена'")
                users[id].state = "Пожелания"
            elif text_match(message, "Оценить бота"):
                create_keyboard(id, "Вы можете поставить лайк или дизлайк боту", "rating")
            else:
                if users[id].state == "":
                    answer = answering(message, model_mlp, data, vectorizer)
                    if answer[1] == "feedback":
                        send_message(id,
                                     "Введите в следующем сообщении свои пожелания по улучшению бота. Они будут "
                                     "переданы разработчику. Если хотите отменить отправку, напишите 'Отмена'")
                        users[id].state = "Пожелания"
                    elif answer[1] == "panda":
                        send_photo(id, "файлы/panda.jpg", answer[0])
                    elif answer[1] == "like":
                        users[id].like = 1
                        with open('mirea_users.pickle', 'wb') as f:
                            pickle.dump(users, f)
                    elif answer[1] == "dislike":
                        users[id].like = -1
                        with open('mirea_users.pickle', 'wb') as f:
                            pickle.dump(users, f)
                    elif answer[1] == "none":
                        users[id].like = 0
                        with open('mirea_users.pickle', 'wb') as f:
                            pickle.dump(users, f)
                    elif answer[1] == "maps":
                        result = json.loads(requests.post(
                            vk.docs.getMessagesUploadServer(type='doc', peer_id=id)[
                                'upload_url'], files={'file': open('файлы/карты.pdf', 'rb')}).text)
                        jsonAnswer = vk.docs.save(file=result['file'], title='title', tags=[])
                        vk.messages.send(peer_id=id, message=answer[0], random_id=0,
                                         attachment=f"doc{jsonAnswer['doc']['owner_id']}_{jsonAnswer['doc']['id']}")
                        """upload = vk_api.VkUpload(vk_session)
                        file = upload.document_message("файлы/карты.pdf")[0]
                        owner_id = file['owner_id']
                        doc_id = file['id']
                        attachment = f'document{owner_id}_{doc_id}'
                        post = {'user_id': id, 'random_id': 0, "attachment": attachment}
                        post['message'] = answer[0]
                        try:
                            vk_session.method('messages.send', post)
                        except BaseException as Ex:
                            print(Ex)
                            """
                        # send_message(id, answer[0], None, file)
                    else:
                        create_keyboard(id, answer[0], answer[1])
