import vk_api
import os
import random
import json
import pickle
import urllib.request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from vk_api.longpoll import VkLongPoll, VkEventType
from vk_api.keyboard import VkKeyboard, VkKeyboardColor
from vk_api.utils import get_random_id
from neuro_defs import *
from threading import Thread



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
        model_mlp = MLPClassifier(hidden_layer_sizes=100, activation='relu', solver='adam', learning_rate='adaptive', max_iter=1000)
        model_mlp.fit(X_vec, y)
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
    Thread(target=add_answer, args=()).start()
    if os.path.isfile('mirea_users.pickle'):
        with open('mirea_users.pickle', 'rb') as f:
            users = pickle.load(f)

    longpoll = VkLongPoll(vk_session)
    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me:
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
                    for i in users.keys():
                        if users[i].name == "ser13volk":
                            user_get = vk.users.get(user_ids=id)
                            first_name = user_get[0]['first_name']
                            last_name = user_get[0]['last_name']
                            send_message(i, "Пользователь " + first_name + " " + last_name + " хочет сказать: " + event.text)
                    send_message(id, "Спасибо за ваше пожелание!")
            elif users[id].state == "waiting":
                users[id].state = ""
            if text_match(message,"расписание"):
                send_message(id,"В данном боте тестируется только система ответов на вопросы, расписание в основной версии ВИКА")
            elif text_match(message,"карта"):
                create_keyboard(id,"Используйте навигатор по Университету", "map")
            elif text_match(message,"рассылка"):
                send_message(id,"В данном боте тестируется только система ответов на вопросы, рассылка в основной версии ВИКА")
            elif text_match(message,"Расписание пересдач"):
                send_message(id,"В данном боте тестируется только система ответов на вопросы, расписание пересдач в основной версии ВИКА")
            elif text_match(message,"Что ты умеешь"):
                send_message(id,"Напишите мне любой вопрос, связанный с нашим университетом, и я постараюсь найти ответ на него. Учтите, что я не живой "
                                "человек и могу ошибаться, однако в этой версии база ответов значительно расширена, а система "
                                "распознавания вопросов улучшена.")
            elif text_match(message,"Пожелания по улучшению"):
                send_message(id,"Введите в следующем сообщении свои пожелания по улучшению бота. Они будут переданы разработчику. Если хотите отменить отправку, напишите 'Отмена'")
                users[id].state = "Пожелания"
            else:
                if users[id].state == "":
                    answer = answering(message, model_mlp, data, vectorizer)
                    if answer[1] == "not_that" or answer[1] == "help_me" or answer[1] == "callhuman":
                        create_keyboard(id, answer[0], "yaro")
                    elif answer[1] == "grifon":
                        create_keyboard(id, answer[0], "grifon")
                    elif answer[1] == "danger" or answer[1] == "feedback_bad" or answer[1] == "motivation":
                        create_keyboard(id, answer[0], "psychology")
                    elif answer[1] == "feedback":
                        send_message(id,
                                     "Введите в следующем сообщении свои пожелания по улучшению бота. Они будут переданы разработчику. Если хотите отменить отправку, напишите 'Отмена'")
                        users[id].state = "Пожелания"
                    elif answer[1] == "rules":
                        create_keyboard(id, answer[0], "rules")
                    elif answer[1] == "museums":
                        create_keyboard(id, answer[0], "museums")
                    elif answer[1] == "maps":
                        upload = vk_api.VkUpload(vk_session)
                        file = upload.document("файлы/карты.pdf", "doc")
                        send_message(id, answer[0], None, file)
                    elif answer[1] == "metodichka":
                        upload = vk_api.VkUpload(vk_session)
                        file = upload.document("файлы/методичка_2022.pdf", "doc")
                        send_message(id, answer[0], None, file)
                    elif answer[1] == "obhodnoy":
                        create_keyboard(id, answer[0], "obhodnoy")
                    elif answer[1] == "work":
                        create_keyboard(id, answer[0], "work")
                    elif answer[1] == "website":
                        create_keyboard(id, answer[0], "website")
                    elif answer[1] == "military":
                        create_keyboard(id, answer[0], "military")
                    elif answer[1] == "rectorate":
                        create_keyboard(id, answer[0], "rectorate")
                    elif answer[1] == "library":
                        create_keyboard(id, answer[0], "library")
                    elif answer[1] == "office":
                        create_keyboard(id, answer[0], "office")
                    elif answer[1] == "scholarship":
                        create_keyboard(id, answer[0], "scholarship")
                    else:
                        create_keyboard(id, answer[0])
