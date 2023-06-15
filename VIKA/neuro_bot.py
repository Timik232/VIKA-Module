import pickle
import random
import urllib.request
import vk_api
import sys
import time
from neuro_defs import *
from learning_functions import *
from threading import Thread
from vk_api.longpoll import VkLongPoll, VkEventType
from keyboard import create_keyboard
from answer_functions import *
from find_panda import get_alexnet, show_prediction, Alexnet
from translation import translate_to_en
import tempfile
from UserClass import UserInfo


def save_image_from_url(image_url, file_name):
    urllib.request.urlretrieve(image_url, file_name)
    print(F'File "{file_name}" was saved to temporary')


def main(model_mlp, data, vectorizer, dictionary, objects, alexnet):
    answering("start", model_mlp, data, vectorizer, dictionary, objects)
    starting_dates = get_starting_date(objects)
    print("Bot is listening")
    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me:
            if event.from_chat:
                continue
            api_message = vk_session.method("messages.getById", {
                "message_ids": [event.message_id],
                "group_id": 213226596
            })
            id = event.user_id

            if not (id in users):  # если нет в базе данных
                users[id] = UserInfo()
                with open(os.path.join(cwd(), 'VIKA-pickle', 'mirea_users.pickle'), 'wb') as f:
                    pickle.dump(users, f)
            message = clean_up(event.text)  # очищенный текст

            if users[id].state == "Пожелания":
                feedback_answer(id, users, message, event)
            elif users[id].state == "waiting":
                users[id].state = ""
            elif users[id].state == "admin":
                admin_answer(id, users, message, event, data)

            elif users[id].state == "delete":
                delete_answer(id, users, data, message, event)
            elif users[id].state == "info":
                info_answer(id, users, message, event, data)
            elif "check_intent" in users[id].state:
                check_intent_answer(id, users, message, event, data)
            elif "add_answer2" in users[id].state:
                add_answer_answer(id, users, event, data)
            elif "add_question2" in users[id].state:
                add_question_answer(id, users, event, data)
            elif users[id].state == "add_intent":
                add_intent_answer(id, users, message, event, data)
            elif users[id].state == "is_end":
                is_end_answer(id, users, message)
            elif users[id].state == "retrain":
                retrain_answer(id, users, message, data)
            elif users[id].state == "get_topic":
                send_message(id, get_intent_bert(message, model_mlp, vectorizer, dictionary, objects))
                users[id].state = "admin"
                create_keyboard(id, "Выберите пункт меню", "admin")
            elif users[id].state == "statistic":
                statistic_answer(id, users, message, event, data)
            elif users[id].state == "edit":
                edit_answer(id, users, message, event)
            elif users[id].state == "delete_question":
                delete_question_answer(id, users, message, event, data)
            elif users[id].state == "delete_answer":
                delete_answer_answer(id, users, message, event, data)
            elif users[id].state == "get_full_topic":
                get_full_topic_answer(id, users, message, event, data, model_mlp, vectorizer, dictionary, objects)
            elif len(users[id].state.split()) > 1:
                if users[id].state.split()[0] == "choose_delete_question":
                    delete_choosed_question(id, users, event, data)
                elif users[id].state.split()[0] == "choose_delete_answer":
                    delete_choosed_answer(id, users, event, data)
                elif users[id].state.split()[0] == "confirm_question":
                    confirmed_del_question(id, users, event, data)
                elif users[id].state.split()[0] == "confirm_answer":
                    confirmed_del_answer(id, users, event, data)

            photo = None
            try:
                photo = api_message["items"][0]["attachments"][0]["photo"]
            except Exception:
                pass
            if photo is not None:
                # save_image_from_url(photo["sizes"][2]["url"], "ispanda.jpg")
                temp_dir = tempfile.TemporaryDirectory()
                save_path = temp_dir.name + "/" + "ispanda.jpg"
                save_image_from_url(photo["sizes"][2]["url"], save_path)
                Thread(target=show_prediction, args=(id, alexnet, save_path, temp_dir)).start()
                users[id].state = "waiting"

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
                    send_message(id,
                                 "В данном боте тестируется только система ответов на вопросы, расписание в основной "
                                 "версии ВИКА")
                elif clean_up(message) == "карта университета":
                    create_keyboard(id, "Используйте навигатор по Университету", "map")
                elif clean_up(message) == "рассылка":
                    send_message(id, "В данном боте тестируется только система ответов на вопросы, рассылка в основной "
                                     "версии ВИКА")
                elif clean_up(message) == "расписание пересдач":
                    send_message(id,
                                 "В данном боте тестируется только система ответов на вопросы, расписание пересдач в"
                                 " основной версии ВИКА")
                elif clean_up(message) == "что ты умеешь":
                    send_message(id, "Напишите мне любой вопрос, связанный с нашим университетом, и я постараюсь найти "
                                     "ответ на него. Учтите, что я не живой "
                                     "человек и могу ошибаться, однако в этой версии база ответов значительно расширена."
                                     " Также вы можете отправить мне любое изображение и я попытаюсь угадать, есть ли"
                                     "панда на картинке, или нет.")
                elif clean_up(message) == "обратная связь":
                    send_message(id, "Введите в следующем сообщении свои пожелания по улучшению бота(в том числе можно "
                                     "указать вопрос, на который вы хотели бы, чтобы бот мог отвечать. Если знаете, "
                                     "то ещё и сразу ответ). Они будут переданы разработчику. Если хотите отменить "
                                     "отправку, напишите 'Отмена'")
                    users[id].state = "Пожелания"
                elif clean_up(message) == "оценить бота":
                    create_keyboard(id, "Вы можете поставить лайк или дизлайк боту", "rating")
                elif clean_up(message) == "english":
                    users[id].language = "en"
                    create_keyboard(id, "Language was changed to english.", "start-english", users)
                elif clean_up(message) == "русский":
                    users[id].language = "ru"
                    create_keyboard(id, "Язык был изменен на русский.", "ru")
                elif clean_up(message) == "what do you can":
                    send_message(id, "Write me any question related to our university, and I will try to find the "
                                     "answer to it. Please note that I am not a living "
                                     "person and I can be wrong, but in this version the database of answers has been"
                                     " significantly expanded."
                                     " You can also send me any image and I'll try to guess if there is"
                                     "panda in the picture, or not.")
                elif clean_up(message) == "schedule":
                    send_message(id, "Not working at this version")
                elif clean_up(message) == "mailing":
                    send_message(id, "Not working at this version")
                elif clean_up(message) == "retake schedule":
                    send_message(id, "Not working at this version")
                elif clean_up(message) == "university map":
                    create_keyboard(id, "Use the navigator", "map")
                elif clean_up(message) == "rate the bot":
                    create_keyboard(id, "You can like or dislike the bot", "rating")
                elif clean_up(message) == "feedback":
                    send_message(id, "Enter in the next message your feedback, it will be sent to the creator.")
                    users[id].state = "Пожелания"
                else:
                    answer = answering(message, model_mlp, data, vectorizer, dictionary, objects)
                    if answer[1] == "feedback":
                        if users[id].language != "en":
                            send_message(id,
                                         "Введите в следующем сообщении свои пожелания по улучшению бота. Они будут "
                                         "переданы разработчику. Если хотите отменить отправку, напишите 'Отмена'")
                        else:
                            send_message(id, "Enter in the next message your feedback, it will be sent to the creator.")
                        users[id].state = "Пожелания"
                    elif answer[1] == "panda":
                        send_photo(id, "files/panda.jpg", answer[0])
                    elif answer[1] == "like":
                        users[id].like = 1
                        with open(os.path.join(cwd(), 'VIKA-pickle', 'mirea_users.pickle'), 'wb') as f:
                            pickle.dump(users, f)
                        if users[id].language == "en":
                            create_keyboard(id, "I am glad, like is fixed", "en", users)
                        else:
                            create_keyboard(id, answer[0])
                    elif answer[1] == "dislike":
                        users[id].like = -1
                        with open(os.path.join(cwd(), 'VIKA-pickle', 'mirea_users.pickle'), 'wb') as f:
                            pickle.dump(users, f)
                        if users[id].language == "en":
                            create_keyboard(id, "I am sorry, if I have bad realization, dislike is fixed", "en", users)
                        else:
                            create_keyboard(id, answer[0])
                    elif answer[1] == "none":
                        users[id].like = 0
                        with open(os.path.join(cwd(), 'VIKA-pickle', 'mirea_users.pickle'), 'wb') as f:
                            pickle.dump(users, f)
                        create_keyboard(id, answer[0])
                    elif answer[1] == "f-bot":
                        send_photo(id, "files/f-bot.jpg", answer[0])
                    elif answer[1] == "номер-недели":
                        if is_teaching_week(starting_dates[0], starting_dates[1], starting_dates[2], starting_dates[3]):
                            if users[id].language == "en":
                                send_message(id, send_message(id, f"Now is {week_number(starting_dates[0], starting_dates[2])} week"))
                            else:
                                send_message(id, f"Сейчас идёт {week_number(starting_dates[0], starting_dates[2])} неделя")
                        else:
                            if users[id].language == "en":
                                send_message(id, "Current week isn't educational")
                            else:
                                send_message(id, "Текущая неделя не является основной учебной неделей.")
                    else:
                        if users[id].language == "en":
                            create_keyboard(id, translate_to_en(answer[1], answer[0], data), answer[1], users)
                        else:
                            # главная функция отправки сообщений на все запросы
                            create_keyboard(id, answer[0], answer[1])


if __name__ == "__main__":
    print("started")
    # parsing(1000) # Использовать если нужно ещё запарсить ответы со справочной
    device = "cpu"
    if not os.path.exists(os.path.join(cwd(), 'VIKA-pickle')):
        os.mkdir(os.path.join(cwd(), 'VIKA-pickle'))
    with open(os.path.join('jsons', 'intents_dataset.json'), 'r', encoding='UTF-8') as f:
        data = json.load(f)
    with open(os.path.join('jsons', 'objects.json'), 'r', encoding='UTF-8') as f:
        objects = json.load(f)
    if not os.path.isfile(os.path.join(cwd(), 'VIKA-pickle','model.pkl')):
        # neuro = make_neuronetwork()
        neuro = make_bertnetwork()
        model_mlp = neuro[0]
        vectorizer = neuro[1]
        fine_tuning(data, vectorizer, model_mlp, dictionary)
    else:
        with open(os.path.join(cwd(), 'VIKA-pickle', 'model.pkl'), 'rb') as f:
            model_mlp = pickle.load(f)
        # with open(f'{cwd()}{slh()}VIKA-pickle{slh()}vector.pkl', 'rb') as f:
        #     # vectorizer = pickle.load(f)
        #     # vectorizer = CPU_Unpickler(f).load()
        #     vectorizer = torch.load(f, map_location=torch.device('cpu'))
        vectorizer = SentenceTransformer('distiluse-base-multilingual-cased')
        vectorizer.load_state_dict(torch.load(os.path.join(cwd(), 'VIKA-pickle', 'vector.pt'), map_location='cpu'))
        print("Model loaded")

    if os.path.isfile(os.path.join(cwd(), 'VIKA-pickle', 'mirea_users.pickle')):
        with open(os.path.join(cwd(), 'VIKA-pickle', 'mirea_users.pickle'), 'rb') as f:
            users = pickle.load(f)
            print("Users loaded")
    if os.path.isfile(os.path.join(cwd(), 'VIKA-pickle', 'dictionary.pickle')):
        with open(os.path.join(cwd(), 'VIKA-pickle', 'dictionary.pickle'), 'rb') as f:
            dictionary = pickle.load(f)
        print("Dictionary loaded")
    else:
        print("Dictionary loading...")
        Thread(target=learn_spell, args=(data,)).start()
    alexnet = get_alexnet()
    # Thread(target=add_answer, args=(users,)).start()
    longpoll = VkLongPoll(vk_session)
    while True:
        try:
            main(model_mlp, data, vectorizer, dictionary, objects, alexnet)
        except requests.exceptions.ReadTimeout:
            print("read-timeout")
            time.sleep(600)
        except Exception as ex:
            print(ex)
