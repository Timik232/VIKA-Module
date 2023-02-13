import pickle

from vk_api.keyboard import VkKeyboard, VkKeyboardColor
from vk_api.utils import get_random_id
import vk_api
from private_api import token_api  # токен который не должен быть у всех, поэтому вынес в отдельный файл.
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
import random
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import nltk

nltk.download('punkt')
nltk.download('russian')

from nltk.tokenize import word_tokenize
from nltk.corpus import russian

word_list = set(russian.words())

def spell_check(text):
    misspelled_words = []
    tokenized_words = word_tokenize(text, language='russian')
    for word in tokenized_words:
        if word.lower() not in word_list:
            misspelled_words.append(word)
    return misspelled_words


vk_session = vk_api.VkApi(token=token_api)
vk = vk_session.get_api()


# класс с информацией о пользователе, чтобы можно было запоминать информацию о пользователе, в том числе о рассылках.
# не знаю, как это реализовано в основном боте, поэтому пусть будет так
class UserInfo:
    def __init__(self):
        # предложение запоминать группу пользователя, чтобы не вводить её каждый раз
        self.group = ""
        self.like = 0  # нравится/не нравится бот
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


def send_photo(user_id, img_req, message = None):
    upload = vk_api.VkUpload(vk_session)
    photo = upload.photo_messages(img_req)[0]
    owner_id = photo['owner_id']
    photo_id = photo['id']
    attachment = f'photo{owner_id}_{photo_id}'
    post = {'user_id': user_id, 'random_id': 0, "attachment": attachment}
    if message != None:
        post['message'] = message
    try:
        vk_session.method('messages.send', post)
    except BaseException:
        send_message(id, "Не удалось отправить картинку")
        return


def make_neuronetwork():
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
    model_mlp = MLPClassifier(hidden_layer_sizes=322, activation='relu', solver='adam', learning_rate='adaptive',
                              max_iter=1500)
    model_mlp.fit(X_vec, y)
    y_pred = model_mlp.predict(X_vec)
    print("точность " + str(accuracy_score(y, y_pred)))
    print("f1 " + str(f1_score(y, y_pred, average='macro')))
    with open('model.pkl', 'wb') as f:
        pickle.dump(model_mlp, f)
    with open('vector.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    neuro = []
    neuro.append(model_mlp)
    neuro.append(vectorizer)
    print("Обучено")
    return neuro


# на данный момент только одна клавиатура, но есть возможность создавать другие
def create_keyboard(id, text, response="start"):
    try:
        keyboard = VkKeyboard(one_time=True)
        if response == "not_that" or response == "help_me" or response == "callhuman" or response == "flood":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Ссылка на ВК', "https://vk.com/bramind002")
        elif response == "grifon":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Стикеры с грифоном в ТГ', "https://t.me/addstickers/rtumirea")
        elif response == "psychology" or response == "danger" or response == "feedback_bad" or response == "motivation" or response == "psycho" or response == "psychologist":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Психологическая служба',
                                         "https://student.mirea.ru/psychological_service/staff/")
        elif response == "map":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Навигатор', "https://ischemes.ru/group/rtu-mirea/vern78")
        elif response == "rules":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Устав', "https://www.mirea.ru/upload/medialibrary/d0e/Ustav-Novyy.pdf")
            keyboard.add_openlink_button('Правила внутреннего распорядка', "https://www.mirea.ru/docs/125641/")
            keyboard.add_openlink_button('Этический кодекс',
                                         "https://student.mirea.ru/regulatory_documents/file/3f9468db49ffd14fe96c0d28d8c056bf.pdf")
        elif response == "museums":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Подробнее о музеях',
                                         "https://www.mirea.ru/about/history-of-the-university/the-museum-mirea/")
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
            keyboard.add_openlink_button('Размер стипендии',
                                         "https://student.mirea.ru/scholaship_support/scholarships/state_academic_support/")
        elif response == "social-money":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Материальная помощь', "https://vk.com/topic-42869722_48644800")
            keyboard.add_openlink_button('Бланки', "https://student.mirea.ru/statement/")
        elif response == "subsidy":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Вопросы по дотациям', "https://vk.com/@rtuprofkom-voprosy-po-dotaciyam")
        elif response == "rzhd":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('РЖД-бонус', "https://vk.com/@rtuprofkom-rzhd-bonus-dlya-studentov")
        elif response == "hostel" or response == "hostel-contest":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Подробнее об общежитиях', "https://student.mirea.ru/hostel/campus/")
        elif response == "vuz" or response == "vuc":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('ВУЦ', "https://vuc.mirea.ru/")
        elif response == "expedition":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Экспедиционный корпус', "https://vuc.mirea.ru/ekspeditsionnyy-korpus/")
        elif response == "diving":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Дайвинг клуб', "https://vuc.mirea.ru/kluby/dayving/")
        elif response == "metodichka":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Методичка первокурсника', "https://student.mirea.ru/help/file/metod_perv_2022.pdf")
        elif response == "double-diploma":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Программа двойного диплома", "https://www.mirea.ru/international-activities/training-and-internships/")
        elif response == "car":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Подготовка водителей", "https://www.mirea.ru/about/the-structure-of-the-university/educational-scientific-structural-unit/driving-school-mstu-mirea/")
        elif response == "other-language":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Ссылка", "https://language.mirea.ru/")
        elif response == "business" or response == "softskill":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Ссылка на группу", "https://vk.com/ntv.mirea")
        elif response == "science":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Научные сообщества", "https://student.mirea.ru/student_scientific_society/")
            keyboard.add_openlink_button("Группа в ВК", "https://vk.com/mirea_smu")
        elif response == "constructor" or response == "accelerator":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Акселератор", "https://project.mirea.ru/")
        elif response == "uvisr":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("УВИСР", "https://student.mirea.ru/")
        elif response == "student-union":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа в ВК", "https://vk.com/sumirea")
        elif response == "media-school":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Медиашкола", "https://vk.com/mediaschool_sumirea")
        elif response == "volunteer":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Волонтёрский центр", "https://vk.com/vcrtumirea")
        elif response == "atmosfera":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Атмосфера", "https://vk.com/atmosfera")
        elif response == "apriori":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Априори", "https://vk.com/apriori.moscow")
        elif response == "counselor":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Атмосфера", "https://vk.com/atmosfera")
            keyboard.add_openlink_button("Априори", "https://vk.com/apriori.moscow")
        elif response == "rescue" or response == "rescue-contacts":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа в ВК", "https://vk.com/csovsks")
        elif response == "vector":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Вектор", "https://vk.com/vector_mirea")
        elif response == "rtuitlab":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("RTUITlab", "https://vk.com/rtuitlab")
        elif response == "group-it":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ИИТ в ВК", "https://vk.com/it_sumirea")
        elif response == "group-iii":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ИИИ в ВК", "https://vk.com/iii_sumirea")
        elif response == "group-iri":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ИРИ в ВК", "https://vk.com/iri_sumirea")
        elif response == "group-ikb":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ИКБ в ВК", "https://vk.com/ikb_sumirea")
        elif response == "group-itu":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ИТУ в ВК", "https://vk.com/itu_sumirea")
        elif response == "group-itht":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ИТХТ в ВК", "https://vk.com/itht_sumirea")
        elif response == "group-iptip":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ИПТИП в ВК", "https://vk.com/iptip__sumirea")
        elif response == "group-kpk":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа КПК в ВК", "https://vk.com/college_sumirea")
        elif response == "rating":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_button("Лайк👍", color=VkKeyboardColor.POSITIVE)
            keyboard.add_button("Дизлайк👎", color=VkKeyboardColor.NEGATIVE)
        elif response == "work":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Центр Карьеры", "https://vk.com/careercenterrtumirea")
        elif response == "graduate-union":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Ассоциация выпускников", "https://student.mirea.ru/graduate/")
        elif response == "radio":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Радиорубка и Радиолаб", "https://vk.com/rtu.radio")
        elif response == "admin":
            keyboard = VkKeyboard(one_time=True)
            keyboard.add_button("1.Вывести количество тем", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("2.Вывести все темы", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("3.Добавить тему", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("4.Удалить тему", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("5.Вывести всю информацию по теме", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("6.Добавить ответ к теме", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("7.Добавить вопрос к теме", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("8.Вывести количество пользователей", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("9.Вывести рейтинг", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("10.Выход", color=VkKeyboardColor.NEGATIVE)
        elif response == "yesno":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_button("Да", color=VkKeyboardColor.POSITIVE)
            keyboard.add_button("Нет", color=VkKeyboardColor.NEGATIVE)
        else:
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
            keyboard.add_button('Обратная связь', color=VkKeyboardColor.PRIMARY)
            keyboard.add_button('Оценить бота', color=VkKeyboardColor.PRIMARY)
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
    # corrected_text = spell.candidates(text)
    # corrected_text = TextBlob(text).correct()
    text = spell_check(text)
    print(text)
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



def add_answer(users):
    with open('intents_dataset.json', 'r', encoding='UTF-8') as f:
        data = json.load(f)
    while True:
        print("Выберите пункт меню:\n1.Вывести количество тем\n2.Вывести все темы\n3.Добавить тему\n4.Удалить тему\n5.Вывести всю информацию по теме\n6.Добавить ответ к теме\n7.Добавить вопрос к теме\n8.Вывести количество пользователей\n9.Вывести рейтинг\n10.Переобучить модель")
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
                with open('intents_dataset.json', 'w', encoding='UTF-8') as f:
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
                with open('intents_dataset.json', 'w', encoding='UTF-8') as f:
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
                with open('intents_dataset.json', 'w', encoding='UTF-8') as f:
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
        elif choice == "10":
            make_neuronetwork()
        else:
            print("Неверный пункт меню")