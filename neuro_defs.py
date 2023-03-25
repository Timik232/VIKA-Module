import os
import pickle
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from vk_api.keyboard import VkKeyboard, VkKeyboardColor
from vk_api.utils import get_random_id
import vk_api
from private_api import token_api  # токен который не должен быть у всех, поэтому вынес в отдельный файл.
from private_api import service_token
import nltk
import requests
import re
import random
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from spellchecker import SpellChecker
import tokenization
from transformers import AutoModel, AutoTokenizer
from transformers import BertTokenizer, BertModel
import torch
from transformers import logging
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DataCollatorWithPadding
import numpy as np
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import pipeline
from huggingface_hub import notebook_login
from sentence_transformers import losses
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from datasets import load_dataset

# notebook_login()

logging.set_verbosity_error()

dictionary = SpellChecker(language='ru', distance=1)

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


def learn_spell(data):
    words = set()
    for name in data:
        for question in data[name]['examples']:
            question = clean_up(question)
            for word in question.split():
                words.add(word)
    dictionary.word_frequency.load_words(words)
    with open(f'{os.path.dirname(os.getcwd())}\\VIKA_pickle\\dictionary.pickle', 'wb') as f:
        pickle.dump(dictionary, f)
    print("Словарь обучен")


# def fine_tuning():
#     model_name = 'sentence-transformers/LaBSE'
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name)
#
#     # Получаем словарь из токенайзера
#     vocab = tokenizer.get_vocab()
#
#     # Добавляем новое слово
#     new_word = 'вуц'
#     vocab[new_word] = len(vocab)
#     from transformers import AutoTokenizer, AutoModel
#     import torch
#     from torch.utils.data import TensorDataset, DataLoader
#     from sentence_transformers import SentenceTransformer, InputExample
#     # Дообучаем модель на примерах
#     sentences = ['это предложение с новым словом вуц', 'это другое предложение без новых слов']
#     examples = [InputExample(texts=[s], label=0) for s in sentences]
#
#     train_data = DataLoader(TensorDataset(torch.arange(len(examples))), batch_size=2)
#     model.train()
#     for epoch in range(3):
#         for batch in train_data:
#             model.zero_grad()
#             batch = tuple(t.to('cuda') for t in batch)
#             inputs = tokenizer([examples[i].texts[0] for i in batch], padding=True, truncation=True, return_tensors='pt')
#             inputs = {k: v.to('cuda') for k, v in inputs.items()}
#             outputs = model(**inputs)[1]
#             loss = torch.mean(outputs[:, 0])
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

def cosine_sim(query, vectorizer):
    if os.path.isfile(f'{os.path.dirname(os.getcwd())}\\VIKA_pickle\\base.pkl'):
        with open(f'{os.path.dirname(os.getcwd())}\\VIKA_pickle\\base.pkl', "rb") as f:
            base = pickle.load(f)
            print("cosine base loaded")
    else:
        with open('jsons\\intents_dataset.json', 'r', encoding='UTF-8') as f:
            data = json.load(f)
        x = []
        y = []
        for name in data:
            for question in data[name]['examples']:
                x.append(vectorizer.encode([question]))
                y.append(name)
        base = [x,y]
        with open(f'{os.path.dirname(os.getcwd())}\\VIKA_pickle\\base.pkl', "wb") as f:
            pickle.dump(base, f)
    elems = []
    #maximum = cosine_similarity(vectorizer.encode([query]), base[0][0])
    for i in range(1, len(base[0])):
        cos = cosine_similarity(vectorizer.encode([query]), base[0][i])
        elems.append(cos)
    # print(max(elems))
    return base[1][elems.index(max(elems))]


def transformer_classification(data):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    imdb = {"test": [
    ],
    "train":[
    ]
    }
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    # label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    id2label = {}
    label2id = {}
    count = 0
    for name in data:
        for question in data[name]['examples']:
            buf = {
                "label": count,
                "text": question
                   }
            imdb["train"].append(buf)
            id2label[count] = name
            label2id[name] = count
        for phrase in data[name]['responses']:
            buf = {
                "label": count,
                "text": phrase
                   }
        count += 1

    from datasets import load_dataset

    imdb = load_dataset("imdb")
    print(imdb)
    tokenized_imdb = imdb.map(preprocess_function, batched=True)
    print(tokenized_imdb)
    #tokenized_imdb = preprocess_function(imdb)
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilabert-base-uncased", num_labels=1324, id2label=id2label, label2id=label2id
    )
    # нужно как-то залогиниться, тогда может заработает, пока нет
    training_args = TrainingArguments(
        output_dir="my_awesome_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_imdb["train"],
        eval_dataset=tokenized_imdb["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    text = "Кудж"
    tokenizer = AutoTokenizer.from_pretrained(trainer)
    inputs = tokenizer(text, return_tensors="pt")
    model = AutoModelForSequenceClassification.from_pretrained("stevhliu/my_awesome_model")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    print(model.config.id2label[predicted_class_id])
    classifier = pipeline("sentiment-analysis", model=trainer)
    print(classifier(text))
    # trainer.push_to_hub()
    return trainer


def fine_tuning(data, vectorizer, dictionary, model_mlp):
    train_loss = losses.MultipleNegativesRankingLoss(model=vectorizer)
    n_examples = [
    ]
    count = 0
    for name in data:
        buf = []
        for question in data[name]['examples']:
            buf.append(question)
        n_examples.append(buf)

    train_examples = []
    for i in range(len(n_examples)):
        example = n_examples[i]
        print(example)
        #print(n_examples)
        if example != "" and example != [] and len(example) >= 3:
            train_examples.append(InputExample(texts=[example[0], example[1], example[2]]))
    print(train_examples)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    print(type(train_dataloader))
    vectorizer.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10)
    print(get_intent_bert("кудж", model_mlp, vectorizer, dictionary))
    with open(f'{os.path.dirname(os.getcwd())}\\VIKA_pickle\\vector.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)


def make_bertnetwork():
    with open('jsons\\intents_dataset.json', 'r', encoding='UTF-8') as f:
        data = json.load(f)
    x = []
    y = []
    for name in data:
        for question in data[name]['examples']:
            x.append(clean_up(question))
            y.append(name)
        for phrase in data[name]['responses']:
            x.append(phrase)
            y.append(name)

    device = torch.device("cuda")
    vectorizer = SentenceTransformer('distiluse-base-multilingual-cased')
    vectorizer.to(device)
    x_vec = vectorizer.encode(x)
    model_mlp = MLPClassifier(hidden_layer_sizes=322, activation='relu', solver='adam', learning_rate='adaptive',
                              max_iter=1500)
    model_mlp.fit(x_vec, y)
    y_pred = model_mlp.predict(x_vec)
    print("точность " + str(accuracy_score(y, y_pred)))
    print("f1 " + str(f1_score(y, y_pred, average='macro')))
    with open(f'{os.path.dirname(os.getcwd())}\\VIKA_pickle\\model.pkl', 'wb') as f:
        pickle.dump(model_mlp, f)
    with open(f'{os.path.dirname(os.getcwd())}\\VIKA_pickle\\vector.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    neuro = [model_mlp, vectorizer]
    print("Обучено")
    return neuro


def make_neuronetwork():
    with open('jsons\\intents_dataset.json', 'r', encoding='UTF-8') as f:
        data = json.load(f)
    x = []
    y = []
    for name in data:
        for question in data[name]['examples']:
            x.append(question)
            y.append(name)
        for phrase in data[name]['responses']:
            x.append(phrase)
            y.append(name)

    # векторизируем файлы и обучаем модель
    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(x)
    model_mlp = MLPClassifier(hidden_layer_sizes=322, activation='relu', solver='adam', learning_rate='adaptive',
                              max_iter=1500)
    model_mlp.fit(X_vec, y)
    y_pred = model_mlp.predict(X_vec)
    print("точность " + str(accuracy_score(y, y_pred)))
    print("f1 " + str(f1_score(y, y_pred, average='macro')))
    with open(f'{os.path.dirname(os.getcwd())}\\VIKA_pickle\\model.pkl', 'wb') as f:
        pickle.dump(model_mlp, f)
    with open(f'{os.path.dirname(os.getcwd())}\\VIKA_pickle\\vector.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    neuro = [model_mlp, vectorizer]
    print("Обучено")
    return neuro


def create_keyboard(id, text, response="start"):
    try:
        keyboard = VkKeyboard(one_time=True)
        if response == "not_that" or response == "help_me" or response == "rss" or response == "callhuman" or response == "flood":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Ссылка на ВК', "https://vk.com/bramind002")
        elif response == "grifon":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Стикеры с грифоном в ТГ', "https://t.me/addstickers/rtumirea")
        elif response == "psychology" or response == "danger" or response == "feedback_bad" or response == "motivation" or response == "psycho" or response == "psychologist":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Психологическая служба',
                                         "https://student.mirea.ru/psychological_service/staff/")
        elif response == "map" or response == "location":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Навигатор', "https://ischemes.ru/group/rtu-mirea/vern78")
        elif response == "rules":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Устав', "https://www.mirea.ru/upload/medialibrary/d0e/Ustav-Novyy.pdf")
            keyboard.add_openlink_button('Правила внутреннего распорядка', "https://www.mirea.ru/docs/125641/")
            keyboard.add_line()
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
        elif response == "military" or response == "для-военкомата":
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
        elif response == "metodichka" or response == "maps":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Методичка первокурсника',
                                         "https://student.mirea.ru/help/file/metod_perv_2022.pdf")
        elif response == "double-diploma":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Программа двойного диплома",
                                         "https://www.mirea.ru/international-activities/training-and-internships/")
        elif response == "car":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Подготовка водителей",
                                         "https://www.mirea.ru/about/the-structure-of-the-university/educational-scientific-structural-unit/driving-school-mstu-mirea/")
        elif response == "other-language":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Ссылка", "https://language.mirea.ru/")
        elif response == "business" or response == "softskill":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Ссылка на группу", "https://vk.com/ntv.mirea")
        elif response == "забыл-вещи":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Бюро находок", "https://vk.com/public79544978")
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
            # keyboard.add_button("1.Вывести количество тем", color=VkKeyboardColor.PRIMARY)
            # keyboard.add_line()
            keyboard.add_button("1.Вывести все темы", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("2.Добавить тему", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("3.Удалить тему", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("4.Вывести всю информацию по теме", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("5.Добавить ответ к теме", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("6.Добавить вопрос к теме", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("7.Найти тему по вопросу", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("8.Статистика и рейтинг", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("9.Выход", color=VkKeyboardColor.NEGATIVE)
        elif response == "statistic":
            keyboard = VkKeyboard(one_time=True)
            keyboard.add_button("1.Вывести количество тем", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("2.Рейтинг бота", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("3.Количество вопросов", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("4.Вывести количество пользователей", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("5.Вернуться", color=VkKeyboardColor.NEGATIVE)
        elif response == "yesno":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_button("Да", color=VkKeyboardColor.POSITIVE)
            keyboard.add_button("Нет", color=VkKeyboardColor.NEGATIVE)
        elif response == "uch-otd" or response == "учебный-отдел-ит":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Учебный отдел",
                                         "https://www.mirea.ru/education/the-institutes-and-faculties/institute-of-information-technology/contacts/")
        elif response == "профсоюз":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Профсоюз", "https://vk.com/rtuprofkom")
        elif response == "center_culture":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Центр культуры", "https://student.mirea.ru/center_culture/creativity/")
            keyboard.add_openlink_button("Группа в ВК ЦКТ", "https://vk.com/cktmirea")
        elif response == "ври-лес" or response == "школа-леса":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("ВРИ Лес", "https://vk.com/vri_les")
        elif response == "швизис":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("ШВИЗИС", "https://vk.com/shvizis")
        elif response == "заявление-в-студ":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Вступить в СтудСоюз", "https://sumirea.ru/connect/")
        elif response == "отдел-по-работе-общежитие":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Работа с общежитиями", "https://student.mirea.ru/about/section1/")
        elif response == "справочник":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Справочник", "https://tel.mirea.ru/")
        elif response == "кафедра-общей-информатики":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Подробнее",
                                         "https://www.mirea.ru/education/the-institutes-and-faculties/institut-iskusstvennogo-intellekta/the-structure-of-the-institute/chair-of-general-informatics/")
        elif response == "вт" or response == "платонова":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ВК", "https://vk.com/kvt_mirea")
            keyboard.add_openlink_button("Подробнее",
                                         "https://www.mirea.ru/education/the-institutes-and-faculties/institute-of-information-technology/the-structure-of-the-institute/department-of-computer-engineering/")
        elif response == "мосит" or response == "головин":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ВК", "https://vk.com/mireamosit")
            keyboard.add_openlink_button("Подробнее",
                                         "https://www.mirea.ru/education/the-institutes-and-faculties/institute-of-information-technology/the-structure-of-the-institute/department-of-mathematical-provision-and-standardization-of-information-technology/")
        elif response == "иппо" or response == "болбаков":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ВК", "https://vk.com/ippo_it")
            keyboard.add_openlink_button("Подробнее",
                                         "https://www.mirea.ru/education/the-institutes-and-faculties/institute-of-information-technology/the-structure-of-the-institute/department-of-instrumental-and-applied-software/")
        elif response == "ппи" or response == "ппи-зав":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ВК", "https://vk.com/ppi_it")
            keyboard.add_openlink_button("Подробнее",
                                         "https://www.mirea.ru/education/the-institutes-and-faculties/institute-of-information-technology/the-structure-of-the-institute/the-department-of-practical-and-applied-computer-science/")
        elif response == "кафедра-пм" or response == "дзержинский":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ВК", "https://vk.com/kafprimat")
            keyboard.add_openlink_button("Подробнее",
                                         "https://www.mirea.ru/education/the-institutes-and-faculties/institute-of-information-technology/the-structure-of-the-institute/the-department-of-applied-mathematics/")
        elif response == "кис" or response == "адрианова":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ВК", "https://vk.com/kis_it_mirea")
            keyboard.add_openlink_button("Подробнее",
                                         "https://www.mirea.ru/education/the-institutes-and-faculties/institute-of-information-technology/the-structure-of-the-institute/the-department-of-corporate-information-systems/")
        elif response == "программа-обмена":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Отд.Международного сотрудничества",
                                         "https://www.mirea.ru/about/the-structure-of-the-university/administrative-structural-unit/the-department-of-international-relations/the-department-of-international-cooperation/")
        elif response == "положение-элитной" or response == "отчисление-с-элитной":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Положение Элитной Подготовки",
                                         "https://www.mirea.ru/upload/iblock/555/scb628vl1c1v3ah22653z0grta7pz3fd/pr_1179_10_09_2020_Polozhenie-po-EP.pdf")
        elif response == "стипендия-по-приоритетным-направлениям":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Перечень", "https://base.garant.ru/70842752/#block_3")
        elif response == "страйкбол":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Страйкбольный клуб", "https://vk.com/rtuairsoftvuc")
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


def tokenize(text):
    vocab_path = 'bert/vocab.txt'
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=True)
    return tokenizer.tokenize(text)


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


def get_intent(text, model_mlp, vectorizer, dictionary):
    corrected_text = ""
    for word in text.split():
        word = str(dictionary.correction(word))
        corrected_text += word + ' '
    # corrected_text = dictionary.correction(text)
    text_vec = vectorizer.transform([corrected_text])
    return model_mlp.predict(text_vec)[0]


def get_intent_bert(text, model_mlp, vectorizer, dictionary):
    corrected_text = ""
    for word in text.split():
        word = str(dictionary.correction(word))
        corrected_text += word + ' '
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
            "Выберите пункт меню:\n1.Вывести количество тем\n2.Вывести все темы\n3.Добавить тему\n4.Удалить тему\n5.Вывести всю информацию по теме\n6.Добавить ответ к теме\n7.Добавить вопрос к теме\n8.Вывести количество пользователей\n9.Вывести рейтинг\n10.Переобучить модель")
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
        elif choice == "10":
            # make_neuronetwork()
            make_bertnetwork()
        else:
            print("Неверный пункт меню")


def parsing():
    question = []
    answer = []
    dict = {}
    offset = 0
    all_posts = []
    while offset < 1000:
        vk_page = requests.get("https://api.vk.com/method/wall.get",
                               params={
                                   'access_token': service_token,
                                   'v': 5.131,
                                   'domain': 'ask_mirea',
                                   'count': 100,
                                   'offset': offset
                               })
        # vk_page = requests.get('https://vk.com/ask_mirea')
        try:
            page = vk_page.json()['response']['items']
            all_posts.extend(page)
        except BaseException as ex:
            print(ex, vk_page)
        offset += 100
    for i in all_posts:
        try:
            msg = i['text']
            if msg.find("Вопрос:") == -1 or msg.find('Ответ:') == -1:
                continue
            question.append(msg[msg.find("Вопрос:"):msg.find("Ответ:"):].strip())
            answer.append(msg[msg.find("Ответ:"):].strip())
        except BaseException as ex:
            print(ex)
            print(i['text'])
    # print(question)
    # print(answer)
    for i in range(len(question)):
        # print(question[i], answer[i])
        question[i] = question[i].replace("Вопрос:", "", 1)
        question[i] = clean_up(
            question[i].replace("Здравствуйте", "", 1).replace("спасибо", "").replace("\n", " ").strip())
        answer[i] = answer[i].replace("Ответ:", "", 1)
        answer[i] = answer[i].replace("Здравствуйте!", "", 1).replace("Здравствуйте,", "", 1).replace("Здравствуйте.",
                                                                                                      "", 1).strip()

        tempq = [question[i]]
        tempa = [answer[i]]
        dict[f"topic{i}"] = {
            "examples": tempq,
            "responses": tempa
        }
    with open("second_dict.json", 'w', encoding='UTF-8') as f:
        json.dump(dict, f, ensure_ascii=False, indent=4)
    # print(dict)
    print("json ready")
