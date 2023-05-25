from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sentence_transformers import losses
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DataCollatorWithPadding
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from spellchecker import SpellChecker
import requests
import torch
import pickle
import os
import json
import evaluate
import numpy as np
from neuro_defs import clean_up
#from private_api import service_token


dictionary = SpellChecker(language='ru', distance=1)


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
    #print(get_intent_bert("кудж", model_mlp, vectorizer, dictionary))
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
    #vectorizer = SentenceTransformer("all-mpnet-base-v2")
    vectorizer.to(device)
    x_vec = vectorizer.encode(x)
    model_mlp = MLPClassifier(hidden_layer_sizes=222, activation='relu', solver='adam', learning_rate='adaptive',
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


def parsing(number):
    question = []
    answer = []
    dict = {}
    offset = 0
    all_posts = []
    while offset < number:
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
