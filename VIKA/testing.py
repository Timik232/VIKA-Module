import os
import pickle
import torch
import json
import pandas as pd
import openpyxl
from sentence_transformers import SentenceTransformer
from neuro_defs import cwd, answering
from learning_functions import fine_tuning, learn_spell, make_bertnetwork

if __name__ == "__main__":
    if not os.path.exists(os.path.join(cwd(), 'VIKA-pickle')):
        os.mkdir(os.path.join(cwd(), 'VIKA-pickle'))
    # загрузка данных для обучения
    with open(os.path.join('jsons', 'intents_dataset.json'), 'r', encoding='UTF-8') as f:
        data = json.load(f)
        print("Data loaded")
    # обучение и загрузка словаря
    if os.path.isfile(os.path.join(cwd(), 'VIKA-pickle', 'dictionary.pickle')):
        with open(os.path.join(cwd(), 'VIKA-pickle', 'dictionary.pickle'), 'rb') as f:
            dictionary = pickle.load(f)
        print("Dictionary loaded")
    else:
        dictionary = learn_spell(data)
        print("Dictionary loaded")
    # обучение и загрузка модели
    if not os.path.isfile(os.path.join(cwd(), 'VIKA-pickle', 'model.pkl')):
        print("Neural network is learning")
        neuro = make_bertnetwork()
        model_mlp = neuro[0]
        vectorizer = neuro[1]
        fine_tuning(data, vectorizer)
    else:
        with open(os.path.join(cwd(), 'VIKA-pickle', 'model.pkl'), 'rb') as f:
            model_mlp = pickle.load(f)
        vectorizer = SentenceTransformer('distiluse-base-multilingual-cased')
        vectorizer.load_state_dict(torch.load(os.path.join(cwd(), 'VIKA-pickle', 'vector.pt'), map_location='cpu'))
        print("Model loaded")
    # вспомогательные объекты: аббревиатуры, конкретные аудитории, которые может распознавать ВИКА
    with open(os.path.join('jsons', 'objects.json'), 'r', encoding='UTF-8') as f:
        objects = json.load(f)

    with open("test.txt", 'r', encoding='UTF-8') as f:
        questions = f.readlines()
        print("Questions loaded")

    # print(questions)

    answers = []
    for question in questions:
        if question.strip() == '':
            continue
        answer = answering(question, model_mlp, data, vectorizer, dictionary, objects)
        answers.append(answer[0])

    with open("answers.txt", 'w', encoding="UTF-8") as f:
        f.writelines(answers)

    answer_dict = {
        "Answers": answers
    }
    df = pd.DataFrame(answer_dict)
    df.to_excel('answers.xlsx', index=False)

