import json

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier


def neuro(layers,activation,solver, learnin_rate, max_iter):
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
    model_mlp = MLPClassifier(hidden_layer_sizes=layers, activation=activation, solver=solver, learning_rate=learnin_rate,
                              max_iter=max_iter)
    model_mlp.fit(X_vec, y)
    y_pred = model_mlp.predict(X_vec)
    print("точность " + str(accuracy_score(y, y_pred)))
    print("f1 " + str(f1_score(y, y_pred, average='macro')))

    # тестируем модель
neuro(322, 'relu', 'adam', 'adaptive', 1500)
neuro(322, 'relu', 'adam', 'adaptive', 5000)
neuro(322, 'relu', 'adam', 'adaptive', 15000)
print()
neuro(1000, 'relu', 'adam', 'adaptive', 1500)
neuro(1000, 'relu', 'adam', 'adaptive', 5000)
neuro(5000, 'relu', 'adam', 'adaptive', 15000)
print()
neuro(322, 'relu', 'lbfgs', 'adaptive', 1500)
neuro(322, 'relu', 'sgd', 'adaptive', 1500)
neuro(322, 'relu', 'adam', 'adaptive', 1500)
print()
neuro(322, 'logistic', 'adam', 'adaptive', 1500)
neuro(322, 'tanh', 'adam', 'adaptive', 1500)
neuro(322, 'identity', 'adam', 'adaptive', 1500)
neuro(322, 'relu', 'adam', 'adaptive', 1500)
print()
neuro(322, 'relu', 'adam', 'constant', 1500)
neuro(322, 'relu', 'adam', 'invscaling', 1500)
neuro(322, 'relu', 'adam', 'adaptive', 1500)
print()
neuro(100, 'relu', 'adam', 'adaptive', 1500)
neuro(5, 'relu', 'adam', 'adaptive', 1500)
neuro(322, 'relu', 'adam', 'adaptive', 500)



