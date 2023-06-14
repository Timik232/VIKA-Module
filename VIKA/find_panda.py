from skimage import io, color  # Для импотра и экспорта изображений
import torch
import torch.nn as nn  # Модуль PyTorch для слоёв нейронных сетей
from torchvision import transforms  # Модуль PyTorch для предобработки изображений
from neuro_defs import send_message
import numpy as np
from neuro_defs import slh, cwd, CPU_Unpickler


class Alexnet(nn.Module):
    def __init__(self):
        super(Alexnet, self).__init__()

        # Свёрточные слои нейронной сети
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=6)

        # Действия между слоями в нейронной сети
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)  # Макспулинг 3х3 с шагом 2
        self.relu = nn.ReLU()  # функция активации ReLU
        # Этот слой помогает нам избежать вычисления размера выходной карты признаков при загрузке в линейный слой в PyTorch
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(6, 6))

        # Реализация из библиотеки PyTorch
        self.norm = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=1)
        # Наша реализация
        # self.norm = LocalResponseNormalization(neighbourhood_length=5, normalisation_const_alpha=1e-4, contrast_const_beta=0.75, noise_k=1.0)

        self.dropout = nn.Dropout()  # Слой регуляризации перед выходом на полносвязный слой нейронной сети

    # Алгоритм прямого распространения информации
    def forward(self, x):
        # Первый свёрточный слой
        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.max_pool(x)

        # Второй свёрточный слой
        x = self.conv2(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.max_pool(x)

        # Две последовательные свёртки 3х3 без пулинга
        # третий свёрточный слой
        x = self.conv3(x)
        x = self.relu(x)
        # четвёртый свёрточный слой
        x = self.conv4(x)
        x = self.relu(x)

        # Пятый слой с adatpive pool
        x = self.conv5(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.adaptive_pool(x)

        # Сплющивание данных перед подачей в полносвязный выходной слой
        x = torch.flatten(x, 1)

        # Полносвязная нейронная сеть на выходе
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


transform_img = transforms.Compose([
        transforms.ToPILImage(),  # Python Imaging Library
        transforms.Resize((224, 224)),  # Принудительное сжатие и интерполяция
        transforms.ToTensor()  # Трансформация к тензору исходных данных
    ])
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"


def get_alexnet():
    #alexnet = torch.load(f'{cwd()}{slh()}VIKA-pickle{slh()}neuro.pt', map_location=torch.device("cpu"))  # Загрузка модели
    with open(f'{cwd()}{slh()}VIKA-pickle{slh()}neuro.pt', "rb") as f:
        #alexnet = CPU_Unpickler(f).load()
        alexnet = torch.load(f, map_location=torch.device('cpu'))
    alexnet.eval()  # Отключение режима работы с модулями PyTorch
    return alexnet 


def show_prediction(id, alexnet, img_path, temp_dir):
    idx2label = {0: "не панду",
                 1: "панду"
                 }
    img = io.imread(img_path)

    # Проверка числа каналов изображения
    if len(img.shape) == 2:
        # Если изображение имеет только один канал, повторите его три раза
        img = img[:, :, None]  # Добавление нового измерения для канала
        img = np.repeat(img, 3, axis=2)  # Повторение канала три раза

    try:
        transformed_img = transform_img(img)
        out = alexnet(transformed_img.unsqueeze(0).to(device=device))
        _, pred = out.max(1)
        send_message(id, "Я думаю, что изображение содержит {}".format(idx2label[pred.item()]))
    except RuntimeError as e:
        send_message(id, "Ошибка при обработке изображения: {}".format(str(e)))
    temp_dir.cleanup()
