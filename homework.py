import math
import numpy as np


def main(y, z):
    return math.sqrt(y ** 6 - (77 * y ** 3 + z) ** 3) + \
           (z ** 6 + 64 * y ** 3) / (y ** 2 + z ** 4)


# print(main(-0.18, 0.09))
# print(main(-0.56, 0.36))
# print(main(0.17, -0.58))
# print(main(-0.74, -0.3))
# print(main(-0.96, -0.06))

def main1(y):
    if y < 30:
        return 81 * y ** 6 + y ** 7 + 92 * math.cos(y) ** 2
    elif y < 103:
        return (math.cos(y) ** 3) / 83
    elif y < 135:
        return y ** 3 / 89
    else:
        return 96 * (abs(98 - y ** 2 - 40 * y)) ** 4 - \
               36 * (math.log2(y + 66 * y ** 2 + y ** 3)) ** 3


# print(main1(141))
# print(main1(100))
# print(main1(26))
# print(main1(87))
# print(main1(123))


def main2(n, y, a):
    summ = 0
    for j in range(1, n+1):
        summ += 33 * math.cos(j) - ((31 * y ** 2 + 1) ** 5) / 61 - \
                93 * j ** 2
    for c in range(1, n+1):
        for k in range(1, a+1):
            summ -= ((0.02 + c ** 3 + c) ** 5) / 35 + \
                    45 * ((6 + c ** 3 + k) ** 3) + 1
    return summ



# print(main2(4, -0.22, 4))
# print(main2(4, -0.89, 2))
# print(main2(7, -0.27, 5))
# print(main2(3, 0.44, 3))
# print(main2(5, -0.95, 5))


def main3(n):
    if n == 0:
        return .87
    else:
        return main3(n - 1) ** 3 - main3(n - 1) / 93


# print(main3(5))
# print(main3(1))
# print(main3(6))
# print(main3(4))
# print(main3(2))


def main4(x, y):
    summ = 0
    n = len(y)
    for i in range(1, n + 1):
        summ += 90 * ((y[n - i]) ** 2 - 37 * x[n - math.ceil(i / 4)] - 1) ** 5
    return summ


# print(main4([0.46, 0.6, 0.46, -0.52, 0.2, -0.43, -0.02],
# [-0.44, 0.56, 0.01, -0.42, -0.18, -0.49, 0.35]))
# print(main4(
# [-0.13, -0.26, 0.4, 0.18, 0.86, -0.43, -0.4], [0.25, -0.19, -0.01, -0.25, 0.54, -0.38, 0.61]
# ))
# print(main4(
# [0.72, 0.06, -0.89, 0.72, 0.74, -0.51, 0.96], [-0.08, 0.98, -0.08, -0.08, -0.03, 0.07, 0.44])
# )
# print(main4(
# [-0.38, 0.41, -0.62, -0.82, -0.5, 0.55, 0.7], [-0.27, -0.18, 0.61, -0.18, 0.58, 0.58, 0.63]
# ))
# print(main4(
# [-0.37, -0.63, 0.61, -0.87, 0.57, -0.24, 0.93], [0.72, -0.5, 0.96, -0.72, -0.48, 0.68, 0.83]
# ))

import math
import tkinter as tk


def pyshader(func, w, h):
    scr = bytearray((0, 0, 0) * w * h)
    for y in range(h):
        for x in range(w):
            p = (w * y + x) * 3
            scr[p:p + 3] = [max(min(int(c * 255), 255), 0)
                            for c in func(x / w, y / h)]
    return bytes('P6\n%d %d\n255\n' % (w, h), 'ascii') + scr


def lerp(a,b,c):
    return (c * a) + ((1-c) * b)


def value_noise(x, y, size):
    xr = x * size // 1 / size  # округляем x
    distx = (1 / size) - (x - xr)
    distx = (1 - distx * size)
    distx = distx * distx * (3 - 2 * distx)
    # вычисляем значение в двух крайних точках и затем проводим между ними интерполяцию, чтобы получить размытие
    a = value_noise_y(x, y, size)
    b = value_noise_y(x - 1 / size, y, size)
    r = lerp(a, b, distx)  # интерполируем
    return r

# стабилиазция шума
def value_noise_y(x, y, size):
    yr = y * size // 1 / size # округляем y
    disty = (1 / size) - (y - yr)
    disty = (1 - disty * size)
    disty = disty * disty * (3 - 2 * disty)
    # вычисляем значение в двух крайних точках и затем проводим между ними интерполяцию, чтобы получить размытие
    a = noise(x, y, size)
    b = noise(x, y - 1 / size, size)
    r = lerp(a, b, disty)
    return r


def noise(x, y, size):
    x = x * size // 1 / size
    y = y * size // 1 / size
    n = x * 52.1 + y * 57.2
    c = math.cos(n) + math.tan(n)
    c = c * c * 4232.123  # от c ** c пришлось отказаться(
    c = c % 1
    return c


def func(x, y):
    # c = value_noise(x, y, 40)
    c = value_noise(x,y,5)
    c1 = value_noise(x, y, 10)
    c2 = value_noise(x, y, 15)
    c = lerp(c,c1,c2)
    c += value_noise(x, y, 20)*0.15
    c += value_noise(x, y, 40) * 0.15
    return c,c,1



label = tk.Label()
img = tk.PhotoImage(data=pyshader(func, 256, 256)).zoom(2, 2)
label.pack()
label.config(image=img)
tk.mainloop()

