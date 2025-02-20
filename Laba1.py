import numpy as np
from PIL import Image, ImageOps

import math as math


def draw_line_obvious(image, x0, y0, x1, y1, count, color):
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            image[y, x] = color


def draw_line_obvious_with_fix(image, x0, y0, x1, y1, color):
    count = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color


def draw_line_x(image, x0, y0, x1, y1, color):
    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range(int(x0), int(x1)):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color


def draw_line_Bresenham(image, x0, y0, x1, y1, color):
    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = int(y0)
    dy = 2 * abs(y1 - y0)
    derror = 0.0

    y_update = 1 if y1 > y0 else -1

    for x in range(int(x0), int(x1)):

        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color

        derror += dy
        if (derror > (x1 - x0)):
            derror -= 2 * (x1 - x0)
            y += y_update


def Task3():
    img_mat = np.zeros((200, 200, 3), dtype=np.uint8)

    for i in range(13):
        x0 = 100
        y0 = 100
        x1 = 100 + 95 * math.cos(i * 2 * math.pi / 13)
        y1 = 100 + 95 * math.sin(i * 2 * math.pi / 13)

        draw_line_obvious(img_mat, x0, y0, x1, y1, 20, 255)
        draw_line_obvious_with_fix(img_mat, x0, y0, x1, y1, 255)
        draw_line_x(img_mat, x0, y0, x1, y1, 255)
        draw_line_Bresenham(img_mat, x0, y0, x1, y1, 255)

    img = Image.fromarray(img_mat, mode='RGB')
    img.save('frame.png')


def Task4():
    img_mat = np.zeros((4000, 4000, 3), dtype=np.uint8)

    model = open('model_1.obj')
    list_v, list_f = [], []

    for line in model:
        s = line.split()
        if s[0] == 'v':
            list_v.append((float(s[1]), float(s[2]), float(s[3])))
        else:
            if s[0] == 'f':
                list_f.append((int(s[1].split('/')[0]), int(s[2].split('/')[0]), int(s[3].split('/')[0])))

    # for parts in list_v:
    #     x, y = int(5000 * float(parts[0]) + 500), int(5000 * float(parts[1]) + 200)
    #     img_mat[y, x] = 255

    for parts in list_f:
        x0, y0 = float(list_v[parts[0] - 1][0]) * 20000 + 2000, float(list_v[parts[0] - 1][1]) * 20000 + 800
        x1, y1 = float(list_v[parts[1] - 1][0]) * 20000 + 2000, float(list_v[parts[1] - 1][1]) * 20000 + 800
        x2, y2 = float(list_v[parts[2] - 1][0]) * 20000 + 2000, float(list_v[parts[2] - 1][1]) * 20000 + 800

        draw_line_Bresenham(img_mat, x0, y0, x1, y1, 255)

        draw_line_Bresenham(img_mat, x1, y1, x2, y2, 255)

        draw_line_Bresenham(img_mat, x2, y2, x0, y0, 255)

    img = Image.fromarray(img_mat, mode='RGB')
    img = ImageOps.flip(img)
    img.save('frame.png')


Task4()
