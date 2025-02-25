import numpy as np
from PIL import Image, ImageOps

import math
import random


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


def bicentric_coor(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1

    return lambda0, lambda1, lambda2


def normal_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    n = np.array([((y1-y2)*(z1-z0) - (z1-z2)*(y1-y0)), ((x1-x2)*(z1-z0) - (z1-z2)*(x1-x0)), ((x1-x2)*(y1-y0) - (y1-y2)*(x1-x0))])
    n = n / np.linalg.norm(n)
    return n


def draw_triangle(image, z_buffer, x0, y0, z0, x1, y1, z1, x2, y2, z2, color):
    normal = normal_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2)

    if normal[2] < 0:

        p_x0, p_y0, p_x1, p_y1, p_x2, p_y2 = set_zoom_for_drawing(x0, y0, x1, y1, x2, y2)

        p_x_min = int(round(min(p_x0, p_x1, p_x2))) if min(p_x0, p_x1, p_x2) > 0 else 0
        p_x_max = int(round(max(p_x0, p_x1, p_x2))) if max(p_x0, p_x1, p_x2) < 4000 else 4000
        p_y_min = int(round(min(p_y0, p_y1, p_y2))) if min(p_y0, p_y1, p_y2) > 0 else 0
        p_y_max = int(round(max(p_y0, p_y1, p_y2))) if max(p_y0, p_y1, p_y2) < 4000 else 4000

        for x in range(p_x_min, p_x_max):
            for y in range(p_y_min, p_y_max):
                lambda0, lambda1, lambda2 = bicentric_coor(x, y, p_x0, p_y0, p_x1, p_y1, p_x2, p_y2)
                if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                    z = lambda0*z0 + lambda1*z1 + lambda2*z2
                    if z <= z_buffer[y][x]:
                        z_buffer[y][x] = z
                        image[y, x] = (-255 * normal[2], 0, 0)


def set_zoom_for_drawing(x0, y0, x1, y1, x2, y2):
    x0, y0 = x0 * 20000 + 2000, y0 * 20000 + 800
    x1, y1 = x1 * 20000 + 2000, y1 * 20000 + 800
    x2, y2 = x2 * 20000 + 2000, y2 * 20000 + 800
    return x0, y0, x1, y1, x2, y2


def main():
    img_mat = np.zeros((4000, 4000, 3), dtype=np.uint8)
    z_buffer = np.full((4000, 4000), 10000, dtype=np.float32)

    # Task 9
    # draw_triangle(img_mat, 20, 0, 1, 0, 0, 40)

    model = open('model_1.obj')
    list_v, list_f = [], []

    for line in model:
        s = line.split()
        if s[0] == 'v':
            list_v.append((float(s[1]), float(s[2]), float(s[3])))
        else:
            if s[0] == 'f':
                list_f.append((int(s[1].split('/')[0]), int(s[2].split('/')[0]), int(s[3].split('/')[0])))

    for parts in list_f:
        x0, y0, z0 = float(list_v[parts[0] - 1][0]), float(list_v[parts[0] - 1][1]), float(list_v[parts[0] - 1][2])
        x1, y1, z1 = float(list_v[parts[1] - 1][0]), float(list_v[parts[1] - 1][1]), float(list_v[parts[1] - 1][2])
        x2, y2, z2 = float(list_v[parts[2] - 1][0]), float(list_v[parts[2] - 1][1]), float(list_v[parts[2] - 1][2])

        draw_triangle(img_mat, z_buffer, x0, y0, z0, x1, y1, z1, x2, y2, z2, random.randrange(256))

    img = Image.fromarray(img_mat, mode='RGB')
    img = ImageOps.flip(img)
    img.save('frame.png')


main()
