import numpy as np
from PIL import Image, ImageOps


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


def get_bicentric_coor(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1

    return lambda0, lambda1, lambda2


def get_normal_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    n = np.array([((y1 - y2) * (z1 - z0) - (z1 - z2) * (y1 - y0)), ((x1 - x2) * (z1 - z0) - (z1 - z2) * (x1 - x0)),
                  ((x1 - x2) * (y1 - y0) - (y1 - y2) * (x1 - x0))])
    n = n / np.linalg.norm(n)
    return n


def get_min_and_max_XY(p_x0, p_y0, p_x1, p_y1, p_x2, p_y2):
    p_x_min = int(round(min(p_x0, p_x1, p_x2))) if min(p_x0, p_x1, p_x2) > 0 else 0
    p_x_max = int(round(max(p_x0, p_x1, p_x2))) if max(p_x0, p_x1, p_x2) < 4000 else 4000
    p_y_min = int(round(min(p_y0, p_y1, p_y2))) if min(p_y0, p_y1, p_y2) > 0 else 0
    p_y_max = int(round(max(p_y0, p_y1, p_y2))) if max(p_y0, p_y1, p_y2) < 4000 else 4000

    return p_x_min, p_x_max, p_y_min, p_y_max


def check_z_and_change_z_buffer_and_draw_pixel(image, z_buffer, z, x, y, color):
    if z <= z_buffer[y][x]:
        z_buffer[y][x] = z
        image[y, x] = color


def draw_triangle(image, z_buffer, x0, y0, z0, x1, y1, z1, x2, y2, z2):
    normal = get_normal_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    cosine_of_the_light = normal[2]

    if cosine_of_the_light < 0:

        p_x0, p_y0, p_x1, p_y1, p_x2, p_y2 = set_zoom_for_drawing(x0, y0, z0, x1, y1, z1, x2, y2, z2)
        p_x_min, p_x_max, p_y_min, p_y_max = get_min_and_max_XY(p_x0, p_y0, p_x1, p_y1, p_x2, p_y2)

        for x in range(p_x_min, p_x_max):
            for y in range(p_y_min, p_y_max):
                lambda0, lambda1, lambda2 = get_bicentric_coor(x, y, p_x0, p_y0, p_x1, p_y1, p_x2, p_y2)
                if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                    z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                    check_z_and_change_z_buffer_and_draw_pixel(image, z_buffer, z, x, y,
                                                               (-255 * cosine_of_the_light, 0, 0))


def set_zoom_for_drawing(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    zoom = 7000
    x0, y0 = (x0 * zoom) / z0 + 2000, (y0 * zoom) / z0 + 2000
    x1, y1 = (x1 * zoom) / z1 + 2000, (y1 * zoom) / z1 + 2000
    x2, y2 = (x2 * zoom) / z2 + 2000, (y2 * zoom) / z2 + 2000

    return x0, y0, x1, y1, x2, y2


def get_rotation_matrix():
    a, b, c = 31, 29.9, 31
    R_x = np.array([(1, 0, 0), (0, np.cos(a), np.sin(a)), (0, -np.sin(a), np.cos(a))])
    R_y = np.array([(np.cos(b), 0, np.sin(b)), (0, 1, 0), (-np.sin(b), 0, np.cos(b))])
    R_z = np.array([(np.cos(c), np.sin(c), 0), (-np.sin(c), np.cos(c), 0), (0, 0, 1)])

    return R_x, R_y, R_z


def rotate_vertex(vertex, shift):
    R_x, R_y, R_z = get_rotation_matrix()

    vertex[...] = np.dot(R_z, vertex)
    vertex[...] = np.dot(R_y, vertex)
    vertex[...] = np.dot(R_x, vertex)

    vertex[...] = vertex + np.array(shift)


def get_coor(list_v, vertex0, vertex1, vertex2):
    x0, y0, z0 = float(list_v[vertex0][0]), float(list_v[vertex0][1]), float(list_v[vertex0][2])
    x1, y1, z1 = float(list_v[vertex1][0]), float(list_v[vertex1][1]), float(list_v[vertex1][2])
    x2, y2, z2 = float(list_v[vertex2][0]), float(list_v[vertex2][1]), float(list_v[vertex2][2])

    return x0, y0, z0, x1, y1, z1, x2, y2, z2


def fill_in_the_lists(model_name, list_v, list_f):
    model = open(model_name)

    for line in model:
        s = line.split()
        if s[0] == 'v':
            list_v.append(np.array([float(s[1]), float(s[2]), float(s[3])]))

            rotate_vertex(list_v[-1], [0, -0.05, 0.24])

        else:
            if s[0] == 'f':
                list_f.append((int(s[1].split('/')[0]), int(s[2].split('/')[0]), int(s[3].split('/')[0])))


def draw_model(img_mat, z_buffer, list_v, list_f):
    for parts in list_f:
        x0, y0, z0, x1, y1, z1, x2, y2, z2 = get_coor(list_v, parts[0] - 1, parts[1] - 1, parts[2] - 1)

        draw_triangle(img_mat, z_buffer, x0, y0, z0, x1, y1, z1, x2, y2, z2)


def main():
    img_mat = np.zeros((4000, 4000, 3), dtype=np.uint8)
    z_buffer = np.full((4000, 4000), 10000, dtype=np.float32)

    list_v, list_f = [], []
    fill_in_the_lists('model_1.obj', list_v, list_f)

    draw_model(img_mat, z_buffer, list_v, list_f)

    img = Image.fromarray(img_mat, mode='RGB')
    img = ImageOps.flip(img)
    img.save('frame.png')


main()