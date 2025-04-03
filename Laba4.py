import numpy as np
from PIL import Image, ImageOps


def get_bicentric_coor(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1

    return lambda0, lambda1, lambda2


def get_min_and_max_XY(p_x0, p_y0, p_x1, p_y1, p_x2, p_y2, max_pixel):
    p_x_min = int(round(min(p_x0, p_x1, p_x2))) if min(p_x0, p_x1, p_x2) > 0 else 0
    p_x_max = int(round(max(p_x0, p_x1, p_x2))) + 1 if max(p_x0, p_x1, p_x2) < max_pixel else max_pixel
    p_y_min = int(round(min(p_y0, p_y1, p_y2))) if min(p_y0, p_y1, p_y2) > 0 else 0
    p_y_max = int(round(max(p_y0, p_y1, p_y2))) + 1 if max(p_y0, p_y1, p_y2) < max_pixel else max_pixel

    return p_x_min, p_x_max, p_y_min, p_y_max


def check_z_and_change_z_buffer_and_draw_pixel(image, z_buffer, z, x, y, color):
    if z <= z_buffer[y][x]:
        z_buffer[y][x] = z
        image[y, x] = color


def get_normal_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    n = np.array([((y1 - y2) * (z1 - z0) - (z1 - z2) * (y1 - y0)), ((x1 - x2) * (z1 - z0) - (z1 - z2) * (x1 - x0)),
                  ((x1 - x2) * (y1 - y0) - (y1 - y2) * (x1 - x0))])
    n = n / np.linalg.norm(n)
    return n


def get_rotation_matrix():
    a, b, c = 31.4, 28.94, 31.4  # 31, 29.9, 31 # 31.4, 28.85, 31.4  # 31.4, 29.9, 31.4
    R_x = np.array([(1, 0, 0), (0, np.cos(a), np.sin(a)), (0, -np.sin(a), np.cos(a))])
    R_y = np.array([(np.cos(b), 0, np.sin(b)), (0, 1, 0), (-np.sin(b), 0, np.cos(b))])
    R_z = np.array([(np.cos(c), np.sin(c), 0), (-np.sin(c), np.cos(c), 0), (0, 0, 1)])

    return R_x, R_y, R_z


def set_zoom_for_drawing(x0, y0, z0, x1, y1, z1, x2, y2, z2, average_width=2000, average_height=2000):
    zoom = 7000
    x0, y0 = (x0 * zoom) / z0 + average_width, (y0 * zoom) / z0 + average_height
    x1, y1 = (x1 * zoom) / z1 + average_width, (y1 * zoom) / z1 + average_height
    x2, y2 = (x2 * zoom) / z2 + average_width, (y2 * zoom) / z2 + average_height

    return x0, y0, x1, y1, x2, y2


def draw_triangle_part1(image, z_buffer, list_intensity, list_texture, x0, y0, z0, x1, y1, z1, x2, y2, z2, rgb_im):
    normal = get_normal_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    cosine_of_the_light = normal[2]

    if cosine_of_the_light < 0:

        # draw_triangle_part2(image, z_buffer, x0, y0, z0, x1, y1, z1, x2, y2, z2)

        p_x0, p_y0, p_x1, p_y1, p_x2, p_y2 = set_zoom_for_drawing(x0, y0, z0, x1, y1, z1, x2, y2, z2)
        p_x_min, p_x_max, p_y_min, p_y_max = get_min_and_max_XY(p_x0, p_y0, p_x1, p_y1, p_x2, p_y2, 4000)

        intensity0, intensity1, intensity2 = list_intensity[0], list_intensity[1], list_intensity[2]

        for x in range(p_x_min, p_x_max):
            for y in range(p_y_min, p_y_max):
                lambda0, lambda1, lambda2 = get_bicentric_coor(x, y, p_x0, p_y0, p_x1, p_y1, p_x2, p_y2)
                if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                    z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2

                    # color = (-225 * (lambda0 * intensity0 + lambda1 * intensity1 + lambda2 * intensity2), 0, 0)
                    color = get_color_of_the_pixel_in_the_texture(rgb_im, list_texture, lambda0, lambda1, lambda2)

                    if color[0] >= 0:
                        check_z_and_change_z_buffer_and_draw_pixel(image, z_buffer, z, x, y, color)


def get_coor(list_v, vertex0, vertex1, vertex2):
    x0, y0, z0 = float(list_v[vertex0][0]), float(list_v[vertex0][1]), float(list_v[vertex0][2])
    x1, y1, z1 = float(list_v[vertex1][0]), float(list_v[vertex1][1]), float(list_v[vertex1][2])
    x2, y2, z2 = float(list_v[vertex2][0]), float(list_v[vertex2][1]), float(list_v[vertex2][2])

    return x0, y0, z0, x1, y1, z1, x2, y2, z2


def update_normal_triangle_map(list_of_vertex, normal, f_normal_triangle_map, v_normal_triangle_map):
    for vertex in list_of_vertex:

        normal_list = f_normal_triangle_map.get(vertex)

        if normal_list:
            normal_list.append(normal)
            f_normal_triangle_map[vertex] = normal_list
            v_normal_triangle_map[vertex] += normal

        else:
            f_normal_triangle_map[vertex] = [normal]
            v_normal_triangle_map[vertex] = normal


def get_intensity(vertexes_normal, vertex0, vertex1, vertex2):
    n1 = vertexes_normal.get(vertex0)
    n2 = vertexes_normal.get(vertex1)
    n3 = vertexes_normal.get(vertex2)

    intensity0 = (n1 / np.linalg.norm(n1))[2]
    intensity1 = (n2 / np.linalg.norm(n2))[2]
    intensity2 = (n3 / np.linalg.norm(n3))[2]

    return [intensity0, intensity1, intensity2]


def fill_in_normal_triangle_map(list_f, list_v):
    f_normal_triangle_map = {}
    v_normal_triangle_map = {}

    for parts in list_f:
        x0, y0, z0, x1, y1, z1, x2, y2, z2 = get_coor(list_v, parts[0] - 1, parts[1] - 1, parts[2] - 1)
        normal = get_normal_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2)

        list_of_vertex = [parts[0], parts[1], parts[2]]

        update_normal_triangle_map(list_of_vertex, normal, f_normal_triangle_map, v_normal_triangle_map)

    return v_normal_triangle_map


def get_color_of_the_pixel_in_the_texture(rgb_im, list_texture, lambda0, lambda1, lambda2, width_img=1024,
                                          height_img=1024):

    u0, v0 = list_texture[0][0], list_texture[0][1]
    u1, v1 = list_texture[1][0], list_texture[1][1]
    u2, v2 = list_texture[2][0], list_texture[2][1]

    pixel = (
    int(width_img * (u0 * lambda0 + u1 * lambda1 + u2 * lambda2)), int(height_img * (v0 * lambda0 + v1 * lambda1 + v2 * lambda2)))

    r, g, b = rgb_im.getpixel(pixel)

    return r, g, b


def draw_model(img_mat, z_buffer, list_v, list_vt, list_f_vt, list_f):
    vertexes_normal = fill_in_normal_triangle_map(list_f, list_v)

    texture_img = ImageOps.flip(Image.open('bunny-atlas.jpg'))
    rgb_im = texture_img.convert('RGB')

    for i in range(len(list_f) - 1):
        print(i)
        vertexes = list_f[i]

        x0, y0, z0, x1, y1, z1, x2, y2, z2 = get_coor(list_v, vertexes[0] - 1, vertexes[1] - 1, vertexes[2] - 1)

        list_intensity = get_intensity(vertexes_normal, vertexes[0], vertexes[1], vertexes[2])
        list_texture = [list_vt[list_f_vt[i][0] - 1], list_vt[list_f_vt[i][1] - 1], list_vt[list_f_vt[i][2] - 1]]

        draw_triangle_part1(img_mat, z_buffer, list_intensity, list_texture, x0, y0, z0, x1, y1, z1, x2, y2, z2, rgb_im)


def fill_in_the_lists(model_name, list_v, list_vt, list_f_vt, list_f):
    model = open(model_name)

    for line in model:
        s = line.split()
        if s:
            if s[0] == 'v':

                list_v.append(np.array([float(s[1]), float(s[2]), float(s[3])]))
                rotate_and_shift_vertex(list_v[-1], [0, -0.04, 0.24])

            else:
                if s[0] == 'vt':
                    list_vt.append(np.array([float(s[1]), float(s[2]), 0]))
                else:
                    if s[0] == 'f':
                        list_f.append((int(s[1].split('/')[0]), int(s[2].split('/')[0]), int(s[3].split('/')[0])))
                        list_f_vt.append((int(s[1].split('/')[1]), int(s[2].split('/')[1]), int(s[3].split('/')[1])))


def rotate_and_shift_vertex(vertex, shift):
    R_x, R_y, R_z = get_rotation_matrix()

    vertex[...] = np.dot(R_y, vertex)
    vertex[...] = np.dot(R_z, vertex)
    vertex[...] = np.dot(R_x, vertex)

    vertex[...] = vertex + np.array(shift)


def main():
    width, height = 4000, 4000
    img_mat = np.zeros((width, height, 3), dtype=np.uint8)
    z_buffer = np.full((width, height), 10000, dtype=np.float32)

    list_v, list_vt, list_f_vt, list_f = [], [], [], []
    fill_in_the_lists('model_1.obj', list_v, list_vt, list_f_vt, list_f)

    draw_model(img_mat, z_buffer, list_v, list_vt, list_f_vt, list_f)

    img = Image.fromarray(img_mat, mode='RGB')
    img = ImageOps.flip(img)
    img.save('frame2.png')


main()
