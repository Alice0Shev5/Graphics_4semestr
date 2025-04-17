import numpy as np
from PIL import Image, ImageOps

width, height = 4000, 4000
img_mat = np.zeros((width, height, 3), dtype=np.uint8)
z_buffer = np.full((width, height), 60000, dtype=np.float32)


class Drawing:
    model_name = None
    texture_name = None

    width, height = 4000, 4000
    width_texture, height_texture = 1024, 1024

    shift = None
    zoom = None
    rotate = None

    list_v, list_vt, list_f_vt, list_f = [], [], [], []

    def __init__(self, model_name, width_model, height_model, rotate=None, shift=None, zoom=7000, texture_name=None,
                 width_texture=1024, height_texture=1024):

        self.model_name = model_name
        self.width, self.height = width_model, height_model

        if shift is None:
            shift = [0, -0.04, 0.24]

        self.shift = shift
        self.zoom = zoom
        self.rotate = rotate

        self.texture_name = texture_name
        self.width_texture, self.height_texture = width_texture, height_texture

        self.list_v, self.list_vt, self.list_f_vt, self.list_f = [], [], [], []

    def main(self):
        self.fill_in_the_lists()
        self.draw_model()

    def fill_in_the_lists(self):
        model = open(self.model_name)

        for line in model:
            s = line.split()
            if s:
                if s[0] == 'v':

                    self.list_v.append(np.array([float(s[1]), float(s[2]), float(s[3])]))
                    rotate_and_shift_vertex(self.list_v[-1], self.shift, self.rotate)

                else:
                    if s[0] == 'vt':
                        coords = [float(x) for x in s[1:]]
                        while len(coords) < 3:
                            coords.append(0.0)
                        self.list_vt.append(np.array(coords[:3]))

                    else:
                        if s[0] == 'f':
                            element_f, element_f_vt = (), ()

                            for i in range(1, len(s)):
                                element_f += (int(s[i].split('/')[0]),)
                                element_f_vt += (int(s[i].split('/')[1]),)

                            self.list_f.append(element_f)
                            self.list_f_vt.append(element_f_vt)

    def draw_model(self):
        vertexes_normal = fill_in_normal_triangle_map(self.list_f, self.list_v)

        if self.texture_name:
            texture_img = ImageOps.flip(Image.open(self.texture_name))
            rgb_im = texture_img.convert('RGB')
        else:
            rgb_im = None

        for i in range(len(self.list_f)):
            print(i)
            vertexes = self.list_f[i]
            len_vertexes = len(vertexes)

            for j in range(len_vertexes):

                vertex1, vertex2, vertex3 = (vertexes[j % len_vertexes],
                                             vertexes[(j + 1) % len_vertexes], vertexes[(j + 2) % len_vertexes])

                x0, y0, z0, x1, y1, z1, x2, y2, z2 = get_coor(self.list_v, vertex1 - 1, vertex2 - 1, vertex3 - 1)

                list_intensity = get_intensity(vertexes_normal, vertex1, vertex2, vertex3)

                list_texture = [self.list_vt[self.list_f_vt[i][j % len_vertexes] - 1],
                                self.list_vt[self.list_f_vt[i][(j + 1) % len_vertexes] - 1],
                                self.list_vt[self.list_f_vt[i][(j + 2) % len_vertexes] - 1]]

                if check_cosine_of_light(x0, y0, z0, x1, y1, z1, x2, y2, z2):
                    self.draw_triangle_part1(list_intensity, x0, y0, z0, x1, y1, z1, x2, y2, z2, rgb_im, list_texture)

    def draw_triangle_part1(self, list_intensity, x0, y0, z0, x1, y1, z1, x2, y2, z2, rgb_im=None, list_texture=None):

        p_x0, p_y0, p_x1, p_y1, p_x2, p_y2 = set_zoom_for_drawing(x0, y0, z0, x1, y1, z1, x2, y2, z2, self.zoom)
        p_x_min, p_x_max, p_y_min, p_y_max = get_min_and_max_XY(p_x0, p_y0, p_x1, p_y1, p_x2, p_y2, self.width,
                                                                self.height)

        intensity0, intensity1, intensity2 = list_intensity[0], list_intensity[1], list_intensity[2]

        for x in range(p_x_min, p_x_max):
            for y in range(p_y_min, p_y_max):
                self.draw_triangle_part2(x, y, p_x0, p_y0, p_x1, p_y1, p_x2, p_y2, z0, z1, z2,
                                         intensity0, intensity1, intensity2, rgb_im, list_texture)

    def draw_triangle_part2(self, x, y, p_x0, p_y0, p_x1, p_y1, p_x2, p_y2, z0, z1, z2,
                            intensity0, intensity1, intensity2, rgb_im=None, list_texture=None):

        lambda0, lambda1, lambda2 = get_bicentric_coor(x, y, p_x0, p_y0, p_x1, p_y1, p_x2, p_y2)
        if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
            z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2

            if self.texture_name:
                color = get_color_of_the_pixel_in_the_texture(rgb_im, list_texture, lambda0, lambda1, lambda2)
            else:
                color = (-225 * (lambda0 * intensity0 + lambda1 * intensity1 + lambda2 * intensity2), 0, 0)

            if color[0] >= 0:
                check_z_and_change_z_buffer_and_img_mat(z, x, y, color)


def check_z_and_change_z_buffer_and_img_mat(z, x, y, color):
    global z_buffer, img_mat
    if z <= z_buffer[y][x]:
        z_buffer[y][x] = z
        img_mat[y, x] = color


def check_cosine_of_light(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    normal = get_normal_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    cosine_of_the_light = normal[2]

    return cosine_of_the_light < 0


def get_bicentric_coor(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1

    return lambda0, lambda1, lambda2


def get_min_and_max_XY(p_x0, p_y0, p_x1, p_y1, p_x2, p_y2, max_pixel_x, max_pixel_y):
    p_x_min = int(round(min(p_x0, p_x1, p_x2))) if min(p_x0, p_x1, p_x2) > 0 else 0
    p_x_max = int(round(max(p_x0, p_x1, p_x2))) + 1 if max(p_x0, p_x1, p_x2) < max_pixel_x else max_pixel_x
    p_y_min = int(round(min(p_y0, p_y1, p_y2))) if min(p_y0, p_y1, p_y2) > 0 else 0
    p_y_max = int(round(max(p_y0, p_y1, p_y2))) + 1 if max(p_y0, p_y1, p_y2) < max_pixel_y else max_pixel_y

    return p_x_min, p_x_max, p_y_min, p_y_max


def get_normal_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    n = np.array([((y1 - y2) * (z1 - z0) - (z1 - z2) * (y1 - y0)), ((x1 - x2) * (z1 - z0) - (z1 - z2) * (x1 - x0)),
                  ((x1 - x2) * (y1 - y0) - (y1 - y2) * (x1 - x0))])
    n = n / np.linalg.norm(n)
    return n


def get_coor(list_v, vertex0, vertex1, vertex2):
    x0, y0, z0 = float(list_v[vertex0][0]), float(list_v[vertex0][1]), float(list_v[vertex0][2])
    x1, y1, z1 = float(list_v[vertex1][0]), float(list_v[vertex1][1]), float(list_v[vertex1][2])
    x2, y2, z2 = float(list_v[vertex2][0]), float(list_v[vertex2][1]), float(list_v[vertex2][2])

    return x0, y0, z0, x1, y1, z1, x2, y2, z2


def get_intensity(vertexes_normal, vertex0, vertex1, vertex2):
    n1 = vertexes_normal.get(vertex0)
    n2 = vertexes_normal.get(vertex1)
    n3 = vertexes_normal.get(vertex2)

    intensity0 = (n1 / np.linalg.norm(n1))[2]
    intensity1 = (n2 / np.linalg.norm(n2))[2]
    intensity2 = (n3 / np.linalg.norm(n3))[2]

    return [intensity0, intensity1, intensity2]


def get_color_of_the_pixel_in_the_texture(rgb_im, list_texture, lambda0, lambda1, lambda2, width_img=1024,
                                          height_img=1024):
    u0, v0 = list_texture[0][0], list_texture[0][1]
    u1, v1 = list_texture[1][0], list_texture[1][1]
    u2, v2 = list_texture[2][0], list_texture[2][1]

    pixel = (
        int(width_img * (u0 * lambda0 + u1 * lambda1 + u2 * lambda2)),
        int(height_img * (v0 * lambda0 + v1 * lambda1 + v2 * lambda2)))

    r, g, b = rgb_im.getpixel(pixel)

    return r, g, b


def get_rotation_matrix(rotate):
    a, b, c = rotate
    # a, b, c = 31.4, 28.94, 31.4  # 31, 29.9, 31 # 31.4, 28.85, 31.4  # 31.4, 29.9, 31.4

    R_x = np.array([(1, 0, 0), (0, np.cos(a), np.sin(a)), (0, -np.sin(a), np.cos(a))])
    R_y = np.array([(np.cos(b), 0, np.sin(b)), (0, 1, 0), (-np.sin(b), 0, np.cos(b))])
    R_z = np.array([(np.cos(c), np.sin(c), 0), (-np.sin(c), np.cos(c), 0), (0, 0, 1)])

    return R_x, R_y, R_z


def quaternion_mult(p, q):
    a1, b1, c1, d1 = p
    a2, b2, c2, d2 = q
    return np.array([a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
                     a1 * b2 - b1 * a2 + c1 * d2 - d1 * c2,
                     a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
                     a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2])


def quaternion_conj(q):
    a, b, c, d = q
    return np.array([a, -b, -c, -d])


#  ПРОБЛЕМА ТУТ
def set_zoom_for_drawing(x0, y0, z0, x1, y1, z1, x2, y2, z2, zoom=7000, average_width=2000, average_height=2000):
    x0, y0 = (x0 * zoom) / z0 + average_width, (y0 * zoom) / z0 + average_height
    x1, y1 = (x1 * zoom) / z1 + average_width, (y1 * zoom) / z1 + average_height
    x2, y2 = (x2 * zoom) / z2 + average_width, (y2 * zoom) / z2 + average_height

    # print(f"z0: {z0}, z1: {z1}, z2: {z2}")

    return x0, y0, x1, y1, x2, y2


def update_normal_triangle_map(list_of_vertex, normal, v_normal_triangle_map):
    for vertex in list_of_vertex:

        check_empty = v_normal_triangle_map.get(vertex)

        if check_empty is not None:
            v_normal_triangle_map[vertex] += normal

        else:
            v_normal_triangle_map[vertex] = normal


def fill_in_normal_triangle_map(list_f, list_v):
    v_normal_triangle_map = {}

    for parts in list_f:
        for i in range(len(parts)):
            x0, y0, z0, x1, y1, z1, x2, y2, z2 = get_coor(list_v,
                                                          parts[i % len(parts)] - 1, parts[(i + 1) % len(parts)] - 1,
                                                          parts[(i + 2) % len(parts)] - 1)
            normal = get_normal_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2)

            list_of_vertex = [parts[i % len(parts)], parts[(i + 1) % len(parts)], parts[(i + 2) % len(parts)]]

            update_normal_triangle_map(list_of_vertex, normal, v_normal_triangle_map)

    return v_normal_triangle_map


def rotate_and_shift_vertex(vertex, shift, rotate=None):
    if rotate is None:
        rotate = [31.4, 28.94, 31.4]

    if len(rotate) == 3:
        R_x, R_y, R_z = get_rotation_matrix(rotate)

        vertex[...] = np.dot(R_y, vertex)
        vertex[...] = np.dot(R_z, vertex)
        vertex[...] = np.dot(R_x, vertex)

    else:
        angle, a, b, c = rotate

        x, y, z = vertex[0], vertex[1], vertex[2]

        axis = np.array([a, b, c])
        axis = axis / np.linalg.norm(axis)

        q_rot = np.array([np.cos(angle / 2),
                          axis[0] * np.sin(angle / 2),
                          axis[1] * np.sin(angle / 2),
                          axis[2] * np.sin(angle / 2)])

        q_rot_new = quaternion_mult(quaternion_mult(q_rot, [0, x, y, z]), quaternion_conj(q_rot))
        vertex[...] = q_rot_new[1:]

    vertex[...] = vertex + np.array(shift)


def main():
    model2 = Drawing('model_1.obj', 4000, 4000, [60, 0, 1, 0], [0.1, -0.1, 0.24], 2000, 'bunny-atlas.jpg',
                     1024, 1024)
    model2.main()

    model = Drawing('model_frog.obj', 4000, 4000, [31, 27, 37], [0.5, -0.04, 7], 300, 'model_frog.jpg', 1024, 1024)
    model.main()

    img = Image.fromarray(img_mat, mode='RGB')
    img = ImageOps.flip(img)
    img.save('frame2.png')


main()
