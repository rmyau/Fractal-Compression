import math
import numpy as np


def find_average_pixel(data_ar):
    average_pixel = np.sum(data_ar) / (data_ar.shape[0] * data_ar.shape[1])
    return average_pixel


def standart_deviation(data):  # среднеквадратическое отклонение
    data_ar = np.array(data)
    s_d = np.sqrt((1 / (data_ar.shape[0] * data_ar.shape[1])) * np.sum(np.square(data_ar - find_average_pixel(data_ar))))
    return s_d


def skewness(data):  # ассиметрия
    data_ar = np.array(data)
    aver_p = find_average_pixel(data_ar)
    stand_dev = standart_deviation(data)
    if stand_dev != 0.0:
        s = (np.sum((np.linalg.matrix_power(data_ar - aver_p, 3)) / np.power(stand_dev, 3)))
        s /= (data_ar.shape[0] * data_ar.shape[1])
    else:
        s = 0
    return s


def find_center(data):  # координаты центрального пикселя
    y = len(data) // 2
    x = len(data[0]) // 2
    return x, y


def find_distance_center(data):  # R
    data_ar = np.array(data)
    xr, yr = find_center(data)
    n, m = data_ar.shape[0], data_ar.shape[1]
    res = 0
    for i in range(n):
        for j in range(m):
            res += math.sqrt(math.pow(i - yr, 2) + math.pow(j - xr, 2))
    return res / (n * m)


def maximum_gradient(data):
    xr, yr = find_center(data)
    data_ar = np.array(data)
    aver_p = find_average_pixel(data_ar)
    n, m = data_ar.shape[0], data_ar.shape[1]
    distance_x_center, distance_y_center = 0, 0
    horizont_gradient, vertical_gradient = 0, 0
    for i in range(n):
        for j in range(m):
            d_x = math.pow(j - xr, 2)
            d_y = math.pow(i - yr, 2)
            distance_y_center += d_y
            distance_x_center += d_x
            d = data_ar[i][j] - aver_p
            horizont_gradient += (d_y * d)
            vertical_gradient += (d_x * d)

    horizont_gradient /= distance_y_center
    vertical_gradient /= distance_x_center
    return max(horizont_gradient, vertical_gradient)


def beta(data):
    data_ar = np.array(data)
    xr, yr = find_center(data)
    aver_p = find_average_pixel(data_ar)
    n, m = data_ar.shape[0], data_ar.shape[1]
    R = find_distance_center(data)
    sumR_, result = 0, 0
    for i in range(n):
        for j in range(m):
            R_ = R - math.sqrt(math.pow(i - yr, 2) + math.pow(j - xr, 2))
            sumR_ += math.pow(R_, 2)
            result += (data_ar[i][j] - aver_p) * R_
    return result / sumR_


def neighbor_contrast(data, k=1):  # межпиксельная контрастность
    data_ar = np.array(data)
    n, m = data_ar.shape[0], data_ar.shape[1]
    result = 0
    for i in range(k, n):
        for j in range(k, m):
            result += abs(data_ar[i][j] - data_ar[i - k][j]) + abs(data_ar[i][j] - data_ar[i][j - k])
    return result / ((n - k) * (m - k))


