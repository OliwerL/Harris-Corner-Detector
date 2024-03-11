import time
from functools import wraps

import cv2
import numpy as np
import matplotlib.pyplot as plt


def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('func:%r took: %2.4f sec' % (f.__name__, te-ts))
        return result
    return wrap

@timeit
def hariscorners(path='./gut.jpg'):
    img = cv2.imread(path)
    height, width, channels = img.shape
    b = np.zeros((height, width))
    g = np.zeros((height, width))
    r = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            b[i, j] = img[i, j, 0]
            g[i, j] = img[i, j, 1]
            r[i, j] = img[i, j, 2]

    gray_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            gray_image[i, j] = b[i, j] * 0.299 + r[i, j] * 0.587 + g[i, j] * 0.114

    # derivative_x = np.zeros((height, width), dtype=np.int32)
    # derivative_y = np.zeros((height, width), dtype=np.int32)
    # derivative_x2 = np.zeros((height, width), dtype=np.int32)
    # derivative_y2 = np.zeros((height, width), dtype=np.int32)
    # derivative_xy = np.zeros((height, width), dtype=np.int32)
    #
    # for i in range(1, height - 1):
    #     for j in range(1, width - 1):
    #         derivative_x[i, j] = gray_image[i, j + 1] - gray_image[i, j - 1]
    #         derivative_y[i, j] = gray_image[i + 1, j] - gray_image[i - 1, j]
    #
    # for i in range(1, height - 1):
    #     for j in range(1, width - 1):
    #         derivative_x2[i, j] = derivative_x[i, j + 1] - derivative_x[i, j - 1]
    #         derivative_y2[i, j] = derivative_y[i + 1, j] - derivative_y[i - 1, j]
    #         derivative_xy[i, j] = derivative_x[i + 1, j] - derivative_x[i - 1, j]

    derivative_x = np.gradient(gray_image, axis=1)
    derivative_y = np.gradient(gray_image, axis=0)
    derivative_x2 = derivative_x ** 2
    derivative_y2 = derivative_y ** 2
    derivative_xy = derivative_x * derivative_y

    det_M = derivative_x2 * derivative_y2 - derivative_xy ** 2
    trace_M = derivative_x2 + derivative_y2

    k = 0.05
    T = -100000
    corners = []

    harris_response = det_M - k * (trace_M ** 2)

    # for i in range(1, height - 1):
    #     for j in range(1, width - 1):
    #         harris_value = det_M[i, j] - k * (trace_M[i, j]) ** 2
    #         if harris_value > T:
    #             is_local_maximum = True
    #             for x in range(i - 1, i + 2):
    #                 for y in range(j - 1, j + 2):
    #                     if harris_value < det_M[x, y] - k * (trace_M[x, y]) ** 2:
    #                         is_local_maximum = False
    #                         break
    #                 if not is_local_maximum:
    #                     break
    #
    #             if is_local_maximum:
    #                 corners.append((i, j))

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if harris_response[i, j] < T:
                local_min = harris_response[i - 1:i + 2, j - 1:j + 2]
                if harris_response[i, j] == local_min.min():
                    corners.append((i, j))

    corners_image = gray_image.copy()
    for corner in corners:
        i, j = corner
        corners_image[i, j] = 255

    # Wyświetl obraz wynikowy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(derivative_x, cmap='gray')
    plt.title('Derivative X')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(derivative_y, cmap='gray')
    plt.title('Derivative Y')
    plt.axis('off')

    plt.show()

    # Wyświetl obraz z zaznaczonymi naroznikami
    plt.figure()
    plt.imshow(corners_image, cmap='gray')
    plt.title('Corners')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    hariscorners()
