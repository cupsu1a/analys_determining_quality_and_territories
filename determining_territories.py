# ver 1
# import cv2
# import numpy as np
#
# # Загружаем изображение
# my_photo = cv2.imread('data2/flour_556.jpg')
# my_photo = cv2.resize(my_photo, None, fx=0.3, fy=0.3)  # Уменьшаем изображение
#
# # Преобразуем в оттенки серого
# img_grey = cv2.cvtColor(my_photo, cv2.COLOR_BGR2GRAY)
#
# # Задаем порог
# thresh = 182
#
# # Получаем бинаризованное изображение
# ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
#
# # Находим контуры
# contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# # Создаем пустую картинку для отображения контуров
# img_contours = np.zeros(my_photo.shape, dtype=np.uint8)
#
# # Отображаем контуры
# cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 1)
# img_contours = img_contours[1:581, 1:776]
#
# # Рисуем прямоугольники вокруг каждого контура
# for contour in contours:
#     # Вычисляем ограничивающий прямоугольник
#     x, y, w, h = cv2.boundingRect(contour)
#
#     # Рисуем прямоугольник на изображении
#     cv2.rectangle(img_contours, (x, y), (x + w, y + h), (0, 255, 0), 1)
#
# # Обрезаем изображение (если нужно)
#
# # Сохраняем изображение с контурами и прямоугольниками
# cv2.imwrite('image_with_rectangles.jpg', img_contours)
#
# # Отображаем изображение
# cv2.imshow('contours', img_contours)
#
#
# # Ждем нажатия клавиши
# cv2.waitKey()
# cv2.destroyAllWindows()
# ver2
# import cv2
# import numpy as np
#
# # Загружаем изображение
# my_photo = cv2.imread('data2/flour_556.jpg')
# my_photo = cv2.resize(my_photo, None, fx=0.3, fy=0.3)  # Уменьшаем изображение
#
# # Преобразуем в оттенки серого
# img_grey = cv2.cvtColor(my_photo, cv2.COLOR_BGR2GRAY)
#
# # Задаем порог
# thresh = 182
#
# # Получаем бинаризованное изображение
# ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
#
# # Находим контуры
# contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# print(hierarchy)
# # Удаляем контур, охватывающий всю область изображения
# max_area = 0
# max_contour_index = -1
#
# for i, contour in enumerate(contours):
#     area = cv2.contourArea(contour)  # Вычисляем площадь контура
#     if area > max_area:
#         max_area = area
#         max_contour_index = i
#
# # Если найден контур, охватывающий всю область, удаляем его
# if max_contour_index != -1:
#     contours = [contours[i] for i in range(len(contours)) if i != max_contour_index]
#
# # Создаем пустую картинку для отображения контуров
# img_contours = np.zeros(my_photo.shape, dtype=np.uint8)
#
# # Отображаем контуры
# cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 1)
#
# # Рисуем прямоугольники вокруг каждого контура
# for contour in contours:
#     # Вычисляем ограничивающий прямоугольник
#     x, y, w, h = cv2.boundingRect(contour)
#
#     # Рисуем прямоугольник на изображении
#     cv2.rectangle(img_contours, (x, y), (x + w, y + h), (0, 255, 0), 1)
#
# # Обрезаем изображение (если нужно)
# # img_contours = img_contours[1:581, 1:776]
#
# # Сохраняем изображение с контурами и прямоугольниками
# cv2.imwrite('image_with_rectangles.jpg', img_contours)
#
# # Отображаем изображение
# cv2.imshow('contours', img_contours)
#
# # Сохраняем контуры в текстовый файл
# with open("array.txt", "w") as file:
#     for contour in contours:
#         # Преобразуем координаты контура в строку и записываем в файл
#         file.write(" ".join(map(str, contour.flatten())) + "\n")
#
# # Ждем нажатия клавиши
# cv2.waitKey()
# cv2.destroyAllWindows()
# ver 3
# import cv2
# import numpy as np
#
# # Загружаем изображение
# my_photo = cv2.imread('data2/flour_556.jpg')
# my_photo = cv2.resize(my_photo, None, fx=0.3, fy=0.3)  # Уменьшаем изображение
#
# # Преобразуем в оттенки серого
# img_grey = cv2.cvtColor(my_photo, cv2.COLOR_BGR2GRAY)
#
# # Задаем порог
# thresh = 183
#
# # Получаем бинаризованное изображение
# ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
#
# # Находим контуры
# contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# # Удаляем контур, охватывающий всю область изображения
# max_area = 0
# max_contour_index = -1
#
# for i, contour in enumerate(contours):
#     area = cv2.contourArea(contour)  # Вычисляем площадь контура
#     if area > max_area:
#         max_area = area
#         max_contour_index = i
#
# # Если найден контур, охватывающий всю область, удаляем его
# if max_contour_index != -1:
#     contours = [contours[i] for i in range(len(contours)) if i != max_contour_index]
#
# # Создаем пустую картинку для отображения контуров
# img_contours = np.zeros(my_photo.shape, dtype=np.uint8)
#
# # Удаляем маленькие контуры по площади
# min_contour_area = 200  # Минимальная площадь контура для сохранения
# filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]
#
# # Отображаем контуры
# cv2.drawContours(img_contours, filtered_contours, -1, (255, 255, 255), 1)
#
# # Рисуем прямоугольники вокруг каждого контура
# for contour in filtered_contours:
#     # Вычисляем ограничивающий прямоугольник
#     x, y, w, h = cv2.boundingRect(contour)
#
#     # Рисуем прямоугольник на изображении
#     cv2.rectangle(img_contours, (x, y), (x + w, y + h), (0, 255, 0), 1)
#
# # Обрезаем изображение (если нужно)
# # img_contours = img_contours[1:581, 1:776]
#
# # Сохраняем изображение с контурами и прямоугольниками
# cv2.imwrite('image_with_rectangles.jpg', img_contours)
#
# # Отображаем изображение
# cv2.imshow('contours', img_contours)
#
# # Сохраняем контуры в текстовый файл
# with open("array.txt", "w") as file:
#     for contour in filtered_contours:
#         # Преобразуем координаты контура в строку и записываем в файл
#         file.write(" ".join(map(str, contour.flatten())) + "\n")
#
# # Ждем нажатия клавиши
# cv2.waitKey()
# cv2.destroyAllWindows()
# ver 4
# import cv2
# import numpy as np
#
# # Загружаем изображение
# my_photo = cv2.imread('data2/flour_556.jpg')
# my_photo = cv2.resize(my_photo, None, fx=0.3, fy=0.3)  # Уменьшаем изображение
#
# # Преобразуем в оттенки серого
# img_grey = cv2.cvtColor(my_photo, cv2.COLOR_BGR2GRAY)
#
# # Задаем порог
# thresh = 183
#
# # Получаем бинаризованное изображение
# ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
#
# # Находим контуры
# contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# # Удаляем контур, охватывающий всю область изображения
# max_area = 0
# max_contour_index = -1
#
# for i, contour in enumerate(contours):
#     area = cv2.contourArea(contour)  # Вычисляем площадь контура
#     if area > max_area:
#         max_area = area
#         max_contour_index = i
#
# # Если найден контур, охватывающий всю область, удаляем его
# if max_contour_index != -1:
#     contours = [contours[i] for i in range(len(contours)) if i != max_contour_index]
#
# # Удаляем маленькие контуры по площади
# min_contour_area = 300  # Минимальная площадь контура для сохранения
# filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]
#
# # Функция для проверки пересечения двух прямоугольников
# def do_rectangles_intersect(rect1, rect2):
#     x1, y1, w1, h1 = rect1
#     x2, y2, w2, h2 = rect2
#     return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)
#
# # Объединяем контуры, если их ограничивающие прямоугольники пересекаются
# merged_contours = []
# used_contours = set()
#
# for i, contour1 in enumerate(filtered_contours):
#     if i in used_contours:
#         continue
#     rect1 = cv2.boundingRect(contour1)
#     for j, contour2 in enumerate(filtered_contours):
#         if j <= i or j in used_contours:
#             continue
#         rect2 = cv2.boundingRect(contour2)
#         if do_rectangles_intersect(rect1, rect2):
#             # Объединяем контуры
#             merged_contour = np.vstack((contour1, contour2))
#             merged_contours.append(merged_contour)
#             used_contours.add(i)
#             used_contours.add(j)
#             break
#
# # Добавляем оставшиеся контуры, которые не были объединены
# for i, contour in enumerate(filtered_contours):
#     if i not in used_contours:
#         merged_contours.append(contour)
#
# # Создаем пустую картинку для отображения контуров
# img_contours = np.zeros(my_photo.shape, dtype=np.uint8)
#
# # Отображаем контуры
# cv2.drawContours(img_contours, merged_contours, -1, (255, 255, 255), 1)
#
# # Рисуем прямоугольники вокруг каждого контура
# for contour in merged_contours:
#     # Вычисляем ограничивающий прямоугольник
#     x, y, w, h = cv2.boundingRect(contour)
#
#     # Рисуем прямоугольник на изображении
#     cv2.rectangle(my_photo, (x, y), (x + w, y + h), (0, 255, 0), 1)
# print(len(merged_contours))
# # Обрезаем изображение (если нужно)
# # img_contours = img_contours[1:581, 1:776]
#
# # Сохраняем изображение с контурами и прямоугольниками
# cv2.imwrite('image_rectangles.jpg', img_contours)
#
# # Отображаем изображение
# cv2.imshow('contours', my_photo)
#
# # Сохраняем контуры в текстовый файл
# with open("array.txt", "w") as file:
#     for contour in merged_contours:
#         # Преобразуем координаты контура в строку и записываем в файл
#         file.write(" ".join(map(str, contour.flatten())) + "\n")
#
# # Ждем нажатия клавиши
# cv2.waitKey()
# cv2.destroyAllWindows()
# ver 5

import cv2
import numpy as np


def find_contours(image, threshold_binary=183, min_contour_area=300):
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, thresh_img = cv2.threshold(image_grey, threshold_binary, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_contour_index = -1

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour_index = i

    if max_contour_index != -1:
        contours = [contours[i] for i in range(len(contours)) if i != max_contour_index]

    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

    return filtered_contours

def do_rectangles_intersect(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

def merge_contours(filtered_contours):
    merged_contours = []
    used_contours = set()

    for i, contour1 in enumerate(filtered_contours):
        if i in used_contours:
            continue
        rect1 = cv2.boundingRect(contour1)
        for j, contour2 in enumerate(filtered_contours):
            if j <= i or j in used_contours:
                continue
            rect2 = cv2.boundingRect(contour2)
            if do_rectangles_intersect(rect1, rect2):
                merged_contour = np.vstack((contour1, contour2))
                merged_contours.append(merged_contour)
                used_contours.add(i)
                used_contours.add(j)
                break

    for i, contour in enumerate(filtered_contours):
        if i not in used_contours:
            merged_contours.append(contour)
    return merged_contours

def draw_contours_and_rectangles(image, merged_contours):
    image_with_contours = np.zeros(image.shape, dtype=np.uint8)
    image_with_rectangles = image.copy()
    cv2.drawContours(image_with_contours, merged_contours, -1, (255, 255, 255), 1)

    for contour in merged_contours:
        x, y, w, h = cv2.boundingRect(contour)
        image_with_rectangles = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

        cropped_image = image[y + 1:y + h, x + 1:x + w]
        cv2.imwrite(f'data3/cropped_image_{x}_{y}.jpg', cropped_image)

    return image_with_rectangles, image_with_contours


def result_determining(image_path):
    image = cv2.resize(cv2.imread(image_path), None, fx=0.3, fy=0.3)
    filtered_contours = find_contours(image)
    merged_contours = merge_contours(filtered_contours)
    image_with_rectangles, image_with_contours = draw_contours_and_rectangles(image, merged_contours)
    # cv2.imshow('rectangles', image_with_rectangles)
    # cv2.imshow('contours', image_with_contours)
    # cv2.waitKey(0)
    return image_with_rectangles, merged_contours
