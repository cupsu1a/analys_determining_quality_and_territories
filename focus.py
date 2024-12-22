import cv2
import numpy as np
import os

def is_image_in_focus_fft(image, threshold=100):
    """
    Определяет, находится ли изображение в фокусе, используя анализ частотного спектра.

    :param image: Изображение в оттенках серого.
    :param threshold: Пороговое значение для определения фокуса.
    :return: True, если изображение в фокусе, иначе False.
    """
    # Применение Быстрого преобразования Фурье (FFT)
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)  # Сдвиг нулевой частоты в центр

    # Вычисление амплитудного спектра
    magnitude_spectrum = 20 * np.log(np.abs(fft_shift) + 1)

    # Среднее значение амплитудного спектра
    focus_measure = np.mean(magnitude_spectrum)

#    print(f"Значение фокуса (FFT): {focus_measure}")

    # Сравнение с пороговым значением
    return focus_measure > threshold, focus_measure

def process_images_in_folder(folder_path, threshold=100):
    """
    Обрабатывает все изображения в указанной папке, вычисляет focus_measure и удаляет изображения.

    :param folder_path: Путь к папке с изображениями.
    :param threshold: Пороговое значение для определения фокуса.
    :return: Среднее значение focus_measure для всех изображений.
    """
    focus_measures = []

    # Проверяем, существует ли папка
    if not os.path.exists(folder_path):
        print(f"Ошибка: Папка '{folder_path}' не существует.")
        return 0

    # Получаем список файлов в папке
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if not image_files:
        print(f"Ошибка: В папке '{folder_path}' нет изображений.")
        return 0

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        # Загружаем изображение
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Ошибка: Не удалось загрузить изображение '{image_file}'.")
            continue

        # Вычисляем focus_measure
        focus_measure = is_image_in_focus_fft(image, threshold)
        focus_measures.append(focus_measure[1])
        # Удаляем изображение после обработки
        os.remove(image_path)
        # print(f"Изображение '{image_file}' удалено.")

    # Вычисляем среднее значение focus_measure
    if focus_measures:
        average_focus_measure = np.mean(focus_measures)
#        print(f"Среднее значение focus_measure для всех изображений: {average_focus_measure}")
        return average_focus_measure
    else:
        print("Ошибка: Нет доступных изображений для обработки.")
        return 0


def result_focus(threshold):
    folder_path = 'data3'
    average_focus_measure = process_images_in_folder(folder_path)
    return average_focus_measure > threshold