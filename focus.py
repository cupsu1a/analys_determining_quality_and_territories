import cv2
import numpy as np
import os


def is_image_in_focus_fft(image, threshold=100):

    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)

    magnitude_spectrum = 20 * np.log(np.abs(fft_shift) + 1)

    focus_measure = np.mean(magnitude_spectrum)

    # print(f"Значение фокуса (FFT): {focus_measure}")

    return focus_measure > threshold, focus_measure


def process_images_in_folder(folder_path, threshold=100):

    focus_measures = []

    if not os.path.exists(folder_path):
        print(f"Ошибка: Папка '{folder_path}' не существует.")
        return 0

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if not image_files:
        print(f"Ошибка: В папке '{folder_path}' нет изображений.")
        return 0

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Ошибка: Не удалось загрузить изображение '{image_file}'.")
            continue

        focus_measure = is_image_in_focus_fft(image, threshold)
        focus_measures.append(focus_measure[1])
        os.remove(image_path)
        # print(f"Изображение '{image_file}' удалено.")

    if focus_measures:
        average_focus_measure = np.mean(focus_measures)
        # print(f"Среднее значение focus_measure для всех изображений: {average_focus_measure}")
        return average_focus_measure
    else:
        print("Ошибка: Нет доступных изображений для обработки.")
        return 0


def result_focus(threshold):
    folder_path = 'data3'
    average_focus_measure = process_images_in_folder(folder_path)
    return average_focus_measure > threshold