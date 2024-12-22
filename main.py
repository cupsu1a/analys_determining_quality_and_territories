import cv2
import numpy
import os
from determining_territories import result_determining
from focus import result_focus


image_paths = 'data2'
if not os.path.exists(image_paths):
    print(f"Ошибка: Папка '{image_paths}' не существует.")

image_files = [f for f in os.listdir(image_paths) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

if not image_files:
    print(f"Ошибка: В папке '{image_paths}' нет изображений.")

for image_file in image_files:
        image_path = os.path.join(image_paths, image_file)
        image_with_rectangles, merged_contours = result_determining(image_path)
        if not result_focus(threshold=70):
            print(f'Изображение {image_path} не фокусе')
