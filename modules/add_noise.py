import cv2
import numpy as np
import random
import os

def add_salt_and_pepper(image, amount):
    """
    این تابع نویز نمک و فلفل را به تصویر اضافه می‌کند.

    Args:
        image: تصویر ورودی به صورت آرایه NumPy.
        amount: میزان نویز. مقداری بین 0 و 1 است.

    Returns:
        تصویر با نویز به صورت آرایه NumPy.
    """
    noisy = image.copy()
    h, w = noisy.shape[:2]
    nb_pixels = h * w
    amount = int(amount * nb_pixels)

    for c in range(3): # حلقه برای کانال‌های رنگی
        # اضافه کردن نویز نمک
        for _ in range(amount//2):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)
            noisy[y][x][c] = 255

        # اضافه کردن نویز فلفل
        for _ in range(amount//2):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)
            noisy[y][x][c] = 0
    return noisy


image_folder = "img"

for filename in os.listdir(image_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error reading image: {image_path}")
            continue

        image = cv2.resize(image, (256, 256))

        # # تبدیل تصویر به grayscale در صورت لزوم
        # if len(image.shape) == 3:
        #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # else:
        #     gray = image

        # اضافه کردن نویز (2 تا 7 درصد پیکسل‌ها نویز می‌گیرند)
        noise_percent = random.randint(2, 7)
        noisy_image = add_salt_and_pepper(image, noise_percent/100)

        # نمایش تصویر اصلی و تصویر با نویز
        cv2.imshow('Original Image', image)
        cv2.imshow('Noisy Image', noisy_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # ذخیره تصویر با نویز (اختیاری)
        cv2.imwrite(f'noisy-color{filename}.jpg', noisy_image) 

cv2.destroyAllWindows()