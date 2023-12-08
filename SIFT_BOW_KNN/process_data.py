import os
import cv2
import numpy as np


def readImage(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("False", img_path)
    else:
        return cv2.resize(img, (128, 128))


def reshape(img):
    return img.reshape(1, img.shape[0] * img.shape[1])


def normalize_data(img):
    return cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def get_data(path):
    images = []
    for folder in os.listdir(path):
        for file in os.listdir(os.path.join(path, folder)):
            images.append(os.path.join(path, os.path.join(folder, file)))
    img = np.array([])
    label = np.array([])
    for temp in images:
        if ("cat" in temp):
            label = np.append(label, 0)
        else:
            label = np.append(label, 1)
        img = np.append(img, normalize_data(reshape(readImage(temp)))[0])
    return img, label
