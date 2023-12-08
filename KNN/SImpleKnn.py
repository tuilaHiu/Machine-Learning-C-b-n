import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


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
    img = []
    label = []
    for temp in images:
        if ("cat" in temp):
            label.append(0)
        else:
            label.append(1)
        img.append(normalize_data(reshape(readImage(temp)))[0])
    return img, label


def train_model(path, n_neighbor):
    data, label = get_data(path)
    X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=0, train_size=.75)
    clf = KNeighborsClassifier(n_neighbor, p=2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    return acc
    # return X_train[0][0], X_train[1],data[0]
path='Data'
acc=[]
# neighbors
for i in range(1,100):
    acc.append(train_model(path=path,n_neighbor=i))
    # neighbors.append(i)
plt.plot(acc)
plt.xlabel('neighbors')
plt.ylabel('Accuracy')
plt.yticks([max(acc)])

plt.show()

