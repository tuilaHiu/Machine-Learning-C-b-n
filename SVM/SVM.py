import cv2
from sklearn.svm import SVC
from process_data import get_data, readImage, reshape, normalize_data
import numpy as np

path = 'Data'


def distance(img, w, b):
    d = (b[0] + np.dot(img.T, w.T) / np.sqrt(np.sum(w ** 2)))
    return d


data, label = get_data(path)
data = data.reshape(1471, 128 * 128)
label = label.reshape(1471, 1)
# print(data.shape)
# print(label.shape)
# N = len(data)
# # label = label.reshape((1 * N,))
# data = data.T  # each sample is one row
clf = SVC(kernel='linear', C=1e5)  # just a big number
#
clf.fit(data, label)
#
w = clf.coef_
b = clf.intercept_
# # print('w = ', w)
# # print('b = ', b)
# # print(w.shape)
img = normalize_data(reshape(readImage('img_testtt.jpg')))
# print(img.shape)
print(distance(img, w, b))
