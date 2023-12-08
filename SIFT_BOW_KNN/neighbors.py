import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def get_data(path):
    images = []
    for folder in os.listdir(path):
        for file in os.listdir(os.path.join(path, folder)):
            images.append(os.path.join(path, os.path.join(folder, file)))
    img = []
    label = np.array([])
    for temp in images:
        if ("cat" in temp):
            label=np.append(label,temp)
        else:
            label=np.append(label,temp)
        img.append(readImage(temp))
    return img, label


def readImage(img_path):
    img = cv2.imread(img_path, 0)
    if img is None:
        print("False", img_path)
    else:
        return cv2.resize(img, (128, 128))


def getDescriptors(sift, img):
    kp, des = sift.detectAndCompute(img, None)
    return des


def vstackDescriptors(descriptor_list):
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    return descriptors


def clusterDescriptors(descriptors, no_clusters):
    kmeans = KMeans(n_clusters=no_clusters).fit(descriptors)
    return kmeans


def extractFeatures(kmeans, descriptor_list, image_count, no_clusters):
    im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])
    for i in range(image_count):
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, 128)
            idx = kmeans.predict(feature)
            im_features[i][idx] += 1

    return im_features


def process_data(path, no_clusters,n_neighbors):
    img, label = get_data(path)
    # check(img,label)
    sift = cv2.SIFT_create()
    descriptor_list = []
    image_count = len(img)
    for i in range(image_count):
        des = getDescriptors(sift, img[i])
        descriptor_list.append(des)
    descriptors = vstackDescriptors(descriptor_list)
    kmeans = clusterDescriptors(descriptors, no_clusters)
    im_features = extractFeatures(kmeans, descriptor_list, image_count, no_clusters)
    X_train, X_test, y_train, y_test=train_test_split(im_features, label, random_state=0, train_size=.75), kmeans
    clf = KNeighborsClassifier(n_neighbor, p=2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    return acc



