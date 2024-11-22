import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from sklearn import svm
import joblib


def calcSiftFeature(img):
    # 设置图像sift特征关键点最大为200
    sift = cv2.SIFT_create(nfeatures=200)
    # 计算图片的特征点和特征点描述
    # Ensure the input tensor is on CPU and convert to numpy array
    img = img.cpu().numpy()

    # Initialize SIFT detector
    img = np.transpose(img, (1, 2, 0))
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # print(img.shape)
    # break

    # Detect SIFT features
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    # if descriptors is None:
    #     print("No descriptors")
    #     return
    return descriptors


# 计算词袋
def learnVocabulary(features):
    wordCnt = 50
    # criteria表示迭代停止的模式   eps---精度0.1，max_iter---满足超过最大迭代次数20
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    # 得到k-means聚类的初始中心点
    flags = cv2.KMEANS_RANDOM_CENTERS
    # 标签，中心 = kmeans(输入数据（特征)、聚类的个数K,预设标签，聚类停止条件、重复聚类次数、初始聚类中心点
    compactness, labels, centers = cv2.kmeans(
        features, wordCnt, None, criteria, 20, flags
    )
    return centers


# 计算特征向量
def calcFeatVec(features, centers):
    featVec = np.zeros((1, 50))
    for i in range(0, features.shape[0]):
        # 第i张图片的特征点
        fi = features[i]
        diffMat = np.tile(fi, (50, 1)) - centers
        # axis=1按行求和，即求特征到每个中心点的距离
        sqSum = (diffMat**2).sum(axis=1)
        dist = sqSum**0.5
        # 升序排序
        sortedIndices = dist.argsort()
        # 取出最小的距离，即找到最近的中心点
        idx = sortedIndices[0]
        # 该中心点对应+1
        featVec[0][idx] += 1
    return featVec


# 建立词袋
def build_center(images):
    features = np.float32([]).reshape(0, 128)
    idx_nune_desc = []
    for idx in range(images.shape[0]):
        img = images[idx]
        # 获取图片sift特征点
        img_f = calcSiftFeature(img)
        # 特征点加入训练数据
        # print("build_center", img_f.shape)
        # break
        if img_f is None:
            idx_nune_desc.append(idx)
            continue
        features = np.append(features, img_f, axis=0)
    # 训练集的词袋
    centers = learnVocabulary(features)
    # #将词袋保存
    filename = "./svm_centers.npy"
    np.save(filename, centers)
    print("词袋:", centers.shape)
    return centers, idx_nune_desc


# 计算训练集图片特征向量
def cal_vec(images):
    centers = np.load("./svm_centers.npy")
    print("center:", centers.shape)
    data_vec = np.float32([]).reshape(0, 50)  # 存放训练集图片的特征
    labels = np.float32([])
    # cate=[path+'/'+x for x in os.listdir(path) if os.path.isdir(path+'/'+x)]
    for idx in range(images.shape[0]):
        # 获取图片sift特征点
        # print(idx)
        img_f = calcSiftFeature(images[idx])
        print("img_f:", img_f.shape)
        # return
        if img_f is None:
            pass
        img_vec = calcFeatVec(img_f, centers)
        data_vec = np.append(data_vec, img_vec, axis=0)
        labels = np.append(labels, idx)
        if idx == 10:
            return
    print("data_vec:", data_vec.shape)
    print("image features vector done!")
    return data_vec, labels


# 训练SVM分类器
def SVM_Train(data_vec, labels):
    # 设置SVM模型参数
    clf = svm.SVC(decision_function_shape="ovo")
    # 利用x_train,y_train训练SVM分类器，获得参数
    clf.fit(data_vec, labels)
    joblib.dump(clf, "e:/flowers/svm/svm_model.m")


# train_images, train_labels, test_images, test_labels = read_cifar10_to_tensor(
#     "./data/cifar-10-batches-py/"
# )
# data_vec, labels = cal_vec(train_images)
