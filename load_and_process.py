import pandas as pd
import cv2
import numpy as np


dataset_path = 'fer2013/fer2013/fer2013.csv'
image_size=(48,48)


# 载入数据集，并进行数据归一化
def load_fer2013():
        data = pd.read_csv(dataset_path)
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:     #将每一个出现的人脸进行遍历框选
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'),image_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)  #copy一个faces文件
        faces = np.expand_dims(faces, -1)   #在-1的位置增加faces expand_dims用于改变数组形状
        emotions = pd.get_dummies(data['emotion']).as_matrix()  # get_dummies 方法主要用于对类别型特征做 One-Hot 编码
        return faces, emotions

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x
#加大权重，使图像更清晰