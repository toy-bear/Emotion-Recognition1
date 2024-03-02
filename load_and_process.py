import pandas as pd
import cv2
import numpy as np


dataset_path = 'fer2013/fer2013/fer2013.csv'
image_size=(48,48)


# �������ݼ������������ݹ�һ��
def load_fer2013():
        data = pd.read_csv(dataset_path)
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:     #��ÿһ�����ֵ��������б�����ѡ
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'),image_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)  #copyһ��faces�ļ�
        faces = np.expand_dims(faces, -1)   #��-1��λ������faces expand_dims���ڸı�������״
        emotions = pd.get_dummies(data['emotion']).as_matrix()  # get_dummies ������Ҫ���ڶ������������ One-Hot ����
        return faces, emotions

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x
#�Ӵ�Ȩ�أ�ʹͼ�������