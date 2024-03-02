# -*- coding: utf-8 -*-
# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np  # 数据处理的库 numpy
import argparse
import imutils
import time
import dlib
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMovie



class eye:


    def eye_aspect_ratio(self, eye):
        # 垂直眼标志（X，Y）坐标
        A = dist.euclidean(eye[1], eye[5])  # 计算两个集合之间的欧式距离
        B = dist.euclidean(eye[2], eye[4])
        # 计算水平之间的欧几里得距离
        # 水平眼标志（X，Y）坐标
        C = dist.euclidean(eye[0], eye[3])
        # 眼睛长宽比的计算
        ear = (A + B) / (2.0 * C)
        # 返回眼睛的长宽比
        return ear

    def kaishi(self, frame, bmp):
        EYE_AR_THRESH = 0.2
        EYE_AR_CONSEC_FRAMES = 3
        # 初始化帧计数器和眨眼总数
        COUNTER = 0
        TOTAL = 0
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        frame = imutils.resize(frame, width=720)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = dlib.get_frontal_face_detector()

        rects = detector(gray, 0)


        frameClone = frame.copy()
        # 第六步：使用detector(gray, 0) 进行脸部位置检测

        # 第七步：循环脸部位置信息，使用predictor(gray, rect)获得脸部特征位置的信息
        for rect in rects:

            # 第八步：将脸部特征信息转换为数组array的格式
            # shape = face_utils.shape_to_np(shape)

            # 显示图片在panel上：
            predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)


            # 第九步：提取左眼和右眼坐标
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            # 第十步：构造函数计算左右眼的EAR值，使用平均值作为最终的EAR
            leftEAR = self.eye_aspect_ratio(self,leftEye)
            rightEAR = self.eye_aspect_ratio(self, rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # 第十一步：使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frameClone, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frameClone, [rightEyeHull], -1, (0, 255, 0), 1)


            # 第十二步：进行画图操作，用矩形框标注人脸
            left = rect.left()
            top = rect.top()
            right = rect.right()
            bottom = rect.bottom()
            cv2.rectangle(frameClone, (left, top), (right, bottom), (0, 255, 0), 3)


            '''
                分别计算左眼和右眼的评分求平均作为最终的评分，如果小于阈值，则加1，如果连续3次都小于阈值，则表示进行了一次眨眼活动
            '''
            # 第十三步：循环，满足条件的，眨眼次数+1
            if ear < EYE_AR_THRESH:  # 眼睛长宽比：0.2
                COUNTER += 1

            else:
                # 如果连续3次都小于阈值，则表示进行了一次眨眼活动
                if COUNTER >= EYE_AR_CONSEC_FRAMES:  # 阈值：3
                    TOTAL += 1
                    # text.AppendText("眨眼")
                # 重置眼帧计数器
                COUNTER = 0



            for (x, y) in shape:
                 cv2.circle(frameClone, (x, y), 1, (0, 0, 255), -1)

            # 第十五步：进行画图操作，同时使用cv2.putText将眨眼次数进行显示
            cv2.putText(frameClone, "Faces: {}".format(len(rects)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                        2)
            cv2.putText(frameClone, "Blinks: {}".format(TOTAL), (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frameClone, "COUNTER: {}".format(COUNTER), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                         2)
            cv2.putText(frameClone, "EAR: {:.2f}".format(ear), (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print('眼睛实时长宽比:{:.2f} '.format(ear))

        show = cv2.cvtColor(frameClone, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        bmp.setPixmap(QtGui.QPixmap.fromImage(showImage))
        QtWidgets.QApplication.processEvents()

        # 窗口显示 show with opencv
        # cv2.imshow("Frame", frame)
        # wx.CallAfter(pub.sendMessage, 'update')

        # if the `q` key was pressed, break from the loop

    # 释放摄像头 release camera




# 定义两个常数
# 眼睛长宽比
# 闪烁阈值
