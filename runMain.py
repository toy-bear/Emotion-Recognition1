# -*- coding: utf-8 -*-
"""
运行本项目需要安装的库：
    keras 2.3.0
    PyQt5 5.17.7
    pandas 2.0.3
    scikit-learn 1.3.0
    tensorflow 2.12.0
    imutils 0.5.2
    opencv-python 4.10.25
    dlib 19.24.0
    matplotlib 3.7.1  # 注意：此依赖包为第二版新增，请注意安装

点击运行主程序runMain.py
"""
# @Time    : 2019/5/25 18:29
# @Author  : WuXian
# @Email   : xianwu@shu.edu.cn
# @blog    : wuxian.blog.csdn.net
# @Software: PyCharm
# @version : 2.1 2020/5/12

import warnings
import os
# 忽略警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings(action='ignore')


from EmotionRecongnition import Ui_MainWindow
from sys import argv, exit
from PyQt5.QtWidgets import QApplication,QMainWindow


if __name__ == '__main__':
    app = QApplication(argv)

    window = QMainWindow()
    ui = Ui_MainWindow(window)

    window.show()
    exit(app.exec_())
