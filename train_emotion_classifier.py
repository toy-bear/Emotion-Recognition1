"""
Description: 训练人脸表情识别程序
"""
import warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings(action='ignore')
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from load_and_process import load_fer2013
from load_and_process import preprocess_input
from models.cnn import mini_XCEPTION
from sklearn.model_selection import train_test_split

# 参数
batch_size = 32
num_epochs = 10000
input_shape = (48, 48, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 50
base_path = 'models/'

# 构建模型
model = mini_XCEPTION(input_shape, num_classes)
# model.compile函数(属于keras库)用来配置训练模型参数，可以指定你设想的随机梯度下降中的网络的损失函数、优化方式等参数
model.compile(optimizer='adam',  # 优化器采用adam
              loss='categorical_crossentropy',  # 多分类的对数损失函数
              metrics=['accuracy'])
model.summary()

# 定义回调函数 Callbacks 用于训练过程
#callbacks回调：通过调用CSVLogger、EarlyStopping、ReduceLROnPlateau、ModelCheckpoint等函数得到训练参数存到一个list内
log_file_path = base_path + '_emotion_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience / 4),
                              verbose=1)
# 模型位置及命名
trained_models_path = base_path + '_mini_XCEPTION'
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'

# 定义模型权重位置、命名等
model_checkpoint = ModelCheckpoint(model_names,
                                   'val_loss', verbose=1,
                                   save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

# 载入数据集
faces, emotions = load_fer2013()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape

# 划分训练、测试集
xtrain, xtest, ytrain, ytest = train_test_split(faces, emotions, test_size=0.2, shuffle=True)

# 图片产生器，在批量中对数据进行增强，扩充数据集大小
# data generator调用ImageDataGenerator函数实现实时数据增强生成小批量的图像数据。
data_generator = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=.1,
    horizontal_flip=True)

# 利用数据增强进行训练
# training model调用fit_generator函数训练模型
model.fit_generator(data_generator.flow(xtrain, ytrain, batch_size),
                    steps_per_epoch=len(xtrain) / batch_size,
                    epochs=num_epochs,
                    verbose=1, callbacks=callbacks,
                    validation_data=(xtest, ytest))

#FACS 特征点 结合
#多线程编程