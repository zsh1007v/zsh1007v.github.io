'''
频域特征提取梯度消失！！！
'''


import glob
import sys
import time
import keras
import matplotlib.pyplot as plt
import numpy as np
from hyperas import optim
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
from hyperas.distributions import choice, uniform
from keras import Input, Model
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Dense, MaxPool2D, concatenate, Flatten, LSTM, Conv1D, MaxPool1D, Dropout, \
    BatchNormalization
from keras.preprocessing import image
from keras.utils import to_categorical


# 写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

        # 按照batch来进行追加数据

    def on_batch_end(self, batch, logs={}):
        # 每一个batch完成后向容器里面追加loss，acc
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
        # 每五秒按照当前容器里的值来绘图
        if int(time.time()) % 0.001 == 0:
            self.draw_p(self.losses['batch'], 'loss', 'train_batch')
            self.draw_p(self.accuracy['batch'], 'acc', 'train_batch')
            self.draw_p(self.val_loss['batch'], 'loss', 'val_batch')
            self.draw_p(self.val_acc['batch'], 'acc', 'val_batch')

    def on_epoch_end(self, batch, logs={}):
        # 每一个epoch完成后向容器里面追加loss，acc
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        # 每五秒按照当前容器里的值来绘图
        if int(time.time()) % 0.001 == 0:
            self.draw_p(self.losses['epoch'], 'loss', 'train_epoch')
            self.draw_p(self.accuracy['epoch'], 'acc', 'train_epoch')
            self.draw_p(self.val_loss['epoch'], 'loss', 'val_epoch')
            self.draw_p(self.val_acc['epoch'], 'acc', 'val_epoch')

    # 绘图，这里把每一种曲线都单独绘图，若想把各种曲线绘制在一张图上的话可修改此方法
    def draw_p(self, lists, label, type):
        plt.figure()
        plt.plot(range(len(lists)), lists, 'r', label=label)
        plt.ylabel(label)
        plt.xlabel(type)
        plt.legend(loc="upper right")
        plt.savefig(type + '_' + label + '.jpg')

    # 由于这里的绘图设置的是5s绘制一次，当训练结束后得到的图可能不是一个完整的训练过程（最后一次绘图结束，有训练了0-5秒的时间）
    # 所以这里的方法会在整个训练结束以后调用
    def end_draw(self):
        self.draw_p(self.losses['batch'], 'loss', 'train_batch')
        self.draw_p(self.accuracy['batch'], 'acc', 'train_batch')
        self.draw_p(self.val_loss['batch'], 'loss', 'val_batch')
        self.draw_p(self.val_acc['batch'], 'acc', 'val_batch')
        self.draw_p(self.losses['epoch'], 'loss', 'train_epoch')
        self.draw_p(self.accuracy['epoch'], 'acc', 'train_epoch')
        self.draw_p(self.val_loss['epoch'], 'loss', 'val_epoch')
        self.draw_p(self.val_acc['epoch'], 'acc', 'val_epoch')


def list_pic(path):
    pic_names = glob.glob(path)
    img = []
    # 把图片读取出来放到列表中
    for i in range(len(pic_names)):
        images = image.load_img(pic_names[i])
        x = image.img_to_array(images)
        x = np.expand_dims(x, axis=0)
        img.append(x)
        # print('loading no.%s val image' % i)
    # 把图片数组联合在一起
    pic_list = np.concatenate([x for x in img])
    pic_list = pic_list / 255
    # plt.imshow(pic_list[1])
    # plt.show()
    return pic_list


def model_3():
    input1_ = Input(shape=(240, 320, 3), name='input1')
    x1 = Conv2D(16, (4, 4), activation='relu', padding='same')(input1_)
    x1 = MaxPool2D(pool_size=(4, 4))(x1)
    x1 = Conv2D(32, (2, 4), activation='relu', padding='same')(x1)
    x1 = MaxPool2D(pool_size=(2, 4))(x1)
    # x1 = Dense(16, activation='relu')(x1)
    x1 = Flatten()(x1)

    input2_ = Input(shape=(500, 16), name='input2')
    x2 = Conv1D(20, kernel_size=5, activation='relu', padding='same')(input2_)
    x2 = MaxPool1D(pool_size=3)(x2)
    x2 = Conv1D(25, kernel_size=3, activation='relu', padding='same')(x2)
    x2 = MaxPool1D(pool_size=3)(x2)
    # x2 = Dense(16, activation='relu')(x2)
    x2 = Flatten()(x2)

    input3_ = Input(shape=(500, 16), name='input3')
    x3 = LSTM(300, return_sequences=True)(input3_)
    # x3 = Dense(16, activation='relu')(x3)
    x3 = Flatten()(x3)

    x = concatenate([x1, x2, x3])
    output_ = Dense(4, activation='softmax', name='output')(x)
    model = Model(inputs=[input1_, input2_, input3_], outputs=[output_])
    model.summary()
    return model


def model_12():
    input1_ = Input(shape=(240, 320, 3), name='input1')
    x1 = Conv2D(16, (4, 4), activation='relu', padding='same')(input1_)
    x1 = MaxPool2D(pool_size=(4, 4))(x1)
    x1 = Conv2D(32, (2, 4), activation='relu', padding='same')(x1)
    x1 = MaxPool2D(pool_size=(2, 4))(x1)
    # x1 = Dense(16, activation='relu')(x1)
    x1 = Flatten()(x1)

    input2_ = Input(shape=(500, 16), name='input2')
    x2 = Conv1D(20, kernel_size=5, activation='relu', padding='same')(input2_)
    x2 = MaxPool1D(pool_size=3)(x2)
    x2 = Conv1D(25, kernel_size=3, activation='relu', padding='same')(x2)
    x2 = MaxPool1D(pool_size=3)(x2)
    # x2 = Dense(16, activation='relu')(x2)
    x2 = Flatten()(x2)

    x = concatenate([x1, x2])
    output_ = Dense(4, activation='softmax', name='output')(x)
    model = Model(inputs=[input1_, input2_], outputs=[output_])
    model.summary()
    return model


def model_13():
    input1_ = Input(shape=(240, 320, 3), name='input1')
    x1 = Conv2D(16, (4, 4), activation='relu', padding='same')(input1_)
    x1 = MaxPool2D(pool_size=(4, 4))(x1)
    x1 = Conv2D(32, (2, 4), activation='relu', padding='same')(x1)
    x1 = MaxPool2D(pool_size=(2, 4))(x1)
    # x1 = Dense(16, activation='relu')(x1)
    x1 = Flatten()(x1)

    input3_ = Input(shape=(500, 16), name='input3')
    x3 = LSTM(400, return_sequences=True)(input3_)
    # x3 = Dense(16, activation='relu')(x3)
    x3 = Flatten()(x3)

    x = concatenate([x1, x3])
    output_ = Dense(4, activation='softmax', name='output')(x)
    model = Model(inputs=[input1_, input3_], outputs=[output_])
    model.summary()
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])
    return model


def model_23():
    input2_ = Input(shape=(500, 16), name='input2')
    x2 = Conv1D(20, kernel_size=5, activation='relu', padding='same')(input2_)
    x2 = MaxPool1D(pool_size=3)(x2)
    x2 = Conv1D(25, kernel_size=3, activation='relu', padding='same')(x2)
    x2 = MaxPool1D(pool_size=3)(x2)
    # x2 = Dense(16, activation='relu')(x2)
    x2 = Flatten()(x2)

    input3_ = Input(shape=(500, 16), name='input3')
    x3 = LSTM(400, return_sequences=True)(input3_)
    # x3 = Dense(16, activation='relu')(x3)
    x3 = Flatten()(x3)

    x = concatenate([x2, x3])
    output_ = Dense(4, activation='softmax', name='output')(x)
    model = Model(inputs=[input2_, input3_], outputs=[output_])
    model.summary()
    return model


def model_01():
    input1_ = Input(shape=(240, 320, 3), name='input1')
    x1 = Conv2D(10, (30, 20), activation='tanh', padding='same')(input1_)
    x1 = MaxPool2D(pool_size=(20, 8))(x1)#12*16
    x1 = Conv2D(15, (10, 2), activation='tanh', padding='same')(x1)
    x1 = MaxPool2D(pool_size=(6, 4))(x1)# 2*4
    x1 = Dense(128)(x1)
    x1 = Flatten()(x1)
    output_ = Dense(4, activation='softmax', name='output')(x1)
    model = Model(inputs=[input1_], outputs=[output_])
    model.summary()
    return model


def model_02():
    input2_ = Input(shape=(500, 16), name='input2')
    x2 = Conv1D(20, kernel_size=5, activation='relu', padding='same')(input2_)
    x2 = MaxPool1D(pool_size=3)(x2)
    x2 = Conv1D(25, kernel_size=3, activation='relu', padding='same')(x2)
    x2 = MaxPool1D(pool_size=3)(x2)
    # x2 = Dense(16, activation='relu')(x2)
    x2 = Flatten()(x2)
    output_ = Dense(4, activation='softmax', name='output')(x2)
    model = Model(inputs=[input2_], outputs=[output_])
    model.summary()
    return model


def model_03():
    input3_ = Input(shape=(500, 16), name='input3')
    x3 = LSTM(300, return_sequences=True)(input3_)
    # x3 = Dense(16, activation='relu')(x3)
    x3 = Flatten()(x3)
    output_ = Dense(4, activation='softmax', name='output')(x3)
    model = Model(inputs=[input3_], outputs=[output_])
    model.summary()
    return model


if __name__ == '__main__':
    X_train_2d = np.load(file="X_train_2d_0.5.npy")
    X_test_2d = np.load(file="X_test_2d_0.5.npy")
    X_val_2d = np.load(file="X_val_2d_0.5.npy")
    X_train_1d = np.load(file="X_train_1d_0.5.npy")
    X_test_1d = np.load(file="X_test_1d_0.5.npy")
    X_val_1d = np.load(file="X_val_1d_0.5.npy")
    y_train = np.load(file="y_train_0.5.npy") - 1
    y_test = np.load(file="y_test_0.5.npy") - 1
    y_val = np.load(file="y_val_0.5.npy") - 1
    # one-hot类型
    y_train, y_val, y_test = to_categorical(y_train, 4), to_categorical(y_val, 4), to_categorical(y_test, 4)
    X_train_pic = list_pic('D:/PycharmProjects/untitled/img_train/X_train_1d' + '*.png')
    X_test_pic = list_pic('D:/PycharmProjects/untitled/img_test/X_test_1d' + '*.png')
    X_val_pic = list_pic('D:/PycharmProjects/untitled/img_val/X_val_1d' + '*.png')

    # losshistory = LossHistory()
    # model_3 = model_3()
    # model_3.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])
    # model_3.fit([X_train_pic, X_train_2d, X_train_2d], y_train, epochs=1,
    #             batch_size=20, validation_data=([X_val_pic, X_val_2d, X_val_2d], y_val),
    #             verbose=1, shuffle=True, callbacks=[]) # TensorBoard(log_dir='mytensorboard')
    # score_3, acc_3 = model_3.evaluate([X_test_pic, X_test_2d, X_test_2d], y_test, verbose=0)
    #
    #
    # model_12 = model_12()
    # model_12.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])
    # model_12.fit([X_train_pic, X_train_2d], y_train, epochs=1,
    #             batch_size=20, validation_data=([X_val_pic, X_val_2d], y_val),
    #             verbose=1, shuffle=True, callbacks=[])  # TensorBoard(log_dir='mytensorboard')
    # score_12, acc_12 = model_12.evaluate([X_test_pic, X_test_2d], y_test, verbose=0)
    #
    #
    # model_13 = model_13()
    # model_13.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])
    # model_13.fit([X_train_pic, X_train_2d], y_train, epochs=1,
    #              batch_size=20, validation_data=([X_val_pic, X_val_2d], y_val),
    #              verbose=1, shuffle=True, callbacks=[])  # TensorBoard(log_dir='mytensorboard')
    # score_13, acc_13 = model_13.evaluate([X_test_pic, X_test_2d], y_test, verbose=0)
    #
    #
    # model_23 = model_23()
    # model_23.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])
    # model_23.fit([X_train_2d, X_train_2d], y_train, epochs=1,
    #              batch_size=20, validation_data=([X_val_2d, X_val_2d], y_val),
    #              verbose=1, shuffle=True, callbacks=[])  # TensorBoard(log_dir='mytensorboard')
    # score_23, acc_23 = model_23.evaluate([X_test_2d, X_test_2d], y_test, verbose=0)


    model_01 = model_01()
    model_01.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])
    from keras.callbacks import ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.01, patience=1, mode='auto')
    model_01.fit(X_train_pic, y_train, epochs=2,
                 batch_size=20, validation_data=(X_val_pic, y_val),
                 verbose=1, shuffle=True, callbacks=[reduce_lr])  # TensorBoard(log_dir='mytensorboard')
    score_01, acc_01 = model_01.evaluate(X_test_pic, y_test, verbose=0)


    # model_02 = model_02()
    # model_02.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])
    # model_02.fit(X_train_2d, y_train, epochs=1,
    #              batch_size=20, validation_data=(X_val_2d, y_val),
    #              verbose=1, shuffle=True, callbacks=[])  # TensorBoard(log_dir='mytensorboard')
    # score_02, acc_02 = model_02.evaluate(X_test_2d, y_test, verbose=0)
    #
    #
    # model_03 = model_03()
    # model_03.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])
    # model_03.fit(X_train_2d, y_train, epochs=1,
    #              batch_size=20, validation_data=(X_val_2d, y_val),
    #              verbose=1, shuffle=True, callbacks=[])  # TensorBoard(log_dir='mytensorboard')
    # score_03, acc_03 = model_03.evaluate(X_test_2d, y_test, verbose=0)
    # print('model_3----', 'Test accuracy:', acc_3, 'Score:', score_3)
    # print('model_12----', 'Test accuracy:', acc_12, 'Score:', score_12)
    # print('model_13----', 'Test accuracy:', acc_13, 'Score:', score_13)
    # print('model_23----', 'Test accuracy:', acc_23, 'Score:', score_23)
    print('model_01----', 'Test accuracy:', acc_01, 'Score:', score_01)
    # print('model_02----', 'Test accuracy:', acc_02, 'Score:', score_02)
    # print('model_03----', 'Test accuracy:', acc_03, 'Score:', score_03)

