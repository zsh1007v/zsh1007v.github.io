'''
无频域，只有时序和形态特征
'''
from time import time
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import Input, Model
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPool1D, Dropout, Bidirectional, Lambda
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from keras import backend as K
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

# 写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        # 按照batch来进行追加数据

    def on_batch_end(self, batch, logs={}):
        # 每一个batch完成后向容器里面追加loss，acc
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))


    def on_epoch_end(self, epoch, logs={}):
        # 每一个epoch完成后向容器里面追加loss，acc
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r')
        print('Accuracy', self.accuracy[loss_type])
        plt.xlabel(loss_type)
        plt.ylabel('Accuracy')
        plt.show()
        # loss
        plt.plot(iters, self.losses[loss_type], 'g')
        print('loss', self.losses[loss_type])
        # plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.show()


def model_23():
    input2_ = Input(shape=(500, 16), name='input2')  # [batch, in_width, in_channels]
    x2 = Conv1D(10, kernel_size=100, strides=5, activation='relu', padding='same')(input2_)  # 100, 10
    x2 = MaxPool1D(pool_size=2)(x2)  # 50, 10
    x2 = Conv1D(16, kernel_size=50, activation='relu', padding='same')(x2)  # 50, 16
    x2 = Flatten()(x2)  # 800
    x2 = Lambda(lambda x: x * 0.5)(x2)

    input3_ = Input(shape=(16, 500), name='input3')# 第一个时间序列timesteps，第二个是特征数
    # x3 = Bidirectional(LSTM(250, return_sequences=True))(input3_)
    x3 = Bidirectional(LSTM(25, return_sequences=True))(input3_)
    x3 = Flatten()(x3)
    x3 = Lambda(lambda x: x * 0.5)(x3)
    # added = concatenate([x2, x3])
    added = keras.layers.Add()([x2, x3])
    added = Dropout(0.6)(added)
    output_ = Dense(4, activation='softmax', name='output')(added)
    model = Model(inputs=[input2_, input3_], outputs=[output_])
    model.summary()
    return model

if __name__ == '__main__':
    X_train_2d = np.load(file="X_train_2d_0.5.npy")
    X_test_2d = np.load(file="X_test_2d_0.5.npy")
    X_train_1d = np.load(file="X_train_1d_0.5.npy")
    X_test_1d = np.load(file="X_test_1d_0.5.npy")
    y_train = np.load(file="y_train_0.5.npy") - 1
    y_test = np.load(file="y_test_0.5.npy") - 1
    X_train_2d_T = X_train_2d.transpose(0, 2, 1)
    X_test_2d_T = X_test_2d.transpose(0, 2, 1)
    print(X_train_2d.shape)# 500,16
    print(X_test_2d.shape)
    y_train_hot,  y_test_hot = to_categorical(y_train, 4), to_categorical(y_test, 4)


    model_adam = model_23()
    lh_adam = LossHistory()
    model_adam.compile(optimizer=Adam(lr=0.0007), loss='categorical_crossentropy', metrics=['acc']) #Adam((lr=0.0007))
    model_adam.fit([X_train_2d, X_train_2d_T], y_train_hot, batch_size=40, epochs=1, verbose=1, shuffle=True, callbacks=[lh_adam])
    score_adam, acc_adam = model_adam.evaluate([X_test_2d, X_test_2d_T], y_test_hot, verbose=0)
    y_pre = model_adam.predict([X_test_2d, X_test_2d_T])
    y_pre = np.argmax(y_pre, axis=1)
    print('acc', accuracy_score(y_test, y_pre))
    print('precision_score', metrics.precision_score(y_test, y_pre, average='macro'))
    print('recall_score', metrics.recall_score(y_test, y_pre, average='macro'))
    print('f1_score', metrics.f1_score(y_test, y_pre, average='weighted'))
    lh_adam.loss_plot('batch')

    model_sgd = model_23()
    lh_sgd = LossHistory()
    model_sgd.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])  # Adam((lr=0.0007))
    model_sgd.fit([X_train_2d, X_train_2d_T], y_train_hot, batch_size=40, epochs=1, verbose=1, shuffle=True,
                   callbacks=[lh_sgd])
    score_sgd, acc_sgd = model_sgd.evaluate([X_test_2d, X_test_2d_T], y_test_hot, verbose=0)
    y_pre = model_sgd.predict([X_test_2d, X_test_2d_T])
    y_pre = np.argmax(y_pre, axis=1)
    print('acc', accuracy_score(y_test, y_pre))
    print('precision_score', metrics.precision_score(y_test, y_pre, average='macro'))
    print('recall_score', metrics.recall_score(y_test, y_pre, average='macro'))
    print('f1_score', metrics.f1_score(y_test, y_pre, average='weighted'))
    lh_sgd.loss_plot('batch')

    model_RMSprop = model_23()
    lh_RMSprop = LossHistory()
    model_RMSprop.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['acc'])  # Adam((lr=0.0007))
    model_RMSprop.fit([X_train_2d, X_train_2d_T], y_train_hot, batch_size=40, epochs=1, verbose=1, shuffle=True,
                  callbacks=[lh_RMSprop])
    score_RMSprop, acc_RMSprop = model_RMSprop.evaluate([X_test_2d, X_test_2d_T], y_test_hot, verbose=0)
    y_pre = model_RMSprop.predict([X_test_2d, X_test_2d_T])
    y_pre = np.argmax(y_pre, axis=1)
    print('acc', accuracy_score(y_test, y_pre))
    print('precision_score', metrics.precision_score(y_test, y_pre, average='macro'))
    print('recall_score', metrics.recall_score(y_test, y_pre, average='macro'))
    print('f1_score', metrics.f1_score(y_test, y_pre, average='weighted'))
    lh_RMSprop.loss_plot('batch')

    model_AdaGrad = model_23()
    lh_AdaGrad = LossHistory()
    model_AdaGrad.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['acc'])  # Adam((lr=0.0007))
    model_AdaGrad.fit([X_train_2d, X_train_2d_T], y_train_hot, batch_size=40, epochs=1, verbose=1, shuffle=True,
                      callbacks=[lh_AdaGrad])
    score_AdaGrad, acc_AdaGrad = model_AdaGrad.evaluate([X_test_2d, X_test_2d_T], y_test_hot, verbose=0)
    y_pre = model_AdaGrad.predict([X_test_2d, X_test_2d_T])
    y_pre = np.argmax(y_pre, axis=1)
    print('acc', accuracy_score(y_test, y_pre))
    print('precision_score', metrics.precision_score(y_test, y_pre, average='macro'))
    print('recall_score', metrics.recall_score(y_test, y_pre, average='macro'))
    print('f1_score', metrics.f1_score(y_test, y_pre, average='weighted'))
    lh_AdaGrad.loss_plot('batch')


    print('model_23----', 'Test accuracy:', acc_adam, 'Score:', score_adam)
    print('model_02----', 'Test accuracy:', acc_sgd, 'Score:', score_sgd)
    print('model_03----', 'Test accuracy:', acc_RMSprop, 'Score:', score_RMSprop)
    print('model_01----', 'Test accuracy:', acc_AdaGrad, 'Score:', score_AdaGrad)

    iters_AdaGrad = range(len(lh_AdaGrad.losses['batch']))
    iters_RMSprop = range(len(lh_RMSprop.losses['batch']))
    iters_sgd = range(len(lh_sgd.losses['batch']))
    iters_adam = range(len(lh_adam.losses['batch']))

    plt.figure()
    plt.plot(iters_AdaGrad, lh_AdaGrad.accuracy['batch'], 'darkorange', label='AdaGrad')
    plt.plot(iters_RMSprop, lh_RMSprop.accuracy['batch'], 'r', label='RMSprop')
    plt.plot(iters_sgd, lh_sgd.accuracy['batch'], 'g', label='SGD')
    plt.plot(iters_adam, lh_adam.accuracy['batch'], 'b', label='Adam')
    plt.xlabel('batch')
    plt.ylabel('acc')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(iters_AdaGrad, lh_AdaGrad.losses['batch'], 'darkorange', label='AdaGrad')
    plt.plot(iters_RMSprop, lh_RMSprop.losses['batch'], 'r', label='RMSprop')
    plt.plot(iters_sgd, lh_sgd.losses['batch'], 'g', label='SGD')
    plt.plot(iters_adam, lh_adam.losses['batch'], 'b', label='Adam')
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
