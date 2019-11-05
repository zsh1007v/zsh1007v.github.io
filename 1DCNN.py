'''
无频域，只有时序和形态特征
'''
import glob
import sys
from time import time
import keras
import matplotlib.pyplot as plt
import numpy as np
from hyperas import optim
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
from hyperas.distributions import choice, uniform
from keras import Input, Model
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Dense, MaxPool2D, concatenate, Flatten, LSTM, Conv1D, MaxPool1D, Dropout, \
    BatchNormalization, K, UpSampling1D, Bidirectional, TimeDistributed, Lambda, GlobalAveragePooling1D
from keras.preprocessing import image
from keras.utils import to_categorical, plot_model
import theano,pickle
from matplotlib import cm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from keras import backend as K
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
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        print('acc', self.accuracy[loss_type])
        plt.xlabel(loss_type)
        plt.ylabel('acc')
        plt.title('1DCNN')
        plt.show()
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        print('loss', self.losses[loss_type])
        # plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.title('1DCNN')
        plt.show()


def model_02():
    input2_ = Input(shape=(500, 16), name='input2')  # [batch, in_width, in_channels]
    x2 = Conv1D(10, kernel_size=100, strides=5, activation='relu', padding='same')(input2_)  # 100, 10
    x2 = MaxPool1D(pool_size=2)(x2)  # 50, 10
    x2 = Conv1D(16, kernel_size=50, activation='relu', padding='same')(x2)  # 50, 16
    x2 = Flatten()(x2)  # 800
    x2 = Dropout(0.7)(x2)
    # x2 = Dense(400, activation='relu', name='FC')(x2)
    output_ = Dense(4, activation='softmax', name='output')(x2)
    model = Model(inputs=[input2_], outputs=[output_])
    model.summary()
    return model


def svc(traindata, trainlabel, testdata, testlabel):
    print("Start training SVM...")
    svcClf = SVC(C=0.1, kernel="rbf", cache_size=300)
    svcClf.fit(traindata, trainlabel)
    pred_testlabel = svcClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i] == pred_testlabel[i]]) / float(num)
    print("cnn-svm Accuracy:", accuracy)


def rf(traindata, trainlabel, testdata, testlabel):
    print("Start training Random Forest...")
    rfClf = RandomForestClassifier(n_estimators=100, criterion='entropy')
    rfClf.fit(traindata, trainlabel)
    pred_testlabel = rfClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i] == pred_testlabel[i]]) / float(num)
    print("cnn-rf Accuracy:", accuracy)


if __name__ == '__main__':
    X_train_2d = np.load(file="X_train_2d_0.5.npy")
    X_test_2d = np.load(file="X_test_2d_0.5.npy")
    X_train_1d = np.load(file="X_train_1d_0.5.npy")
    X_test_1d = np.load(file="X_test_1d_0.5.npy")
    y_train = np.load(file="y_train_0.5.npy") - 1
    y_test = np.load(file="y_test_0.5.npy") - 1
    X_train_2d_T = X_train_2d.transpose(0, 2, 1)
    X_test_2d_T = X_test_2d.transpose(0, 2, 1)
    # X_test_1d_T = X_test_1d.reshape(1600, 8000, 1)
    # X_train_1d_T = X_train_1d.reshape(2400, 8000, 1)
    print(X_train_2d.shape)# 500,16
    print(X_test_2d.shape)

    # one-hot类型
    y_train_hot,  y_test_hot = to_categorical(y_train, 4), to_categorical(y_test, 4)
    # X_train_pic = list_pic('D:/PycharmProjects/untitled/img_train/X_train_1d' + '*.png')
    # X_test_pic = list_pic('D:/PycharmProjects/untitled/img_test/X_test_1d' + '*.png')
    # X_val_pic = list_pic('D:/PycharmProjects/untitled/img_val/X_val_1d' + '*.png')

    model_02 = model_02()
    lh_2 = LossHistory()
    model_02.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])
    model_02.fit(X_train_2d, y_train_hot, batch_size=20, epochs=1, verbose=1, shuffle=True, callbacks=[lh_2])  # TensorBoard(log_dir='mytensorboard')
    score_02, acc_02 = model_02.evaluate(X_test_2d, y_test_hot)
    print('model_02----', 'Test accuracy:', acc_02, 'Score:', score_02)
    print('loss_2', lh_2.losses)
    print('acc_2', lh_2.accuracy)
    lh_2.loss_plot('batch')
    # 提取cnn特征
    get_feature2 = K.function([model_02.layers[0].input], [model_02.layers[3].output])
    FC_train_feature2 = get_feature2([X_test_2d])[0]
    print(FC_train_feature2.shape)
    # FC_train_feature2 = FC_train_feature2.transpose(0, 2, 1)
    for i in range(16):
        i = i + 1
        plt.subplot(4, 4, i)
        i = i - 1
        plt.plot(FC_train_feature2[0, :, i])
    plt.suptitle('TYPE3')
    plt.show()
    plt.plot(FC_train_feature2[0, :, :])
    plt.suptitle('TYPE3')
    plt.show()

    for i in range(16):
        i = i + 1
        plt.subplot(4, 4, i)
        i = i - 1
        plt.plot(FC_train_feature2[1, :, i])
    plt.suptitle('TYPE2')
    plt.show()
    plt.plot(FC_train_feature2[1, :, :])
    plt.suptitle('TYPE2')
    plt.show()

    for i in range(16):
        i = i + 1
        plt.subplot(4, 4, i)
        i = i - 1
        plt.plot(FC_train_feature2[2, :, i])
    plt.suptitle('TYPE4')
    plt.show()
    plt.plot(FC_train_feature2[2, :, :])
    plt.suptitle('TYPE4')
    plt.show()

    for i in range(16):
        i = i + 1
        plt.subplot(4, 4, i)
        i = i - 1
        plt.plot(FC_train_feature2[6, :, i])
    plt.suptitle('TYPE1')
    plt.show()
    plt.plot(FC_train_feature2[6, :, :])
    plt.suptitle('TYPE1')
    plt.show()
    # FC
    get_feature2_svm = K.function([model_02.layers[0].input], [model_02.layers[4].output])
    FC_test_feature2_svm = get_feature2_svm([X_test_2d])[0]
    FC_train_feature2_svm = get_feature2_svm([X_train_2d])[0]
    svc(FC_train_feature2_svm, y_train, FC_test_feature2_svm, y_test)
    rf(FC_train_feature2_svm, y_train, FC_test_feature2_svm, y_test)
    # fc层t-sne
    import pandas as pd
    from sklearn.manifold import TSNE
    y_train = pd.DataFrame(y_train)
    FC_train_feature2_svm = pd.DataFrame(FC_train_feature2_svm)
    r = pd.concat([FC_train_feature2_svm, y_train], axis=1)
    r.columns = list(FC_train_feature2_svm.columns) + [r'聚类类别']
    print(r)
    t0 = time()
    tsne = TSNE(n_components=2, init="pca", perplexity=400, random_state=1, learning_rate=30)
    tsne.fit_transform(FC_train_feature2_svm.iloc[:1000, :])  # 进行数据降维,降成两维
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))  # 算法用时
    tsne = pd.DataFrame(tsne.embedding_, index=FC_train_feature2_svm.iloc[:1000, :].index)  # 转换数据格式
    d = tsne[r[r'聚类类别'] == 0]
    plt.scatter(d[0], d[1], cmap=plt.cm.Spectral)  # .
    d = tsne[r[r'聚类类别'] == 1]
    # plt.plot(d[0],d[1],'go')#o
    plt.scatter(d[0], d[1], cmap=plt.cm.Spectral)  # .
    d = tsne[r[r'聚类类别'] == 2]
    # plt.plot(d[0],d[1],'b*')#*
    plt.scatter(d[0], d[1], cmap=plt.cm.Spectral)  # .
    d = tsne[r[r'聚类类别'] == 3]
    # plt.plot(d[0], d[1],'y+')#*
    plt.scatter(d[0], d[1], cmap=plt.cm.Spectral)  # .
    plt.xticks([])
    plt.yticks([])
    plt.legend(['TYPE1', 'TYPE2', 'TYPE3', 'TYPE4'])
    plt.title("t-SNE (%.2g sec)" % (t1 - t0))
    plt.show()
    #
    y_pre = model_02.predict([X_test_2d])
    y_pre = np.argmax(y_pre, axis=1)
    print('y_test', y_test)
    print('y_pre', y_pre)
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, y_pre))
    from sklearn import metrics
    print(metrics.precision_score(y_test, y_pre, average='macro'))
    print(metrics.recall_score(y_test, y_pre, average='macro'))
    print(metrics.f1_score(y_test, y_pre, average='weighted'))
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test, y_pre))
    from sklearn.metrics import classification_report
    target_names = ['class 0', 'class 1', 'class 2', 'class 3']
    print(classification_report(y_test, y_pre, target_names=target_names))

    print('model_02----', 'Test accuracy:', acc_02, 'Score:', score_02)

