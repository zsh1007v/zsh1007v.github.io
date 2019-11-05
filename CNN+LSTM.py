'''
无频域，只有时序和形态特征
'''
import glob
import sys
from time import time
import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
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

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        print('acc', self.accuracy[loss_type])
        plt.xlabel(loss_type)
        plt.ylabel('acc')
        plt.title('1DCNN+LSTM')
        plt.show()
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        print('loss', self.losses[loss_type])
        # plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.title('1DCNN+LSTM')
        plt.show()


def model_23():
    input2_ = Input(shape=(500, 16), name='input2')  # [batch, in_width, in_channels]
    x2 = Conv1D(10, kernel_size=100, strides=5, activation='relu', padding='same')(input2_)  # 100, 10
    x2 = MaxPool1D(pool_size=2)(x2)  # 50, 10
    x2 = Conv1D(16, kernel_size=50, activation='relu', padding='same')(x2)  # 50, 16
    x2 = Flatten()(x2)  # 800
    x2 = Lambda(lambda x: x * 0.5)(x2)

    input3_ = Input(shape=(16, 500), name='input3')# 第一个时间序列timesteps，第二个是特征数
    x3 = Bidirectional(LSTM(250, return_sequences=True))(input3_)
    x3 = Bidirectional(LSTM(25, return_sequences=True))(x3)
    x3 = Flatten()(x3)
    # x3 = GlobalAveragePooling1D()(x3)
    x3 = Lambda(lambda x: x * 0.5)(x3)
    # added = concatenate([x2, x3])
    added = keras.layers.Add()([x2, x3])
    added = Dropout(0.6)(added)
    # added = Dense(100, activation='relu', name='FC')(added)
    output_ = Dense(4, activation='softmax', name='output')(added)
    model = Model(inputs=[input2_, input3_], outputs=[output_])
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

    model_23 = model_23()
    lh_23 = LossHistory()
    model_23.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])
    model_23.fit([X_train_2d, X_train_2d_T], y_train_hot, batch_size=20, epochs=1, verbose=1, shuffle=True, callbacks=[lh_23])  # TensorBoard(log_dir='mytensorboard')
    score_23, acc_23 = model_23.evaluate([X_test_2d, X_test_2d_T], y_test_hot, verbose=0)
    print('Accuracy on test set:{}'.format(model_23.evaluate([X_test_2d, X_test_2d_T], y_test_hot, batch_size=20, verbose=0)))
    y_pre = model_23.predict([X_test_2d, X_test_2d_T])
    y_pre = np.argmax(y_pre, axis=1)
    num = len(y_pre)
    accuracy = len([1 for i in range(num) if y_pre[i] == y_test[i]]) / float(num)
    print('acc:', accuracy)
    print('loss_23', lh_23.losses)
    print('acc_23', lh_23.accuracy)
    lh_23.loss_plot('batch')
    # 提取FC特征
    get_feature23 = K.function([model_23.layers[0].input, model_23.layers[2].input], [model_23.layers[11].output])
    FC_train_feature23 = get_feature23([X_train_2d, X_train_2d_T])[0]
    FC_test_feature23 = get_feature23([X_test_2d, X_test_2d_T])[0]
    print(y_train[:10])
    print(FC_train_feature23.shape)
    plt.plot(FC_train_feature23[0, :])
    plt.show()
    # svc(FC_train_feature23, y_train, FC_test_feature23, y_test)
    # rf(FC_train_feature23, y_train, FC_test_feature23, y_test)
    # fc层t-sne
    import pandas as pd
    from sklearn.manifold import TSNE
    y_train = pd.DataFrame(y_train)
    FC_train_feature23 = pd.DataFrame(FC_train_feature23)
    r = pd.concat([FC_train_feature23, y_train], axis=1)
    r.columns = list(FC_train_feature23.columns) + [r'聚类类别']
    print(r)
    t0 = time()
    tsne = TSNE(n_components=2, init="pca", perplexity=800, random_state=1, learning_rate=30)
    tsne.fit_transform(FC_train_feature23.iloc[:1500, :])  # 进行数据降维,降成两维
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))  # 算法用时
    tsne = pd.DataFrame(tsne.embedding_, index=FC_train_feature23.iloc[:1500, :].index)  # 转换数据格式
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
    # 混淆矩阵
    # y_test = [np.argmax(one_hot) for one_hot in y_test]
    print('y_test', y_test)
    print('y_pre', y_pre)
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, y_pre))
    from sklearn import metrics
    print(metrics.precision_score(y_test, y_pre, average='macro'))
    print(metrics.recall_score(y_test, y_pre, average='macro'))
    print(metrics.f1_score(y_test, y_pre, average='weighted'))
    # 混淆矩阵
    print(confusion_matrix(y_test, y_pre))
    target_names = ['class 0', 'class 1', 'class 2', 'class 3']
    print(classification_report(y_test, y_pre, target_names=target_names))
    print('model_23----', 'Test accuracy:', acc_23, 'Score:', score_23)
    iters_23 = range(len(lh_23.losses['batch']))
    plt.figure()
    #acc
    plt.plot(iters_23, lh_23.accuracy['batch'], 'b', label='23_train acc')
    plt.xlabel('batch')
    plt.ylabel('acc-loss')
    plt.show()
    plt.figure()
    plt.plot(iters_23, lh_23.losses['batch'], 'b', label='23_train loss')
