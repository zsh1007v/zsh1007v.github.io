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


def model_02():
    input2_ = Input(shape=(500, 16), name='input2')  # [batch, in_width, in_channels]
    x2 = Conv1D(10, kernel_size=100, strides=5, activation='relu', padding='same')(input2_)  # 100, 10
    x2 = MaxPool1D(pool_size=2)(x2)  # 50, 10
    x2 = Conv1D(16, kernel_size=50, activation='relu', padding='same')(x2)  # 50, 16
    x2 = Flatten()(x2)  # 800
    x2 = Dropout(0.7)(x2)
    output_ = Dense(4, activation='softmax', name='output')(x2)
    model = Model(inputs=[input2_], outputs=[output_])
    model.summary()
    return model


def model_03():
    input3_ = Input(shape=(16, 500), name='input3')  # 第一个时间序列timesteps，第二个是特征数
    # x3 = Bidirectional(LSTM(250, return_sequences=True))(input3_)
    x3 = Bidirectional(LSTM(25, return_sequences=True))(input3_)# 16, 20
    x3 = Flatten()(x3)#320
    x3 = Dropout(0.7)(x3)
    output_ = Dense(4, activation='softmax', name='output')(x3)
    model = Model(inputs=[input3_], outputs=[output_])
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
    print('acc', accuracy_score(testlabel, pred_testlabel))
    print('precision_score', metrics.precision_score(testlabel, pred_testlabel, average='macro'))
    print('recall_score', metrics.recall_score(testlabel, pred_testlabel, average='macro'))
    print('f1_score', metrics.f1_score(testlabel, pred_testlabel, average='weighted'))


def rf(traindata, trainlabel, testdata, testlabel):
    print("Start training Random Forest...")
    rfClf = RandomForestClassifier(n_estimators=200, criterion='entropy')
    rfClf.fit(traindata, trainlabel)
    pred_testlabel = rfClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i] == pred_testlabel[i]]) / float(num)
    print("cnn-rf Accuracy:", accuracy)
    print('acc', accuracy_score(testlabel, pred_testlabel))
    print('precision_score', metrics.precision_score(testlabel, pred_testlabel, average='macro'))
    print('recall_score', metrics.recall_score(testlabel, pred_testlabel, average='macro'))
    print('f1_score', metrics.f1_score(testlabel, pred_testlabel, average='weighted'))


def knn(traindata, trainlabel, testdata, testlabel):
    print("Start training KNN...")
    knn = KNeighborsClassifier(n_neighbors=5, p=2,leaf_size=30,weights='distance', algorithm='auto')
    knn.fit(traindata, trainlabel)
    pred_testlabel = knn.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i] == pred_testlabel[i]]) / float(num)
    print("cnn-knn Accuracy:", accuracy)
    print('acc', accuracy_score(testlabel, pred_testlabel))
    print('precision_score', metrics.precision_score(testlabel, pred_testlabel, average='macro'))
    print('recall_score', metrics.recall_score(testlabel, pred_testlabel, average='macro'))
    print('f1_score', metrics.f1_score(testlabel, pred_testlabel, average='weighted'))


def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocation = np.array(range(len(labels)))
    plt.xticks(xlocation, labels, rotation=90)
    plt.yticks(xlocation, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



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

    labels = ['TYPE1', 'TYPE2', 'TYPE3', 'TYPE4']
    tick_marks = np.array(range(len(labels))) + 0.5

    # one-hot类型
    y_train_hot,  y_test_hot = to_categorical(y_train, 4), to_categorical(y_test, 4)
    # X_train_pic = list_pic('D:/PycharmProjects/untitled/img_train/X_train_1d' + '*.png')
    # X_test_pic = list_pic('D:/PycharmProjects/untitled/img_test/X_test_1d' + '*.png')
    # X_val_pic = list_pic('D:/PycharmProjects/untitled/img_val/X_val_1d' + '*.png')

    model_23 = model_23()
    lh_23 = LossHistory()
    model_23.compile(optimizer=Adam(lr=0.0007), loss='categorical_crossentropy', metrics=['acc']) #Adam((lr=0.0007))
    model_23.fit([X_train_2d, X_train_2d_T], y_train_hot, batch_size=40, epochs=1, verbose=1, shuffle=True, callbacks=[lh_23])

    score_23, acc_23 = model_23.evaluate([X_test_2d, X_test_2d_T], y_test_hot, verbose=0)
    y_pre = model_23.predict([X_test_2d, X_test_2d_T])
    y_pre = np.argmax(y_pre, axis=1)
    num = len(y_pre)
    accuracy = len([1 for i in range(num) if y_pre[i] == y_test[i]]) / float(num)
    print('loss_23', lh_23.losses)
    print('acc_23', lh_23.accuracy)
    lh_23.loss_plot('batch')
    print('acc', accuracy_score(y_test, y_pre))
    print('precision_score', metrics.precision_score(y_test, y_pre, average='macro'))
    print('recall_score', metrics.recall_score(y_test, y_pre, average='macro'))
    print('f1_score', metrics.f1_score(y_test, y_pre, average='weighted'))
    print('混淆矩阵', confusion_matrix(y_test, y_pre))
    target_names = ['class 0', 'class 1', 'class 2', 'class 3']
    print(classification_report(y_test, y_pre, target_names=target_names))
    cm = confusion_matrix(y_test, y_pre)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='white', fontsize=15, va='center', ha='center')
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    plt.show()

    # ##################################################  提取FC特征 ###################################################
    get_feature23_FC = K.function([model_23.layers[0].input, model_23.layers[3].input], [model_23.layers[10].output])#CNN3 LSRM4
    FC_train_feature23_FC = get_feature23_FC([X_train_2d, X_train_2d_T])[0]
    FC_test_feature23_FC = get_feature23_FC([X_test_2d, X_test_2d_T])[0]
    # SCM RF
    # svc(FC_train_feature23_FC, y_train, FC_test_feature23_FC, y_test)
    # rf(FC_train_feature23_FC, y_train, FC_test_feature23_FC, y_test)
    # knn(FC_train_feature23_FC, y_train, FC_test_feature23_FC, y_test)
    # fc层t-sne
    y_train = pd.DataFrame(y_train)
    FC_train_feature23_FC = pd.DataFrame(FC_train_feature23_FC)
    r = pd.concat([FC_train_feature23_FC, y_train], axis=1)
    r.columns = list(FC_train_feature23_FC.columns) + [r'聚类类别']
    t0 = time()
    tsne = TSNE(n_components=2, init="pca", perplexity=500, random_state=1, learning_rate=50)
    tsne.fit_transform(FC_train_feature23_FC.iloc[:1000, :])  # 进行数据降维,降成两维
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))  # 算法用时
    tsne = pd.DataFrame(tsne.embedding_, index=FC_train_feature23_FC.iloc[:1000, :].index)  # 转换数据格式
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
    plt.title("FC t-SNE (%.2g sec)" % (t1 - t0))
    plt.show()

    #################################################输入聚类######################################################3
    FC_train_feature23_In = np.reshape(X_train_2d, (-1, 8000))
    FC_test_feature23_In = np.reshape(X_test_2d, (-1, 8000))
    # SCM RF knn
    # svc(FC_train_feature23_In, y_train, FC_test_feature23_In, y_test)
    # rf(FC_train_feature23_In, y_train, FC_test_feature23_In, y_test)
    # knn(FC_train_feature23_In, y_train, FC_test_feature23_In, y_test)
    # Input层t-sne
    y_train = pd.DataFrame(y_train)
    FC_train_feature23_In = pd.DataFrame(FC_train_feature23_In)
    r = pd.concat([FC_train_feature23_In, y_train], axis=1)
    r.columns = list(FC_train_feature23_In.columns) + [r'聚类类别']
    t0 = time()
    tsne = TSNE(n_components=2, init="pca", perplexity=500, random_state=1, learning_rate=50)
    tsne.fit_transform(FC_train_feature23_In.iloc[:1000, :])  # 进行数据降维,降成两维
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))  # 算法用时
    tsne = pd.DataFrame(tsne.embedding_, index=FC_train_feature23_In.iloc[:1000, :].index)  # 转换数据格式
    d = tsne[r[r'聚类类别'] == 1]
    plt.scatter(d[0], d[1], cmap=plt.cm.Spectral)  # .
    d = tsne[r[r'聚类类别'] == 2]
    # plt.plot(d[0],d[1],'go')#o
    plt.scatter(d[0], d[1], cmap=plt.cm.Spectral)  # .
    d = tsne[r[r'聚类类别'] == 3]
    # plt.plot(d[0],d[1],'b*')#*
    plt.scatter(d[0], d[1], cmap=plt.cm.Spectral)  # .
    d = tsne[r[r'聚类类别'] == 4]
    # plt.plot(d[0], d[1],'y+')#*
    plt.scatter(d[0], d[1], cmap=plt.cm.Spectral)  # .
    plt.xticks([])
    plt.yticks([])
    plt.legend(['TYPE1', 'TYPE2', 'TYPE3', 'TYPE4'])
    plt.title("Input t-SNE (%.2g sec)" % (t1 - t0))
    plt.show()

    # #############################################提取CNN特征####################################################
    get_feature23_CNN = K.function([model_23.layers[0].input, model_23.layers[3].input], [model_23.layers[4].output])  # CNN3 LSRM4
    FC_train_feature23_CNN = get_feature23_CNN([X_train_2d, X_train_2d_T])[0]
    FC_test_feature23_CNN = get_feature23_CNN([X_test_2d, X_test_2d_T])[0]
    for i in range(16):
        i = i + 1
        plt.subplot(4, 4, i)
        i = i - 1
        plt.plot(FC_train_feature23_CNN[0, :, i])
        plt.xticks([])
        plt.yticks([])
    plt.suptitle('TYPE3')
    plt.show()

    for i in range(16):
        i = i + 1
        plt.subplot(4, 4, i)
        i = i - 1
        plt.plot(FC_train_feature23_CNN[1, :, i])
        plt.xticks([])
        plt.yticks([])
    plt.suptitle('TYPE2')
    plt.show()

    for i in range(16):
        i = i + 1
        plt.subplot(4, 4, i)
        i = i - 1
        plt.plot(FC_train_feature23_CNN[2, :, i])
        plt.xticks([])
        plt.yticks([])
    plt.suptitle('TYPE4')
    plt.show()

    for i in range(16):
        i = i + 1
        plt.subplot(4, 4, i)
        i = i - 1
        plt.plot(FC_train_feature23_CNN[5, :, i])
        plt.xticks([])
        plt.yticks([])
    plt.suptitle('TYPE1')
    plt.show()
    ##############################################cnn聚类#############################################################
    FC_train_feature23_CNN = np.reshape(FC_train_feature23_CNN, (-1, 800))
    # SCM RF
    # svc(FC_train_feature23_CNN, y_train, FC_test_feature23_CNN, y_test)
    # rf(FC_train_feature23_CNN, y_train, FC_test_feature23_CNN, y_test)
    # knn(FC_train_feature23_CNN, y_train, FC_test_feature23_CNN, y_test)
    # fc层t-sne
    y_train = pd.DataFrame(y_train)
    FC_train_feature23_CNN = pd.DataFrame(FC_train_feature23_CNN)
    r = pd.concat([FC_train_feature23_CNN, y_train], axis=1)
    r.columns = list(FC_train_feature23_CNN.columns) + [r'聚类类别']
    t0 = time()
    tsne = TSNE(n_components=2, init="pca", perplexity=500, random_state=1, learning_rate=50)
    tsne.fit_transform(FC_train_feature23_CNN.iloc[:1000, :])  # 进行数据降维,降成两维
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))  # 算法用时
    tsne = pd.DataFrame(tsne.embedding_, index=FC_train_feature23_CNN.iloc[:1000, :].index)  # 转换数据格式
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
    plt.title("1DCNN t-SNE (%.2g sec)" % (t1 - t0))
    plt.show()
    #
    # # ##############################################提取Lstm##########################################################
    get_feature23_L = K.function([model_23.layers[0].input, model_23.layers[3].input], [model_23.layers[5].output])  # CNN3 LSRM4
    FC_train_feature23_L = get_feature23_L([X_train_2d, X_train_2d_T])[0]
    FC_test_feature23_L = get_feature23_L([X_test_2d, X_train_2d_T])[0]
    FC_train_feature23_L = FC_train_feature23_L.transpose(0, 2, 1)
    for i in range(16):
        i = i + 1
        plt.subplot(4, 4, i)
        i = i - 1
        plt.plot(FC_train_feature23_L[0, :, i])
        plt.xticks([])
        plt.yticks([])
    plt.suptitle('TYPE3')
    plt.show()

    for i in range(16):
        i = i + 1
        plt.subplot(4, 4, i)
        i = i - 1
        plt.plot(FC_train_feature23_L[1, :, i])
        plt.xticks([])
        plt.yticks([])
    plt.suptitle('TYPE2')
    plt.show()

    for i in range(16):
        i = i + 1
        plt.subplot(4, 4, i)
        i = i - 1
        plt.plot(FC_train_feature23_L[2, :, i])
        plt.xticks([])
        plt.yticks([])
    plt.suptitle('TYPE4')
    plt.show()

    for i in range(16):
        i = i + 1
        plt.subplot(4, 4, i)
        i = i - 1
        plt.plot(FC_train_feature23_L[6, :, i])
        plt.xticks([])
        plt.yticks([])
    plt.suptitle('TYPE1')
    plt.show()
    #
    FC_train_feature23_L = np.reshape(FC_train_feature23_L, (-1, 800))
    # SCM RF
    # svc(FC_train_feature23_L, y_train, FC_test_feature23_L, y_test)
    # rf(FC_train_feature23_L, y_train, FC_test_feature23_L, y_test)
    # knn(FC_train_feature23_L, y_train, FC_test_feature23_L, y_test)
    # lstm层t-sne
    y_train = pd.DataFrame(y_train)
    FC_train_feature23_L = pd.DataFrame(FC_train_feature23_L)
    r = pd.concat([FC_train_feature23_L, y_train], axis=1)
    r.columns = list(FC_train_feature23_L.columns) + [r'聚类类别']
    t0 = time()
    tsne = TSNE(n_components=2, init="pca", perplexity=500, random_state=1, learning_rate=50)
    tsne.fit_transform(FC_train_feature23_L.iloc[:1000, :])  # 进行数据降维,降成两维
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))  # 算法用时
    tsne = pd.DataFrame(tsne.embedding_, index=FC_train_feature23_CNN.iloc[:1000, :].index)  # 转换数据格式
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
    plt.title("LSTM t-SNE (%.2g sec)" % (t1 - t0))
    plt.show()
    #
    model_02 = model_02()
    lh_2 = LossHistory()
    model_02.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model_02.fit(X_train_2d, y_train_hot, batch_size=40, epochs=1, verbose=1, shuffle=True,
                 callbacks=[lh_2])  # TensorBoard(log_dir='mytensorboard')
    score_02, acc_02 = model_02.evaluate(X_test_2d, y_test_hot)
    print('model_02----', 'Test accuracy:', acc_02, 'Score:', score_02)
    print('loss_2', lh_2.losses)
    print('acc_2', lh_2.accuracy)
    lh_2.loss_plot('batch')
    y_pre2 = model_02.predict([X_test_2d])
    y_pre2 = np.argmax(y_pre2, axis=1)
    print('acc', accuracy_score(y_test, y_pre2))
    print('precision_score', metrics.precision_score(y_test, y_pre2, average='macro'))
    print('recall_score', metrics.recall_score(y_test, y_pre2, average='macro'))
    print('f1_score', metrics.f1_score(y_test, y_pre2, average='weighted'))
    # get_feature2_FC = K.function([model_02.layers[0].input], [model_02.layers[4].output])  # CNN3 LSRM4
    # FC_train_feature2_FC = get_feature2_FC([X_train_2d, X_train_2d_T])[0]
    # FC_test_feature2_FC = get_feature2_FC([X_test_2d, X_train_2d_T])[0]
    # svc(FC_train_feature2_FC, y_train, FC_test_feature2_FC, y_test)
    # rf(FC_train_feature2_FC, y_train, FC_test_feature2_FC, y_test)
    # knn(FC_train_feature2_FC, y_train, FC_test_feature2_FC, y_test)

    model_03 = model_03()
    lh_3 = LossHistory()
    model_03.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model_03.fit(X_train_2d_T, y_train_hot, batch_size=40, epochs=1, verbose=1, shuffle=True, callbacks=[lh_3])  # TensorBoard(log_dir='mytensorboard')
    score_03, acc_03 = model_03.evaluate(X_test_2d_T, y_test_hot)
    print('loss_3', lh_3.losses)
    print('acc_3', lh_3.accuracy)
    lh_3.loss_plot('batch')
    y_pre3 = model_03.predict([X_test_2d_T])
    y_pre3 = np.argmax(y_pre3, axis=1)
    print('acc', accuracy_score(y_test, y_pre3))
    print('precision_score', metrics.precision_score(y_test, y_pre3, average='macro'))
    print('recall_score', metrics.recall_score(y_test, y_pre3, average='macro'))
    print('f1_score', metrics.f1_score(y_test, y_pre3, average='weighted'))

    print('model_23----', 'Test accuracy:', acc_23, 'Score:', score_23)
    print('model_02----', 'Test accuracy:', acc_02, 'Score:', score_02)
    print('model_03----', 'Test accuracy:', acc_03, 'Score:', score_03)

    iters_3 = range(len(lh_3.losses['batch']))
    iters_2 = range(len(lh_2.losses['batch']))
    iters_23 = range(len(lh_23.losses['batch']))

    plt.figure()
    plt.plot(iters_3, lh_3.accuracy['batch'], 'r', label='BiLSTM')
    plt.plot(iters_2, lh_2.accuracy['batch'], 'g', label='1DCNN')
    plt.plot(iters_23, lh_23.accuracy['batch'], 'b', label='1DCNN+BiLSTM')
    plt.xlabel('batch')
    plt.ylabel('acc')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(iters_3, lh_3.losses['batch'], 'r', label='BiLSTM')
    plt.plot(iters_2, lh_2.losses['batch'], 'g', label='1DCNN')
    plt.plot(iters_23, lh_23.losses['batch'], 'b', label='1DCNN+BiLSTM')
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
