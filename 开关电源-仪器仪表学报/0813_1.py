'''
构造数据集40(样本)*8000(16*500, 16个通道数据变为一行)
训练集28，测试集6，验证集6
样本之间y有重合，重合比例0.5，每一类的样本数1000
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# 采样率25k,250K个采样点,10s,  数据集大小250000*16
data1 = pd.read_csv(open(r'D:\PycharmProjects\untitled\ZCdata1_2.csv', 'r'))
data2 = pd.read_csv(open(r'D:\PycharmProjects\untitled\PWMdata1_2.csv', 'r'))
data3 = pd.read_csv(open(r'D:\PycharmProjects\untitled\KGGdata1_2.csv', 'r'))
data4 = pd.read_csv(open(r'D:\PycharmProjects\untitled\Powerdata1_2.csv', 'r'))
minMax = MinMaxScaler()#将数据进行归一化
data1 = pd.DataFrame(minMax.fit_transform(data1))
data2 = pd.DataFrame(minMax.fit_transform(data2))
data3 = pd.DataFrame(minMax.fit_transform(data3))
data4 = pd.DataFrame(minMax.fit_transform(data4))

TYPE1, TYPE2, TYPE3, TYPE4 = 1, 2, 3, 4
type1, type2, type3, type4, d1, d2, d3, d4 = dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict()
chunk1, chunk2, chunk3, chunk4 = [], [], [], []
array1, array2, array3, array4 = [], [], [], []
# 数据集切分，样本之间无重合，a表示每一类的样本数
a = 4000
for i in range(a):#500*16=8000  （1，8000）

    # # 无重叠截取数据块500*16
    # type1[i] = data1.iloc[i*500:i*500 + 500]
    # # 每个数据块的16通道拼接为1列（8000，1）
    # d1[i] = pd.concat(type1[i].iloc[:, j] for j in range(type1[i].shape[1]))
    # # 列表末尾添加新的对象
    # chunk1.append(d1[i].reset_index(drop=True))
    # data_1 = pd.concat(chunk1, axis=1, ignore_index=True)

    # 有重叠截取数据块500*16，重叠比例为50%
    # 构造lstm和1Dcnn数据 三维数组（1000，500，16）
    array1.append(data1.iloc[i*100:i*100+500].values)
    type1_2d = np.array(array1)  # type1_2d[*, :]提取第*个二维矩阵
    # 构造2D频谱图数据 一维数组8000*1
    type1[i] = data1.iloc[i*100:i*100 + 500]
    # 每个数据块的16通道拼接为1列（8000，1）
    d1[i] = pd.concat(type1[i].iloc[:, j] for j in range(type1[i].shape[1]))
    # 列表末尾添加新的对象
    chunk1.append(d1[i].reset_index(drop=True))
    # 每一个样本500*16被拼接为1列，大小为8000*1；1000个样本横向拼接，大小为8000*1000
    type1_1d = pd.concat(chunk1, axis=1, ignore_index=True)

    array2.append(data2.iloc[i*100:i*100 + 500].values)
    type2_2d = np.array(array2)
    type2[i] = data2.iloc[i*100:i*100+500]
    d2[i] = pd.concat(type2[i].iloc[:, j] for j in range(type2[i].shape[1]))
    chunk2.append(d2[i].reset_index(drop=True))
    type2_1d = pd.concat(chunk2, axis=1, ignore_index=True)

    array3.append(data3.iloc[i*100:i*100+500].values)
    type3_2d = np.array(array3)
    type3[i] = data3.iloc[i*100:i*100 + 500]
    d3[i] = pd.concat(type3[i].iloc[:, j] for j in range(type3[i].shape[1]))
    chunk3.append(d3[i].reset_index(drop=True))
    type3_1d = pd.concat(chunk3, axis=1, ignore_index=True)
#
    array4.append(data4.iloc[i*100:i*100+500].values)
    type4_2d = np.array(array4)
    print(i)
    print('维度:', type4_2d.ndim)
    print('shape:', type4_2d.shape)
    type4[i] = data4.iloc[i*100:i*100+500]
    d4[i] = pd.concat(type4[i].iloc[:, j]for j in range(type4[i].shape[1]))
    chunk4.append(d4[i].reset_index(drop=True))
    type4_1d = pd.concat(chunk4, axis=1, ignore_index=True)
#
# 样本转置，并在一行末尾处增加所属类别，每类1000个样本纵向拼接，大小为：4000*8001
type1_y = pd.DataFrame([TYPE1]*a)
type2_y = pd.DataFrame([TYPE2]*a)
type3_y = pd.DataFrame([TYPE3]*a)
type4_y = pd.DataFrame([TYPE4]*a)
data_y = pd.concat([type1_y, type2_y, type3_y, type4_y], axis=0, ignore_index=True).values# array类型
data_1d = pd.concat([type1_1d.T, type2_1d.T, type3_1d.T, type4_1d.T], axis=0, ignore_index=True).values# array类型1000, 8000
print('维度:', data_1d.ndim)
print('shape:', data_1d.shape)
np.save(file="type1_2d.npy", arr=type1_2d)
np.save(file="type2_2d.npy", arr=type2_2d)
np.save(file="type3_2d.npy", arr=type3_2d)
np.save(file="type4_2d.npy", arr=type4_2d)
type1_2d = np.load(file="type1_2d.npy")
type2_2d = np.load(file="type2_2d.npy")
type3_2d = np.load(file="type3_2d.npy")
type4_2d = np.load(file="type4_2d.npy")
data_2d = np.concatenate((type1_2d, type2_2d, type3_2d, type4_2d), axis=0)# narray类型 4000, 500, 16
np.save(file="data_1d.npy", arr=data_1d)
np.save(file="data_2d.npy", arr=data_2d)
np.save(file="data_y.npy", arr=data_y)
print('维度:', data_2d.ndim)
print('shape:', data_2d.shape)
# ######## 训练集28，测试集6，验证集6'#########(4000行*8000列)25kHz 250K  10s
X_train_2d, X_test_2d, y_train, y_test = train_test_split(data_2d, data_y, test_size=0.75, random_state=0, shuffle=True)
# X_test_2d, X_val_2d, y_test, y_val = train_test_split(X_test_2d, y_test, test_size=0.5, random_state=1, shuffle=True)
print(X_train_2d.shape)
# print(X_val_2d.shape)
print(X_test_2d.shape)
np.save(file="X_train_2d_0.5.npy", arr=X_train_2d)
np.save(file="X_test_2d_0.5.npy", arr=X_test_2d)
# np.save(file="X_val_2d_0.5.npy", arr=X_val_2d)
np.save(file="y_train_0.5.npy", arr=y_train)
np.save(file="y_test_0.5.npy", arr=y_test)
# np.save(file="y_val_0.5.npy", arr=y_val)
#
# # ###############################################################################################
X_train_1d, X_test_1d, y_train, y_test = train_test_split(data_1d, data_y, test_size=0.75, random_state=0, shuffle=True)
# # X_test_1d, X_val_1d, y_test, y_val = train_test_split(X_test_1d, y_test, test_size=0.5, random_state=1, shuffle=True)
np.save(file="X_train_1d_0.5.npy", arr=X_train_1d)
np.save(file="X_test_1d_0.5.npy", arr=X_test_1d)
# # np.save(file="X_val_1d_0.5.npy", arr=X_val_1d)


# for i in range(1400):
#     spectrum, freqs, ts, fig = plt.specgram(X_train_1d[i, :], NFFT=256, noverlap=128, Fs=25600, window=np.hanning(M=256),
#                                             scale_by_freq=True, sides='default', mode='default', scale='dB')
#     plt.axis('off')
#     plt.margins(0, 0)
#     plt.gca().xaxis.set_major_locator(plt.NullLocator())
#     plt.gca().yaxis.set_major_locator(plt.NullLocator())
#     plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
#     plt.savefig("X_train_1d"+str(i)+".png", transparent=True, dpi=50, pad_inched=0)
# for i in range(300):
#     spectrum, freqs, ts, fig = plt.specgram(X_test_1d[i, :], NFFT=256, noverlap=128, Fs=25600, window=np.hanning(M=256),
#                                             scale_by_freq=True, sides='default', mode='default', scale='dB')
#     plt.axis('off')
#     plt.margins(0, 0)
#     plt.gca().xaxis.set_major_locator(plt.NullLocator())
#     plt.gca().yaxis.set_major_locator(plt.NullLocator())
#     plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
#     plt.savefig("X_test_1d"+str(i)+".png", transparent=True, dpi=50, pad_inched=0)
# for i in range(300):
#     spectrum, freqs, ts, fig = plt.specgram(X_val_1d[i, :], NFFT=256, noverlap=128, Fs=25600, window=np.hanning(M=256),
#                                             scale_by_freq=True, sides='default', mode='default', scale='dB')
#     plt.axis('off')
#     plt.margins(0, 0)
#     plt.gca().xaxis.set_major_locator(plt.NullLocator())
#     plt.gca().yaxis.set_major_locator(plt.NullLocator())
#     plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
#     plt.savefig("X_val_1d"+str(i)+".png", transparent=True, dpi=50, pad_inched=0)
