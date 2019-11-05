from time import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
# data = pd.read_csv(open(r'D:\+++【做对自己有用的事情！】\淡定淡定每天写一点，做就对了。。。\数据图表\type1.csv', 'r'))
# print(data.iloc[:, 0])
# for i in range(16):
#     ax = plt.subplot(16,1,i+1)
#     plt.plot(data.iloc[:,i])
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     plt.xticks([])
#     plt.yticks([])
# plt.xticks(np.linspace(0, 1500, 7))
# plt.tick_params(labelsize=25)
# plt.show()
from mpl_toolkits.mplot3d import Axes3D

# X_train_2d = np.load(file="X_train_2d_0.5.npy")
# X_test_2d = np.load(file="X_test_2d_0.5.npy")
# X_train_1d = np.load(file="X_train_1d_0.5.npy")
# # X_test_1d = np.load(file="X_test_1d_0.5.npy")
# y_train = np.load(file="y_train_0.5.npy") - 1
# # y_test = np.load(file="y_test_0.5.npy") - 1
# #
# data_1d = np.load(file="data_1d.npy")
# print(data_1d.shape)# 4000,8000
# data_y = np.load(file="data_y.npy")
# print(data_y.shape)# 4000,8000
# y_train = pd.DataFrame(y_train)
# X_train_1d = pd.DataFrame(X_train_1d)
# from sklearn.manifold import TSNE
# r = pd.concat([X_train_1d,y_train],axis=1)
# r.columns = list(X_train_1d.columns) + [r'聚类类别']
# print(r)
# t0 = time()
# tsne=TSNE(n_components=2, init="pca", perplexity=900, random_state=1, learning_rate=500)
# tsne.fit_transform(X_train_1d.iloc[:1000, :])  #进行数据降维,降成两维
# t1 = time()
# print("t-SNE: %.2g sec" % (t1 - t0))  # 算法用时
# tsne=pd.DataFrame(tsne.embedding_, index=X_train_1d.iloc[:1000, :].index) #转换数据格式
# import matplotlib.pyplot as plt
#
# d = tsne[r[r'聚类类别']==0]
# plt.scatter(d[0], d[1],cmap=plt.cm.Spectral)#.
# d = tsne[r[r'聚类类别']==1]
# # plt.plot(d[0],d[1],'go')#o
# plt.scatter(d[0], d[1],cmap=plt.cm.Spectral)#.
# d = tsne[r[r'聚类类别']==2]
# # plt.plot(d[0],d[1],'b*')#*
# plt.scatter(d[0], d[1],cmap=plt.cm.Spectral)#.
# d = tsne[r[r'聚类类别']==3]
# # plt.plot(d[0], d[1],'y+')#*
# plt.scatter(d[0], d[1], cmap=plt.cm.Spectral)#.
# plt.spines['top'].set_visible(False)
# plt.spines['right'].set_visible(False)
# plt.spines['bottom'].set_visible(False)
# plt.spines['left'].set_visible(False)
# plt.xticks([])
# plt.yticks([])
# plt.legend(['TYPE1','TYPE2','TYPE3','TYPE4'])
# plt.title("t-SNE (%.2g sec)" % (t1 - t0))
# plt.show()
# from sklearn.model_selection import train_test_split
# data_1d = np.load(file="data_1d.npy")
# print(data_1d.shape)
# data_2d = np.load(file="data_2d.npy")
# data_y = np.load(file="data_y.npy")
# data_y = data_y-1
# #
# X_train_1d, X_test_1d, y_train, y_test = train_test_split(data_1d, data_y, test_size=0.75, random_state=0, shuffle=True)
# # # X_test_2d, X_val_2d, y_test, y_val = train_test_split(X_test_2d, y_test, test_size=0.5, random_state=1, shuffle=True)
# # print(X_train_2d.shape)
# # # # print(X_val_2d.shape)
# # print(X_test_2d.shape)
# # np.save(file="X_train_2d_0.5.npy", arr=X_train_2d)
# # np.save(file="X_test_2d_0.5.npy", arr=X_test_2d)
# # # # np.save(file="X_val_2d_0.5.npy", arr=X_val_2d)
# # # np.save(file="y_train_0.5.npy", arr=y_train)
# # y_train = np.load(file="y_train_0.5.npy")
# # print(y_train[:10])
# # np.save(file="y_test_0.5.npy", arr=y_test)
#
# y_train = pd.DataFrame(y_train)
# print(y_train)
# X_train_1d = pd.DataFrame(X_train_1d)
# from sklearn.manifold import TSNE
# r = pd.concat([X_train_1d,y_train],axis=1)
# r.columns = list(X_train_1d.columns) + [r'聚类类别']
# print(r)
# t0 = time()
# tsne=TSNE(n_components=2, init="pca", perplexity=8000, random_state=1, learning_rate=200)
# tsne.fit_transform(X_train_1d.iloc[:1500, :])  #进行数据降维,降成两维
# t1 = time()
# print("t-SNE: %.2g sec" % (t1 - t0))  # 算法用时
# tsne=pd.DataFrame(tsne.embedding_, index=X_train_1d.iloc[:1500, :].index) #转换数据格式
# import matplotlib.pyplot as plt
#
# d = tsne[r[r'聚类类别']==0]
# plt.scatter(d[0], d[1],cmap=plt.cm.Spectral)#.
# d = tsne[r[r'聚类类别']==1]
# # plt.plot(d[0],d[1],'go')#o
# plt.scatter(d[0], d[1],cmap=plt.cm.Spectral)#.
# d = tsne[r[r'聚类类别']==2]
# # plt.plot(d[0],d[1],'b*')#*
# plt.scatter(d[0], d[1],cmap=plt.cm.Spectral)#.
# d = tsne[r[r'聚类类别']==3]
# # plt.plot(d[0], d[1],'y+')#*
# plt.scatter(d[0], d[1], cmap=plt.cm.Spectral)#.
# # plt.spines['top'].set_visible(False)
# # plt.spines['right'].set_visible(False)
# # plt.spines['bottom'].set_visible(False)
# # plt.spines['left'].set_visible(False)
# plt.xticks([])
# plt.yticks([])
# plt.legend(['TYPE1','TYPE2','TYPE3','TYPE4'])
# plt.title("t-SNE (%.2g sec)" % (t1 - t0))
# plt.show()
import seaborn as sns
data = pd.read_csv(open(r'C:\Users\Saber\Desktop\data.csv', 'rb'))
data = pd.DataFrame(data)
print(data)
sns.boxplot(x="type",y="BiLSTM",data=data,hue="1DCNN")
plt.show()