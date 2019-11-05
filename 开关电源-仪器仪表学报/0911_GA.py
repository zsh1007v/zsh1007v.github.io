import random
from functools import reduce
from operator import add

import keras
import numpy as np

import glob, warnings, math, sys, time
from keras.callbacks import Callback
from keras import backend as K
from keras import Input, Model
from keras.callbacks import EarlyStopping
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras.callbacks import TensorBoard
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.layers import Conv2D, Dense, MaxPool2D, concatenate, Flatten, LSTM, Dropout, Bidirectional, Conv1D, MaxPool1D
import matplotlib.pyplot as plt
from scipy.optimize import fmin



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


def creat_model(parameters):
    print(parameters)
    learning_rate = parameters['learning_rate']
    # batch_size = parameters['batch_size']
    hidden1_inputs = parameters['hidden1_inputs']
    hidden2_inputs = parameters['hidden2_inputs']
    hidden3_inputs = parameters['hidden3_inputs']
    input1_ = Input(shape=(240, 320, 3), name='input1')
    x1 = Conv2D(hidden1_inputs, (8, 8), activation='relu', padding='same')(input1_)
    x1 = MaxPool2D(pool_size=(4, 8))(x1)
    x1 = Flatten()(x1)

    input2_ = Input(shape=(500, 16), name='input2')
    x2 = Conv1D(hidden2_inputs, kernel_size=4, strides=1, activation='relu', padding='same')(input2_)
    x2 = MaxPool1D(pool_size=4, strides=1, padding='same')(x2)
    x2 = Flatten()(x2)

    input3_ = Input(shape=(500, 16), name='input3')
    x3 = LSTM(hidden3_inputs, return_sequences=True)(input3_)
    x3 = Flatten()(x3)

    x = concatenate([x1, x2, x3])
    output_ = Dense(4, activation='softmax', name='output')(x)
    model = Model(inputs=[input1_, input2_, input3_], outputs=[output_])
    model.summary()
    model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['acc'])
    return model


class Network():
    def __init__(self, parameter_space=None):
        self.accuracy = 0.
        self.parameter_space = parameter_space
        self.network_parameters = {}

    def set_random_parameters(self):
        for parameter in self.parameter_space:
            self.network_parameters[parameter] = random.choice(self.parameter_space[parameter])

    def creat_network(self, network):
        self.network_parameters = network

    def train(self):
        model = creat_model(self.network_parameters)
        tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                                 write_graph=True,  # 是否存储网络结构图
                                 write_grads=True,  # 是否可视化梯度直方图
                                 write_images=True)  # 是否可视化参数

        history = model.fit([X_train_pic, X_train_2d, X_train_2d], y_train, epochs=2,
                            batch_size=20, validation_data=([X_val_pic, X_val_2d, X_val_2d], y_val),
                            verbose=0, shuffle=True, callbacks=[tbCallBack])
        self.accuracy = max(history.history['acc'])
        np.savetxt('train_loss.txt', history.history['loss'])
        np.savetxt('train_acc.txt', history.history['acc'])
        np.savetxt('val_loss.txt', history.history['val_loss'])
        np.savetxt('val_acc.txt', history.history['val_acc'])
        plt.plot(np.loadtxt('train_loss.txt'), color='blue', label='train_loss')
        plt.plot(np.loadtxt('train_acc.txt'), color='green', label='train_acc')
        plt.plot(np.loadtxt('val_loss.txt'), color='blue', label='val_loss')
        plt.plot(np.loadtxt('val_acc.txt'), color='green', label='val_acc')
        plt.legend(loc='best')
        plt.show()

class Genetic_Algorithm():
    def __init__(self, parameter_space, retain=0.3, random_select=0.1, mutate_prob=0.25):
        self.mutate_prob = mutate_prob
        self.random_select = random_select
        self.retain = retain
        self.parameter_space = parameter_space

    def create_population(self, count):
        population = []
        for _ in range(0, count):
            network = Network(self.parameter_space)
            network.set_random_parameters()
            population.append(network)
        return population

    def fitness(network):
        return network.accuracy

    def get_grade(self, population):
        total = reduce(add, (self.fitness(network) for network in population))
        return float(total)/len(population)

    def breed(self, mother, father):
        print('交叉。。。')
        children = []
        for _ in range(2):
            child = {}
            for param in self.parameter_space:
                child[param] = random.choice([mother.network[param], father.network[param]])
                network = Network(self.nn_param_choices)
                network.creat_set(child)
                if self.mutate_chance > random.random():
                    network = self.mutate(network)
                children.append(network)
        return children

    def mutate(self, network):
        print('变异。。。')
        mutation = random.choice(list(self.parameter_space.keys()))
        network.network[mutation] = random.choice(self.parameter_space[mutation])
        return network


    def evolve(self, pop):
        print('进化。。。')
        graded = [(self.fitness(network), network) for network in pop]
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]
        retain_length = int(len(graded) * self.retain)
        parents = graded[:retain_length]
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)
            parents_length = len(parents)
            desired_length = len(pop) - -parents_length
            children = []
            while len(children) < desired_length:
                male = random.randint(0, parents_length - 1)
                female = random.randint(0, parents_length - 1)
                if male != female:
                    male = parents[male]
                    female = parents[female]
                    children_new = self.breed(male, female)
                    for child_new in children_new:
                        if len(children) < desired_length:
                            children.append(child_new)
            parents.extend(children)
            return parents

    def get_population_accuracy(population):
        total_accuracy = 0
        for network in population:
            total_accuracy += network.get_accuracy
        return total_accuracy / len(population)


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
    y_train, y_val = to_categorical(y_train, 4), to_categorical(y_val, 4)
    X_train_pic = list_pic('D:/PycharmProjects/untitled/img_train/X_train_1d'+'*.png')
    X_test_pic = list_pic('D:/PycharmProjects/untitled/img_test/X_test_1d'+'*.png')
    X_val_pic = list_pic('D:/PycharmProjects/untitled/img_val/X_val_1d'+'*.png')

    parameter_space = {'learning_rate': [0.01, 0.001, 0.0001],
                       'hidden1_inputs': [16, 32, 48],
                       'hidden2_inputs': [16, 32, 48],
                       'hidden3_inputs': [20, 40, 60, 80, 100]}
    n_generations = 3
    population_size = 10
    print('构建GA...')
    GA = Genetic_Algorithm(parameter_space)
    print('初始化种群')
    population = GA.create_population(population_size)

    for i in range(n_generations):
        print('Generation{}'.format(i))
        for network in population:
            network.train()
        average_accuracy = GA.get_population_accuracy(population)
        print('Average accuracy:{:.2f}'.format(average_accuracy))
        if i < n_generations-1:
            s = GA.evolve(network)
    print("我也不知道要输出啥。。。")

    predict = network.predict([X_test_pic, X_test_2d, X_test_2d])
    y_pre = np.argmax(predict, axis=1)

    a = 0
    for i in range(len(y_test)):
        if y_pre[i] == y_test[i]:
            a = a + 1
    accuracy = a / len(y_test) * 100
    print(accuracy, '%')

    y_test = to_categorical(y_test)
    score = network.evaluate([X_test_pic, X_test_2d, X_test_2d], y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
