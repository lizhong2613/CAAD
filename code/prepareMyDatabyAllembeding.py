# -*- coding: utf-8 -*-
import random
import argparse
import config
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def getTrainData():
    print("prepare Data")
    # 打开对应文件夹下面的描述文件  其中graph中存储的为边的链接描述信息 data文件中为节点的描述信息 zhihu 为中文描述
    # f = open('../datasets/zhihu/graph.txt' , 'rb')
    f = open('../temp/graph.txt', 'rb')
    edges = [list(map(int, i.strip().decode().split('\t'))) for i in f]
    print(len(edges))
    nodesTrain = list(set([i for j in edges for i in j]))
    # 表示信息
    node2vec = {}
    # 使用结构的嵌入信息
    fline = open('../temp/vec_all.txt', 'rb')
    dvec = {}
    for i, j in enumerate(fline):
        if j.decode() != '\n':
            tempvec = list(map(float, j.strip().decode().split(' ')))
            dvec[tempvec[0]] = list(tempvec[1:])
    # print(dvec)
    f = open('../temp/embed.txt', 'rb')
    for i, j in enumerate(f):
        if j.decode() != '\n':
            a = list(map(float, j.strip().decode().split(' ')))
            # node2vec[i] = list(dvec[i]) +[i *0.3 for  i in a]
            node2vec[i] = list(dvec[i]) +a
    edgesEmbed = []
    for i, j in edges:
        if i in node2vec.keys() and j in node2vec.keys():
            # distance = euclideann2(node2vec[i], node2vec[j])
            # disTrain.append(list(node2vec[i])+list(node2vec[j]))
            edge = np.append(node2vec[i], node2vec[j])  # 计算为正样本的概率
            edgesEmbed.append(edge)
    edgesEmbedarray = np.array(edgesEmbed)
    X_train, X_test = train_test_split(edgesEmbedarray, test_size=0.2, random_state=42)

    edgeAnomaly = []
    fanomaly = open('../temp/anomalyedge3.txt', 'rb')
    edgesAnomaly = [list(map(int, i.strip().decode().split(' '))) for i in fanomaly]
    print(len(edgesAnomaly))
    for i, j in edgesAnomaly:
        if i in node2vec.keys() and j in node2vec.keys():
            # distance = euclideann2(node2vec[i], node2vec[j])
            # disTrain.append(list(node2vec[i])+list(node2vec[j]))
            edge = np.append(node2vec[i], node2vec[j])  # 计算为正样本的概率
            edgeAnomaly.append(edge)

    edgeAnomalyarray = np.array(edgeAnomaly)
    print(edgesEmbedarray.shape)
    print(X_train.shape)
    print(X_test.shape)
    print(edgeAnomalyarray.shape)

    print("prepare Data ended")
    return  X_train,X_test,edgeAnomalyarray


