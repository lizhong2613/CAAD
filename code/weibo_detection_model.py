# -*- coding: utf-8 -*-
from processed import read_file
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import matplotlib.pyplot as plt
from numpy.random import seed
from prepareMyData import getTrainData
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
import random
import argparse
import config
from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.metrics import roc_auc_score,roc_curve,precision_score,auc,precision_recall_curve, \
                            accuracy_score,recall_score,f1_score,confusion_matrix,classification_report
import pandas as pd
import os
from DataSet import dataSet
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from keras.models import load_model

def getTrainEdgeIndex():
    # load data
    f = open('temp/graph.txt', 'rb')
    edges = [list(map(int, i.strip().decode().split('\t'))) for i in f]
    trainEdge, testEdge = train_test_split(edges, test_size=0.2, random_state=42)
    return trainEdge, testEdge, edges


def getEmbedding(hasText, method):
    filpath = 'temp/{}_cora/vec_all60.txt'.format(method)
    # 表示信息
    embedding = {}
    # 使用结构的嵌入信息
    fline = open(filpath, 'rb')
    dvec = {}
    for i, j in enumerate(fline):
        if j.decode() != '\n' and i >0:
            tempvec = list(map(float, j.strip().decode().split(' ')))
            dvec[tempvec[0]] = list(tempvec[1:])
    # print(dvec)
    if hasText == True:
        f = open('temp/embed60.txt', 'rb')
        for i, j in enumerate(f):
            if j.decode() != '\n' :
                a = list(map(float, j.strip().decode().split(' ')))
                # node2vec[i] = list(dvec[i]) +[i *0.3 for  i in a]
                if i in dvec.keys():
                    embedding[i] = list(dvec[i]) + a
                else:
                    embedding[i] = list([0] *128) + a

    else:
        embedding = dvec

    return embedding


def getTrainData(embedding):
    print("prepare Data")
    # load data
    graph_path = os.path.join('temp/graph.txt')  # 边的文件
    text_path = os.path.join("..", "datasets", 'cora', 'data.txt')  # 找到节点的描述文件
    data = dataSet(text_path, graph_path)
    # 打开对应文件夹下面的描述文件  其中graph中存储的为边的链接描述信息 data文件中为节点的描述信息 zhihu 为中文描述

    edgesEmbed = []
    for i, j in edge_Train:
        if i in embedding.keys() and j in embedding.keys():
            lsnodes = data.negNnodes(i, j, 10)

            dot2 = np.dot(embedding[i], embedding[j])
            tempEmbed = []
            for m in range(len(lsnodes)):
                if lsnodes[m] in embedding.keys():
                    dot1 = np.dot(embedding[i], embedding[lsnodes[m]])
                    tempVal = dot2 - dot1
                    tempEmbed.append(tempVal)
                else:
                    tempVal =dot2
                    tempEmbed.append(tempVal)
            edgesEmbed.append(tempEmbed)
    edgesEmbedarray = np.array(edgesEmbed)
    X_train = edgesEmbedarray

    edgesEmbed = []
    for i, j in edge_Test:
        if i in embedding.keys() and j in embedding.keys():
            lsnodes = data.negNnodes(i, j, 10)

            dot2 = np.dot(embedding[i], embedding[j])
            tempEmbed = []
            for m in range(len(lsnodes)):
                if lsnodes[m] in embedding.keys():
                    dot1 = np.dot(embedding[i], embedding[lsnodes[m]])
                    tempVal = dot2 - dot1
                    tempEmbed.append(tempVal)
                else:
                    tempVal =dot2
                    tempEmbed.append(tempVal)

            edgesEmbed.append(tempEmbed)
    edgesEmbedarray = np.array(edgesEmbed)
    X_test = edgesEmbedarray

    edgeAnomaly = []
    fanomaly = open('temp/anomalyedgecora.txt', 'rb')
    edgesAnomaly = [list(map(int, i.strip().decode().split(' '))) for i in fanomaly]
    print(len(edgesAnomaly))
    edgestr=[]
    for i, j in edgesAnomaly:
        if i in embedding.keys() and j in embedding.keys():
            # distance = euclideann2(node2vec[i], node2vec[j])
            # disTrain.append(list(node2vec[i])+list(node2vec[j]))
            lsnodes = data.negNnodes(i, j,10)
            dot2 = np.dot(embedding[i], embedding[j])
            tempEmbed = []
            for m in range(len(lsnodes)):
                if lsnodes[m] in embedding.keys():
                    dot1 = np.dot(embedding[i], embedding[lsnodes[m]])
                    tempVal = dot2 - dot1
                    tempEmbed.append(tempVal)
                else:
                    tempVal =dot2
                    tempEmbed.append(tempVal)
            edgeAnomaly.append(tempEmbed)
            edgestr.append("{},{}".format(i,j))

            # edge = np.append(node2vec[i], node2vec[j])  # 计算为正样本的概率
            # edgeAnomaly.append(edge)
    print('edgestr',edgestr)
    edgeAnomalyarray = np.array(edgeAnomaly)
    print(edgesEmbedarray.shape)
    print(X_train.shape)
    print(X_test.shape)
    print(edgeAnomalyarray.shape)

    print("prepare Data ended")
    return X_train, X_test, edgeAnomalyarray

featureFile =  os.path.join('datasets/microblog/microblogPCU_vertices.csv')  # 边的文件
feature =pd.read_csv(featureFile)
print(feature)
exit(0)
#X_train, X_test, X_fraud = read_file(dataname)
embeddingMethod ='line1'
embedding = getEmbedding(True, embeddingMethod)
edge_Train, edge_Test, edge_All = getTrainEdgeIndex()  # 获取对比的数据的边
X_train, X_test, X_fraud = getTrainData(embedding)

# 设置autoencoder的参数: 隐藏层参数设置：16,8,8,16; epoch_size为50;batch_size 为32
input_dim = X_train.shape[1]
encoding_dim = 16
num_epoch = 50
batch_size = 32

# 利用keras构建模型
input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="tanh",
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim/2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim/2), activation="tanh")(encoder)
decoder = Dense(input_dim, activation="relu")(decoder)
autoencoder = Model(inputs=input_layer,outputs=decoder)
autoencoder.compile(optimizer="adam",
                    loss="mean_squared_error",
                    metrics=['mae'])    # 评价函数和 损失函数 相似，只不过评价函数的结果不会用于训练过程中

# 将模型保存为sofasofa_model.h5,并开始训练。
checkpointer = ModelCheckpoint(filepath="normEdge_{}_model.h5".format(embeddingMethod),
                               verbose=0,
                               save_best_only=True)
history = autoencoder.fit(X_train, X_train,
                          epochs=num_epoch,
                          batch_size=batch_size,
                          shuffle=True,
                          validation_data=(X_test, X_test),
                          verbose=1,
                          callbacks=[checkpointer]).history

# 画出损失函数曲线
plt.figure(figsize=(7,6))
# plt.subplot(121)
plt.plot(history["loss"], c='dodgerblue', lw=3)
plt.plot(history["val_loss"], c='coral', lw=3)
plt.title('model loss')
plt.ylabel('mse'); plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

# plt.subplot(122)
# plt.plot(history['mean_absolute_error'], c='dodgerblue', lw=3)
# plt.plot(history['val_mean_absolute_error'], c='coral', lw=3)
# plt.title('model_mae')
# plt.ylabel('mae');plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
plt.show()
# 读取模型
autoencoder = load_model("normEdge_{}_model.h5".format(embeddingMethod))

# 利用训练好的autoencoder重建测试集(X_test;X_fraud  前者是正常样本，后者是所有的异常样本)
pred_test = autoencoder.predict(X_test)
pred_fraud = autoencoder.predict(X_fraud)
print("fraud length",len(pred_fraud))

# 计算还原误差MSE和MAE ；前者是loss；后者是metrics的mae
mse_test = np.mean(np.power(X_test - pred_test, 2), axis=1)
print("mse_test")
print(mse_test)
mse_fraud = np.mean(np.power(X_fraud - pred_fraud, 2), axis=1)
print("mse_fraud")
print(mse_fraud)
mae_test = np.mean(np.abs(X_test - pred_test), axis=1)
mae_fraud = np.mean(np.abs(X_fraud - pred_fraud), axis=1)
print('mae_test',mae_test)
print('mae_fraud',mae_fraud)
mse_df = pd.DataFrame()

mse_df['Class'] = [0]*len(mse_test) + [1]*len(mse_fraud)
print(mse_df['Class'])
mse_df['MSE'] = np.hstack([mse_test, mse_fraud])
mse_df['MAE'] = np.hstack([mae_test, mae_fraud])
#print(mse_df)
mse_df = mse_df.sample(frac=1).reset_index(drop=True)
#print(mse_df)

"""
 sample（）参数frac是要返回的比例，比如df中有10行数据，我只想返回其中的30%,那么frac=0.3
 set_index()和reset_oindex()的区别 前者为现有的dataframe设置不同于之前的index;而后者是还原和最初的index方式：0,1,2,3,4……
"""

# 分别画出测试集中正样本和负样本的还原误差MAE和MSE
markers = ['o', '^']
colors = ['dodgerblue', 'coral']
labels = ['Non-fraud', 'Fraud']

plt.figure(figsize=(14, 5))
plt.subplot(121)
for flag in [1, 0]:
    temp = mse_df[mse_df['Class'] == flag]
    plt.scatter(temp.index,
                temp['MAE'],
                alpha=0.7,
                marker=markers[flag],
                c=colors[flag],
                label=labels[flag])
plt.title('Reconstruction MAE')
plt.ylabel('Reconstruction MAE'); plt.xlabel('Index')
plt.subplot(122)
for flag in [1, 0]:
    temp = mse_df[mse_df['Class'] == flag]
    plt.scatter(temp.index,
                temp['MSE'],
                alpha=0.7,
                marker=markers[flag],
                c=colors[flag],
                label=labels[flag])
plt.legend(loc=[1, 0], fontsize=12); plt.title('Reconstruction MSE')
plt.ylabel('Reconstruction MSE'); plt.xlabel('Index')
plt.show()

# 画出Precision-Recall曲线
plt.figure(figsize=(14, 6))
for i, metric in enumerate(['MAE', 'MSE']):
    plt.subplot(1, 2, i+1)
    precision, recall, _ = precision_recall_curve(mse_df['Class'], mse_df[metric])
    pr_auc = auc(recall, precision)
    plt.title('Precision-Recall curve based on %s\nAUC = %0.2f'%(metric, pr_auc))
    plt.plot(recall[:-2], precision[:-2], c='coral', lw=4)
    plt.xlabel('Recall'); plt.ylabel('Precision')
plt.show()

#  画出ROC曲线
plt.figure(figsize=(7, 6))
for i, metric in enumerate(['MAE']):
   #plt.subplot(1, 2, i+1)
    fpr, tpr, _ = roc_curve(mse_df['Class'],mse_df[metric])
    roc_auc = auc(fpr, tpr)
    plt.title('ROC based on %s\nAUC = %0.2f'%(metric, roc_auc))
    plt.plot(fpr, tpr, c='coral', lw=4)
    plt.plot([0, 1], [0, 1], c='dodgerblue', ls='--')
    plt.ylabel('TPR');plt.xlabel('FPR')
plt.show()
#  画出ROC曲线
plt.figure(figsize=(7, 6))
for i, metric in enumerate([ 'MSE']):
    #plt.subplot(1, 2, i+1)
    fpr, tpr, _ = roc_curve(mse_df['Class'],mse_df[metric])
    roc_auc = auc(fpr, tpr)
    plt.title('ROC based on %s\nAUC = %0.3f'%(metric, roc_auc))
    plt.plot(fpr, tpr, c='coral', lw=4)
    plt.plot([0, 1], [0, 1], c='dodgerblue', ls='--')
    plt.ylabel('TPR');plt.xlabel('FPR')
plt.show()

# 画出MSE、MAE散点图
markers = ['o', '^']
colors = ['dodgerblue', 'coral']
labels = ['Nomal', 'Anomaly']

plt.figure(figsize=(10, 5))
for flag in [1, 0]:
    temp = mse_df[mse_df['Class'] == flag]
    plt.scatter(temp['MAE'],
                temp['MSE'],
                alpha=0.7,
                marker=markers[flag],
                c=colors[flag],
                label=labels[flag])
plt.legend(loc=[1, 0])
plt.ylabel('Reconstruction RMSE'); plt.xlabel('Reconstruction MAE')
plt.show()