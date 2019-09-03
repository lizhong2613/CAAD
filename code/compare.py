# -*- coding: utf-8 -*-
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

from keras.models import load_model
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import seed


warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from DataSet import dataSet
from sklearn.metrics import roc_curve, auc, precision_recall_curve

ratio = 50
hasText = True


def getTrainEdgeIndex():
    # load data
    f = open('temp/graph.txt', 'rb')
    edges = [list(map(int, i.strip().decode().split('\t'))) for i in f]
    trainEdge, testEdge = train_test_split(edges, test_size=0.2, random_state=42)
    return trainEdge, testEdge, edges


def getEmbedding(hasText, method):
    filpath = 'temp/{}_cora/vec_all{}.txt'.format(method,ratio)
    # 表示信息
    embedding = {}
    # 使用结构的嵌入信息
    fline = open(filpath, 'rb')
    dvec = {}
    for i, j in enumerate(fline):
        if j.decode() != '\n' and i > 0:
            tempvec = list(map(float, j.strip().decode().split(' ')))
            dvec[tempvec[0]] = list(tempvec[1:])
    # print(dvec)
    if hasText == True:
        f = open('temp/embed{}.txt'.format(ratio), 'rb')
        for i, j in enumerate(f):
            if j.decode() != '\n':
                a = list(map(float, j.strip().decode().split(' ')))
                # node2vec[i] = list(dvec[i]) +[i *0.3 for  i in a]
                if i not in dvec.keys():
                    embedding[i] = list([0] * 128 ) + a
                else:
                    embedding[i] = list(dvec[i]) + a
    else:
        embedding = dvec

    return embedding


def getTrainData(embedding):
    print("prepare Data")
    # load data
    graph_path = os.path.join('temp/graph.txt')  # 边的文件
    text_path = os.path.join("..", "datasets", 'zhihu', 'data.txt')  # 找到节点的描述文件
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
            # edge = np.append(node2vec[i], node2vec[j])  # 计算为正样本的概率
            # edgeAnomaly.append(edge)

    edgeAnomalyarray = np.array(edgeAnomaly)
    print(edgesEmbedarray.shape)
    print(X_train.shape)
    print(X_test.shape)
    print(edgeAnomalyarray.shape)

    print("prepare Data ended")
    return X_train, X_test, edgeAnomalyarray


edge_Train, edge_Test, edge_All = getTrainEdgeIndex()  # 获取对比的数据的边
lsMethods = ['line1','line3',"deepwalk", 'node2vec']
fixed_X_test = []
fixed_X_fraud =  []

plt.figure(figsize=(7, 6))
plt.plot([0, 1], [0, 1], c='dodgerblue', ls='--')
plt.ylabel('TPR');
plt.xlabel('FPR')

plt.title('ROC ')

for i in range(len(lsMethods)):

    embedding = getEmbedding(hasText, lsMethods[i])
    X_train, X_test, X_fraud = getTrainData(embedding)
    fixed_X_fraud = X_fraud
    fixed_X_test = X_test
    All_df = pd.DataFrame()

    y_sample = [0] * (len(X_train) + len(X_test)) + [1] * len(X_fraud)
    # print(mse_df['Class'])
    All_train1 = np.append(X_train, X_test, axis=0)
    All_train = np.append(All_train1,X_fraud,axis=0)
    X_sample, y_sample = SMOTE(random_state=6, k_neighbors=2).fit_sample(All_train, y_sample)

    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=2019)
    print(y_sample)
    # 构建参数组合
    param_grid = {'C': [0.0001,0.001,0.01, 0.1, 1, 10, 100, 1000, ],
                  'penalty': ['l1', 'l2']}

    grid_search = GridSearchCV(LogisticRegression(), param_grid,
                               cv=10)  # 确定模型LogisticRegression，和参数组合param_grid ，cv指定10折
    grid_search.fit(X_train, y_train)  # 使用训练集学习算法

    best_model = grid_search.best_estimator_
    print('--------------------{}---------------'.format(lsMethods[i]))
    print(best_model.predict(X_test))

    print('accuracy_score:', accuracy_score(y_test, best_model.predict(X_test)))
    print('roc_auc_score:', roc_auc_score(y_test, best_model.predict(X_test)))
    print('recall_score:', recall_score(y_test, best_model.predict(X_test),pos_label=0))
    print('precision_score:', precision_score(y_test, best_model.predict(X_test),pos_label=0))

    #  画出ROC曲线

    fpr, tpr, _ = roc_curve(y_test, best_model.predict(X_test))
    roc_auc = auc(fpr, tpr)
    #plt.title('ROC based on %s\nAUC = %0.2f' % (metric, roc_auc))
    plt.plot(fpr, tpr, c='coral', lw=4)
    #plt.plot([0, 1], [0, 1], c='dodgerblue', ls='--')
    #plt.ylabel('TPR');
    #plt.xlabel('FPR')
plt.show()
#     print('--------------------{}---{}------------'.format('autoEncoder',lsMethods[i]))
#
#     X_train, X_test, X_fraud = getTrainData(embedding)
#     # 读取模型
#     autoencoder = load_model('normEdge_{}_model.h5'.format(lsMethods[i]))
#
#     # 利用训练好的autoencoder重建测试集(X_test;X_fraud  前者是正常样本，后者是所有的异常样本)
#     pred_test = autoencoder.predict(X_test)
#     pred_fraud = autoencoder.predict(X_fraud)
#     print(pred_fraud)
#
#     # 计算还原误差MSE和MAE ；前者是loss；后者是metrics的mae
#     mse_test = np.mean(np.power(fixed_X_test - pred_test, 2), axis=1)
#     # print("mse_test")
#     # print(mse_test)
#     mse_fraud = np.mean(np.power(fixed_X_fraud - pred_fraud, 2), axis=1)
#     # print("mse_fraud")
#     # print(mse_fraud)
#     mae_test = np.mean(np.abs(fixed_X_test - pred_test), axis=1)
#     mae_fraud = np.mean(np.abs(fixed_X_fraud - pred_fraud), axis=1)
#     mse_df = pd.DataFrame()
#
#     mse_df['Class'] = [0] * len(mse_test) + [1] * len(mse_fraud)
#     # print(mse_df['Class'])
#     mse_df['MSE'] = np.hstack([mse_test, mse_fraud])
#     mse_df['MAE'] = np.hstack([mae_test, mae_fraud])
#     print(mse_df)
#     mse_df = mse_df.sample(frac=1).reset_index(drop=True)
#     print(mse_df)
#
#     """
#      sample（）参数frac是要返回的比例，比如df中有10行数据，我只想返回其中的30%,那么frac=0.3
#      set_index()和reset_oindex()的区别 前者为现有的dataframe设置不同于之前的index;而后者是还原和最初的index方式：0,1,2,3,4……
#     """
#
#     # 分别画出测试集中正样本和负样本的还原误差MAE和MSE
#     markers = ['o', '^']
#     colors = ['dodgerblue', 'coral']
#     labels = ['Non-fraud', 'Fraud']
#
#     plt.figure(figsize=(14, 5))
#     plt.subplot(121)
#     for flag in [1, 0]:
#         temp = mse_df[mse_df['Class'] == flag]
#         plt.scatter(temp.index,
#                     temp['MAE'],
#                     alpha=0.7,
#                     marker=markers[flag],
#                     c=colors[flag],
#                     label=labels[flag])
#     plt.title('Reconstruction MAE')
#     plt.ylabel('Reconstruction MAE');
#     plt.xlabel('Index')
#     plt.subplot(122)
#     for flag in [1, 0]:
#         temp = mse_df[mse_df['Class'] == flag]
#         plt.scatter(temp.index,
#                     temp['MSE'],
#                     alpha=0.7,
#                     marker=markers[flag],
#                     c=colors[flag],
#                     label=labels[flag])
#     plt.legend(loc=[1, 0], fontsize=12);
#     plt.title('Reconstruction MSE')
#     plt.ylabel('Reconstruction MSE');
#     plt.xlabel('Index')
#     plt.show()
#
#     # 画出Precision-Recall曲线
#     plt.figure(figsize=(14, 6))
#     for i, metric in enumerate(['MAE', 'MSE']):
#         plt.subplot(1, 2, i + 1)
#         precision, recall, _ = precision_recall_curve(mse_df['Class'], mse_df[metric])
#         pr_auc = auc(recall, precision)
#         plt.title('Precision-Recall curve based on %s\nAUC = %0.2f' % (metric, pr_auc))
#         plt.plot(recall[:-2], precision[:-2], c='coral', lw=4)
#         plt.xlabel('Recall');
#         plt.ylabel('Precision')
#     plt.show()
#
#     #  画出ROC曲线
#     plt.figure(figsize=(14, 6))
#     for i, metric in enumerate(['MAE', 'MSE']):
#         plt.subplot(1, 2, i + 1)
#         fpr, tpr, _ = roc_curve(mse_df['Class'], mse_df[metric])
#         roc_auc = auc(fpr, tpr)
#         plt.title('Receiver Operating Characteristic based on %s\nAUC = %0.2f' % (metric, roc_auc))
#         plt.plot(fpr, tpr, c='coral', lw=4)
#         plt.plot([0, 1], [0, 1], c='dodgerblue', ls='--')
#         plt.ylabel('TPR');
#         plt.xlabel('FPR')
#     plt.show()
#
#     # 画出MSE、MAE散点图
#     markers = ['o', '^']
#     colors = ['dodgerblue', 'coral']
#     labels = ['Non-fraud', 'Fraud']
#
#     plt.figure(figsize=(10, 5))
#     for flag in [1, 0]:
#         temp = mse_df[mse_df['Class'] == flag]
#         plt.scatter(temp['MAE'],
#                     temp['MSE'],
#                     alpha=0.7,
#                     marker=markers[flag],
#                     c=colors[flag],
#                     label=labels[flag])
#     plt.legend(loc=[1, 0])
#     plt.ylabel('Reconstruction RMSE');
#     plt.xlabel('Reconstruction MAE')
#     plt.show()
#
# #
# # # 画出Precision-Recall曲线
# # plt.figure(figsize=(14, 6))
# # for i, metric in enumerate(['MAE', 'MSE']):
# #     plt.subplot(1, 2, i+1)
# #     precision, recall, _ = precision_recall_curve(mse_df['Class'], mse_df[metric])
# #     pr_auc = auc(recall, precision)
# #     plt.title('Precision-Recall curve based on %s\nAUC = %0.2f'%(metric, pr_auc))
# #     plt.plot(recall[:-2], precision[:-2], c='coral', lw=4)
# #     plt.xlabel('Recall'); plt.ylabel('Precision')
# # plt.show()
#
#
