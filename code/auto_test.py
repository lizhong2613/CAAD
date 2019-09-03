from processed import read_file
from keras.models import load_model
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import seed
from prepareMyData import getTrainData
seed(1)

dataname = './creditcard.csv'
#X_train, X_test, X_fraud = read_file(dataname)
X_train, X_test, X_fraud = getTrainData()

# 读取模型
autoencoder = load_model('normEdge_line_model.h5')

# 利用训练好的autoencoder重建测试集(X_test;X_fraud  前者是正常样本，后者是所有的异常样本)
pred_test = autoencoder.predict(X_test)
pred_fraud = autoencoder.predict(X_fraud)
print(pred_fraud)

# 计算还原误差MSE和MAE ；前者是loss；后者是metrics的mae
mse_test = np.mean(np.power(X_test - pred_test, 2), axis=1)
#print("mse_test")
#print(mse_test)
mse_fraud = np.mean(np.power(X_fraud - pred_fraud, 2), axis=1)
#print("mse_fraud")
#print(mse_fraud)
mae_test = np.mean(np.abs(X_test - pred_test), axis=1)
mae_fraud = np.mean(np.abs(X_fraud - pred_fraud), axis=1)
mse_df = pd.DataFrame()

mse_df['Class'] = [0]*len(mse_test) + [1]*len(mse_fraud)
#print(mse_df['Class'])
mse_df['MSE'] = np.hstack([mse_test, mse_fraud])
mse_df['MAE'] = np.hstack([mae_test, mae_fraud])
print(mse_df)
mse_df = mse_df.sample(frac=1).reset_index(drop=True)
print(mse_df)

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