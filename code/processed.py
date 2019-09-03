import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def read_file(dataname):
    # read data
    d = pd.read_csv(dataname)

    # 查看样本比例
    num_nonfraud = np.sum(d['Class'] == 0)  # 0类样本表示非欺诈样本，是正常样本数据，该数据集中有284315条数据，占比较大。
    num_fraud = np.sum(d['Class'] == 1)     # 1类样本表示欺诈样本，是属于异常检测中的异常样本，该数据集中仅有492条数据，比例很小。
    plt.bar(['fraud', 'non_fraud'], [num_fraud, num_nonfraud], color='dodgerblue')
    plt.show()

    # 删除时间列，并对amount列进行标准化
    data = d.drop(['Time'], axis=1)
    data['Amount'] = StandardScaler().fit_transform(data[['Amount']])

    # 提取负样本(正常数据；label:0),并且按照8:2的比例切分成训练集合测试集。
    mask = (data['Class'] == 0)#这样是添加一个条件
    X_train, X_test = train_test_split(data[mask], test_size=0.2, random_state=920)
    X_train = X_train.drop(['Class'], axis=1).values
    X_test = X_test.drop(['Class'], axis=1).values

    # 提取所有正样本(异常数据，label：1)，作为测试集的一部分
    X_fraud = data[~mask].drop(['Class'], axis=1).values

    return X_train, X_test, X_fraud
