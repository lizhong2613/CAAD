# -*- coding: utf-8 -*-
import random
import numpy as np
import os
def naive_auc(labels,preds):
    """
    最简单粗暴的方法
　　　先排序，然后统计有多少正负样本对满足：正样本预测值>负样本预测值, 再除以总的正负样本对个数
     复杂度 O(NlogN), N为样本数
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    total_pair = n_pos * n_neg
    print(total_pair)

    labels_preds = zip(labels,preds)
    labels_preds = sorted(labels_preds,key=lambda x:x[1])
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(len(labels_preds)):
        if labels_preds[i][0] == 1:
            satisfied_pair += accumulated_neg
        else:
            accumulated_neg += 1

    return satisfied_pair / float(total_pair)
node2vec = {}
# dataset_name = "zhihu"
f = open('temp/embed.txt', 'rb')
for i, j in enumerate(f):
    if j.decode() != '\n':
        node2vec[i] = list(map(float, j.strip().decode().split(' ')))
f1 = open(os.path.join('temp/anomalyedge3.txt'), 'rb')
edges = [list(map(int, i.strip().decode().split(' '))) for i in f1]
nodes = list(set([i for j in edges for i in j]))
a = 0
b = 0
fanomaly = open(os.path.join('temp/anomalyedge4.txt'), 'wb')
for i, j in edges:
    if i in node2vec.keys() and j in node2vec.keys():
        dot1 = np.dot(node2vec[i], node2vec[j])
        random_node = random.sample(nodes, 1)[0]
        while random_node == j or random_node not in node2vec.keys():
            random_node = random.sample(nodes, 1)[0]
        dot2 = np.dot(node2vec[i], node2vec[random_node])
        if dot1 > dot2:
            a += 1
        elif dot1 == dot2:
            a += 0.5
        # if dot1 < dot2:
        #     fanomaly.write("{} {}\n".format(str(i), str(j)).encode())
        b += 1
fanomaly.close()
print("Auc value:", float(a) / b)
