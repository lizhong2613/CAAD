# -*- coding: utf-8 -*-
import random
import argparse
import config
## 本文件实现将传入的数据按照比例进行切分的工作

print("prepare Data")

#打开对应文件夹下面的描述文件  其中graph中存储的为边的链接描述信息 data文件中为节点的描述信息 zhihu 为中文描述
f = open('../datasets/%s/graph.txt' % 'cora', 'rb')
# 对应边的描述文件
edges = [i for i in f]
ratioSet = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
for i in ratioSet:
    #需要训练的边的比例
    selected = int(len(edges) * float(i))
    selected = selected - selected % config.batch_size
    selected = random.sample(edges, selected)#sample(list, k)返回一个长度为k新列表，新列表存放list所产生k个随机不重复的元素
    remain = [i for i in edges if i not in selected]
    s1 = 'temp/graph{}.txt'.format(str(i*100))
    s2 = 'temp/test_graph{}.txt'.format(str(i*100))
    print(s1)
    print(s2)
    fw1 = open(s1, 'wb') # 选中的边写入
    fw2 = open(s2, 'wb')#剩下的边写入

    for i in selected:
        fw1.write(i)
    for i in remain:
        fw2.write(i)
    fw1.close()
    fw2.close()
print("prepare Data ended")