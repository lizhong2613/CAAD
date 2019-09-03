# -*- coding: utf-8 -*-
import random
import argparse
import config

print("prepare Data")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d')
parser.add_argument('--gpu', '-g')
parser.add_argument('--ratio', '-r')
args = parser.parse_args()
#打开对应文件夹下面的描述文件  其中graph中存储的为边的链接描述信息 data文件中为节点的描述信息 zhihu 为中文描述
f = open('../datasets/%s/graph.txt' % args.dataset, 'rb')
# 对应边的描述文件
edges = [i for i in f]
#需要训练的边的比例
selected = int(len(edges) * float(args.ratio))
selected = selected - selected % config.batch_size
selected = random.sample(edges, selected)#sample(list, k)返回一个长度为k新列表，新列表存放list所产生k个随机不重复的元素
remain = [i for i in edges if i not in selected]
fw1 = open('temp/graph.txt', 'wb') # 选中的边写入
fw2 = open('temp/test_graph.txt', 'wb')#剩下的边写入

for i in selected:
    fw1.write(i)
for i in remain:
    fw2.write(i)
print("prepare Data ended")