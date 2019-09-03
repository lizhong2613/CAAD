# coding=utf8
from math import pow
import config
from config import neg_table_size


def InitNegTable(edges):
    a_list, b_list = zip(*edges) #edge是元组的列表 这里是解压 得到节点A 和节点B的数组
    a_list = list(a_list)
    b_list = list(b_list)
    NEG_SAMPLE_POWER = config.NEG_SAMPLE_POWER
    node = a_list
    node.extend(b_list) # 实现一个已经存在的列表的扩展 这里可以理解为直接将 两个列表合在一起

    node_degree = {} #创建一个空的字典 下面的代码实现字典计数的功能
    for i in node:
        if i in node_degree:
            node_degree[i] += 1
        else:
            node_degree[i] = 1

    sum_degree = 0
    for i in node_degree.values():
        sum_degree += pow(i, 0.75)

    por = 0
    cur_sum = 0
    vid = -1
    neg_table = []
    degree_list = list(node_degree.values())#度的列表
    node_id = list(node_degree.keys())#key 的列表
    print("节点的总数目{}".format(str(len(node_id))))
    for i in range(neg_table_size):
        if ((i + 1) / float(neg_table_size)) > por:
            #print("i={},por= {}".format(str(i +1),str(por)))
            cur_sum += pow(degree_list[vid + 1], NEG_SAMPLE_POWER)   # pow x 的y 次方
            por = cur_sum / sum_degree
            vid += 1

        neg_table.append(node_id[vid])#每次添加一个节点
    print("负采样节点数目：")
    print(len(neg_table))
    return neg_table
