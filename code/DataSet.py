# -*- coding: utf-8 -*-

import sys
import config

import numpy as np
from tensorflow.contrib import learn   #TensorFlow高层次机器学习API
from negativeSample import InitNegTable #负采样相关
import random


class dataSet:
    def __init__(self, text_path, graph_path):

        text_file, graph_file = self.load(text_path, graph_path)

        self.edges = self.load_edges(graph_file)

        self.text, self.num_vocab, self.num_nodes = self.load_text(text_file)

        self.negative_table = InitNegTable(self.edges)

    #加载文件函数 返回内容
    def load(self, text_path, graph_path):
        text_file = open(text_path, 'rb').readlines()
        for a in range(0, len(text_file)):
            text_file[a] = str(text_file[a])
        graph_file = open(graph_path, 'rb').readlines()

        return text_file, graph_file
    #加载边函数
    def load_edges(self, graph_file):
        edges = []
        for i in graph_file:
            edges.append(list(map(int, i.strip().decode().split('\t'))))

        print("Total load %d edges." % len(edges))

        return edges

    def load_text(self, text_file):
        print("正在创建字典")
        vocab = learn.preprocessing.VocabularyProcessor(config.MAX_LEN) #创建一个字典   参数是文档的最大长度。如果文本的长度大于最大长度，那么它会被剪切，反之则用0填充
        text = np.array(list(vocab.fit_transform(text_file))) #token化
        num_vocab = len(vocab.vocabulary_)
        print("num_vocab" + str(num_vocab))
        num_nodes = len(text)
        print("num_nodes" + str(num_nodes))
        vocab_dict = vocab.vocabulary_._mapping
        print(vocab_dict)

        return text, num_vocab, num_nodes
    #这个在每一个batch中产生负样本
    def negative_sample(self, edges):
        node1, node2 = zip(*edges)
        sample_edges = []
        func = lambda: self.negative_table[random.randint(0, config.neg_table_size - 1)]
        for i in range(len(edges)):
            neg_node = func()
            while node1[i] == neg_node or node2[i] == neg_node:
                neg_node = func()
            sample_edges.append([node1[i], node2[i], neg_node])

        return sample_edges # 生成一个边的列表 其中包含每一条边和一个负采样元素
    # by lzh  产生n个负采样元素
    def negNnodes(self,node1,node2,N):

        neg_sample = []
        func = lambda: self.negative_table[random.randint(0, config.neg_table_size - 1)]


        for m in range(N):
            neg_node = func()
            while node1== neg_node or node2 == neg_node:
                neg_node = func()
            neg_sample.append(neg_node)
        return neg_sample  # 生成一个边的列表 其中包含每一条边和一个负采样元素



    def generate_batches(self, mode=None):

        num_batch = len(self.edges) // config.batch_size # 向下取整数
        edges = self.edges
        # if mode == 'add':
        #     num_batch += 1
        #     edges.extend(edges[:(config.batch_size - len(self.edges) // config.batch_size)])
        if mode != 'add':
            random.shuffle(edges) #shuffle() 方法将序列的所有元素随机排序。
        sample_edges = edges[:num_batch * config.batch_size]  # 去除掉冗余的元素
        sample_edges = self.negative_sample(sample_edges)

        batches = []
        for i in range(num_batch):
            batches.append(sample_edges[i * config.batch_size:(i + 1) * config.batch_size])
        # print sample_edges[0]
        return batches
