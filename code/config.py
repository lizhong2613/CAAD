# -*- coding: utf-8 -*-
MAX_LEN = 300
neg_table_size = 1000000 #负采样表的大小
NEG_SAMPLE_POWER = 0.75
batch_size = 32 #批大小
num_epoch = 200 # 全数据集训练次数
#num_epoch = 2 # 全数据集训练次数
embed_size = 200 #嵌入的大小？
lr = 1e-3
# 那 batch epoch iteration代表什么呢？
# （1）batchsize：批大小。在深度学习中，一般采用SGD训练，即每次训练在训练集中取batchsize个样本训练；
# （2）iteration：1个iteration等于使用batchsize个样本训练一次；
# （3）epoch：1个epoch等于使用训练集中的全部样本训练一次，通俗的讲epoch的值就是整个数据集被轮几次。
