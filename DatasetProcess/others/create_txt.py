# !/usr/bin/python
# -*- coding: utf-8 -*-
import os
import random

trainval_percent = 0.9  # 验证集+训练集占总比例多少
train_percent = 0.9  # 训练数据集占验证集+训练集比例多少
jsonfilepath = 'images/total2019'
txtsavepath = './'
total_xml = os.listdir(jsonfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open('./trainval2019.txt', 'w')
ftest = open('./test2019.txt', 'w')
ftrain = open('./train2019.txt', 'w')
fval = open('./val2019.txt', 'w')

for i in list:
    #name = total_xml[i][:-5] + '\n'
    name = total_xml[i] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
