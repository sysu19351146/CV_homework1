# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 08:37:29 2021

@author: 53412
"""

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import numpy as np
import mindspore.dataset.vision.c_transforms as CV
import mindspore
from mindspore import nn, Tensor
from mindspore import dtype as mstype


def data_load(data_path,data_type,image_size,batch_size):
    #数据集路径
    DATA_DIR = data_path+"/"+data_type
    
    
    #数据集基本参数
    resize_height = image_size
    resize_width = image_size
    rescale = 1.0 / 255.0
    shift = 0.0
    mindspore.dataset.config.set_seed(1000)
    
    # 数据增强
    random_crop_op = CV.RandomCrop((32, 32), (4, 4, 4, 4)) # padding_mode default CONSTANT
    random_horizontal_op = CV.RandomHorizontalFlip()
    resize_op = CV.Resize((resize_height, resize_width)) # interpolation default BILINEAR
    rescale_op = CV.Rescale(rescale, shift)
    normalize_op = CV.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    changeswap_op = CV.HWC2CHW()

    c_trans = []
    if data_type=="train":
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [rescale_op,resize_op, normalize_op, changeswap_op]


    # 构建数据集
    dataset = ds.Cifar10Dataset(DATA_DIR)

    # 数据类型转换
    type_cast_op_image = C.TypeCast(mstype.float32)
    type_cast_op_label = C.TypeCast(mstype.int32)
    HWC2CHW = CV.HWC2CHW()
    dataset = dataset.map(operations=c_trans, input_columns="image")
    dataset = dataset.map(operations=type_cast_op_label, input_columns="label")
    dataset = dataset.batch(batch_size)
    dataset=  dataset.shuffle(buffer_size=10)
    dataset = dataset.repeat(count=1)
    
    return dataset

