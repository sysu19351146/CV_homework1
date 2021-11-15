# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 08:11:45 2021

@author: 53412
"""
from mindspore import nn, Tensor, Model


#lenet5网络
class Net(nn.Cell):
    def __init__(self, num_class=10, num_channel=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5,weight_init="TruncatedNormal",pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5,weight_init="TruncatedNormal", pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120,weight_init="TruncatedNormal")
        self.fc2 = nn.Dense(120, 84,weight_init="TruncatedNormal")
        self.fc3 = nn.Dense(84, num_class,weight_init="TruncatedNormal")
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x



#resnet34网络
class Res_block(nn.Cell):
    def __init__(self,in_ch,out_ch,strides,padding,need_de):
        super(Res_block,self).__init__()
        self.in_ch=in_ch
        self.out_ch=out_ch
        self.need_de=need_de
        self.strides=strides
        self.conv1=nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=strides,weight_init="TruncatedNormal",pad_mode=padding)
        self.b1=nn.BatchNorm2d(out_ch)
        self.conv2=nn.Conv2d(out_ch,out_ch,kernel_size=3,stride=1,weight_init="TruncatedNormal",pad_mode=padding)
        self.b2 = nn.BatchNorm2d(out_ch)
        self.de=nn.Conv2d(self.in_ch,self.out_ch,1,stride=self.strides,weight_init="TruncatedNormal")
        self.re=nn.ReLU()
    def construct(self,x):
        y=self.conv1(x)
        y=self.b1(y)
        y=self.re(y)
        y=self.conv2(y)
        y=self.b2(y)
        if self.need_de:
            x=self.de(x)
            y=y+x
        else:
            y=y+x
        y = self.re(y)
        return y

class Resnet(nn.Cell):
    def __init__(self,blocks,in_size,in_ch,class_num):
        super(Resnet,self).__init__()
        self.blocks=blocks
        self.in_size=in_size
        self.filters=[64,128,256,512]
        self.conv1_=nn.Conv2d(in_ch,self.filters[0],kernel_size=3,stride=1,weight_init="TruncatedNormal",pad_mode="valid")
        self.b=nn.BatchNorm2d(self.filters[0])
        self.layer1=self.make_layer(blocks[0],self.filters[0],self.filters[0],False)
        self.layer2 = self.make_layer(blocks[1], self.filters[0], self.filters[1], True)
        self.layer3 = self.make_layer(blocks[2], self.filters[1], self.filters[2], True)
        self.layer4 = self.make_layer(blocks[3], self.filters[2], self.filters[3], True)
        self.avgpool=nn.AvgPool2d(1)
        self.fc=nn.Dense(int(self.filters[3]*in_size*in_size/8/8),class_num,weight_init="TruncatedNormal")
        self.re=nn.ReLU()
        self.flat=nn.Flatten()
    def make_layer(self,blocks,in_ch,out_ch,need_de):
        layers=[]
        for block in range(blocks):
            if need_de and block==0:
                layers.append(Res_block(in_ch,out_ch,strides=2,padding="same",need_de=True))
            elif block==0:
                layers.append(Res_block(in_ch, out_ch, strides=1, padding="same", need_de=False))
            else:
                layers.append(Res_block(out_ch,out_ch,strides=1,padding="same",need_de=False))
        return nn.SequentialCell(layers)

    def construct(self,x):
        out=self.re(self.b(self.conv1_(x)))
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.avgpool(out)
        out=self.flat(out)
        out=self.fc(out)
        return out