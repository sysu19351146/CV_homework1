from mindspore import nn, Tensor, Model,context
from mindspore.train.callback import Callback
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.nn import Accuracy
from dataloader import data_load
from model import Resnet,Net
import argparse
from matplotlib import pyplot as plt
import numpy as np
import math
# # custom callback function
# class StepLossAccInfo(Callback):
#     def __init__(self, model, train_dataset,eval_dataset,steps_train,steps_loss, steps_eval):
#         self.model = model
#         self.eval_dataset = eval_dataset
#         self.train_dataset = train_dataset
#         self.steps_loss = steps_loss
#         self.steps_eval = steps_eval
#         self.steps_train= steps_train

#     def step_end(self, run_context):
#         cb_params = run_context.original_args()
#         cur_epoch = cb_params.cur_epoch_num
#         cur_step = (cur_epoch-1)*1875 + cb_params.cur_step_num
#         self.steps_loss["loss_value"].append(str(cb_params.net_outputs))
#         self.steps_loss["step"].append(str(cur_step))
#         if cur_step % 125 == 0:
#             train_acc=self.model.eval(self.train_dataset, dataset_sink_mode=False)
#             valid_acc = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
#             self.steps_eval["step"].append(cur_step)
#             self.steps_eval["acc"].append(valid_acc["Accuracy"])
#             self.steps_train["step"].append(cur_step)
#             self.steps_train["acc"].append(train_acc["Accuracy"])
#             print("训练集准确率为: {} ".format(train_acc["Accuracy"]))
#             print("验证集准确率为: {} ".format(valid_acc["Accuracy"]))




# parser = argparse.ArgumentParser(description='Mindspore Resnet ')
# parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend','CPU'])
# parser.add_argument('--data_path', type=str, default="./cifar10")
# parser.add_argument('--model_path', type=str, default="./model")
# parser.add_argument('--model_type', type=str, default="lenet")
# parser.add_argument('--batch_size', type=int, default=64)
# parser.add_argument('--image_size', type=int, default=32)
# parser.add_argument('--epoches', type=int, default=1)
# parser.add_argument('--lr', type=float, default=1e-3)
# args = parser.parse_args()

# context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
# if args.device_target=='Ascend':
#     sink_mode = True
# else:
#     sink_mode = False


# net = Net()
# if args.model_type=="resnet":
#     net = Resnet(blocks=[3,4,6,3],in_size=args.image_size,in_ch=3,class_num=10)


# dataset_train=data_load(args.data_path,"train",args.image_size,args.batch_size)
# dataset_valid=data_load(args.data_path,"valid",args.image_size,args.batch_size)
# dataset_test=data_load(args.data_path,"test",args.image_size,args.batch_size)

# # 定义超参、损失函数及优化器
# optim = nn.Adam(params=net.trainable_params(), learning_rate=args.lr)
# loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

# # 输入训练轮次和数据集进行训练


# model = Model(net, loss_fn=loss, optimizer=optim, metrics={"Accuracy": Accuracy()} )

# # save the network model and parameters for subsequence fine-tuning
# config_ck = CheckpointConfig(save_checkpoint_steps=1, keep_checkpoint_max=40)
# # group layers into an object with training and evaluation features
# ckpoint_cb = ModelCheckpoint(prefix="checkpoint_resnet", directory=args.model_path, config=config_ck)

# steps_loss = {"step": [], "loss_value": []}
# steps_eval = {"step": [], "acc": []}
# steps_train = {"step": [], "acc": []}
# # collect the steps,loss and accuracy information
# step_loss_acc_info = StepLossAccInfo(model ,dataset_train,dataset_valid, steps_train,steps_loss, steps_eval)

# model.train(epoch=args.epoches, train_dataset=dataset_train,callbacks=[ckpoint_cb, LossMonitor(125), step_loss_acc_info], dataset_sink_mode=sink_mode)
# acc=model.eval(dataset_test,dataset_sink_mode=False)
# print("测试集准确率：{}".format(acc["Accuracy"]))
# plt.plot(steps_eval["step"],steps_eval["acc"])
# np.save("step.npy",steps_loss["step"])
# np.save("loss.npy",steps_loss["loss_value"])
# np.save("train_acc.npy",steps_train["acc"])
# np.save("val_acc.npy",steps_eval["acc"])

from mindspore import nn, Tensor, Model,context
from mindspore.train.callback import Callback
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.nn import Accuracy
from dataloader import data_load
from model import Resnet,Net
import argparse
from matplotlib import pyplot as plt
import numpy as np
from mindspore import load_checkpoint, load_param_into_net
# custom callback function
class StepLossAccInfo(Callback):
    def __init__(self, model, train_dataset,eval_dataset,steps_train,steps_loss, steps_eval):
        self.model = model
        self.eval_dataset = eval_dataset
        self.train_dataset = train_dataset
        self.steps_loss = steps_loss
        self.steps_eval = steps_eval
        self.steps_train= steps_train

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        cur_step = (cur_epoch-1)*750 + cb_params.cur_step_num
        self.steps_loss["loss_value"].append(str(cb_params.net_outputs))
        self.steps_loss["step"].append(str(cur_step))
        if cb_params.cur_step_num % 1 == 0:
            train_acc=self.model.eval(self.train_dataset, dataset_sink_mode=False)
            valid_acc = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
            self.steps_eval["step"].append(cur_step)
            self.steps_eval["acc"].append(valid_acc["Accuracy"])
            self.steps_train["step"].append(cur_step)
            self.steps_train["acc"].append(train_acc["Accuracy"])        
            print("epoch {}训练集准确率为: {} ".format(cur_epoch,train_acc["Accuracy"]))
            print("epoch {}验证集准确率为: {} ".format(cur_epoch,valid_acc["Accuracy"]))



def save_step(args,steps_loss,steps_eval,steps_train):
    np.save(args.model_type+"/step.npy",steps_loss["step"])
    np.save(args.model_type+"/loss.npy",steps_loss["loss_value"])
    np.save(args.model_type+"/train_acc.npy",steps_train["acc"])
    np.save(args.model_type+"/val_acc.npy",steps_eval["acc"])



if __name__=="__main__":
    #命令行参数
    parser = argparse.ArgumentParser(description='Mindspore Resnet ')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend','CPU'])
    parser.add_argument('--data_path', type=str, default="./cifar10")
    parser.add_argument('--model_path', type=str, default="./model")
    parser.add_argument('--model_type', type=str, default="resnet")
    parser.add_argument('--ckpt_path', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--epoches', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    
    
    #根据输入参数选择网络和训练方式
    net = Net()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    if args.device_target=='Ascend':
        sink_mode = True
    else:
        sink_mode = False
    if args.model_type=="resnet":
        net = Resnet(blocks=[3,4,6,3],in_size=args.image_size,in_ch=3,class_num=10)
    if args.ckpt_path!="":
        load_checkpoint(args.ckpt_path, net)
    
    
    #导入数据集
    dataset_train=data_load(args.data_path,"train",args.image_size,args.batch_size)
    dataset_valid=data_load(args.data_path,"valid",args.image_size,args.batch_size)
    dataset_test=data_load(args.data_path,"test",args.image_size,args.batch_size)
    
    # 定义超参、损失函数及优化器
    optim = nn.Adam(params=net.trainable_params(), learning_rate=args.lr,weight_decay=1e-4)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    
    # 输入训练轮次和数据集进行训练
    model = Model(net, loss_fn=loss, optimizer=optim, metrics={"Accuracy": Accuracy()} )
    
    
    #记录训练过程中的参数
    config_ck = CheckpointConfig(save_checkpoint_steps=125, keep_checkpoint_max=50)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_"+args.model_type, directory=args.model_path, config=config_ck)
    steps_loss = {"step": [], "loss_value": []}
    steps_eval = {"step": [], "acc": []}
    steps_train = {"step": [], "acc": []}
    step_loss_acc_info = StepLossAccInfo(model ,dataset_train,dataset_valid,dataset_test, steps_train,steps_loss, steps_eval)
    
    #开始训练
    model.train(epoch=args.epoches, train_dataset=dataset_train,callbacks=[ckpoint_cb, LossMonitor(1), step_loss_acc_info], dataset_sink_mode=sink_mode)
    
    #测试
    acc=model.eval(dataset_test,dataset_sink_mode=False)
    print("测试集准确率：{}".format(acc["Accuracy"]))
    #训练过程可视化保存
    save_step(args,steps_loss,steps_eval,steps_train)





numpy_path="lenet/"
step=np.load(numpy_path+"step.npy")
loss=np.load(numpy_path+"loss.npy")
train_acc=np.load(numpy_path+"train_acc.npy")
val_acc=np.load(numpy_path+"val_acc.npy")
train_acc2=np.load(numpy_path+"train_acc2.npy")
val_acc2=np.load(numpy_path+"val_acc2.npy")
loss2=np.load(numpy_path+"loss2.npy")
step = list(map(float, step))
#step = list(map(lambda x: math.log(x,2), step))
step = np.array(step)
loss = list(map(float, loss))
#loss = list(map(lambda x: math.log(x,2), loss))
loss = np.array(loss)

loss2 = list(map(float, loss2))
#loss = list(map(lambda x: math.log(x,2), loss))
loss2 = np.array(loss2)

step2=step+199250

step3=np.array([0.1]*200)
train_acc3=np.array([0.1]*200)
val_acc3=np.array([0.1]*200)
loss3=np.array([0.1]*200)
step3[:100]=step
step3[100:]=step2
loss3[:100]=loss
loss3[100:]=loss2
train_acc3[:100]=train_acc
train_acc3[100:]=train_acc2
val_acc3[:100]=val_acc
val_acc3[100:]=val_acc2
red_line=np.array((step3[100],step3[100]))
red_line2=np.array((step3[0],step3[0]))
y=np.array((0,0.8))
#loss = list(map(lambda x: math.log(x,2), loss))
# plt.plot(step,loss,"b")
# plt.plot(step2,loss2,"b")
# plt.plot(step,train_acc,"b",step2,train_acc2,"b")
# plt.plot(step,val_acc,"g",step2,val_acc2,"g")

plt.plot(step3,train_acc3,label="train")
plt.plot(step3,val_acc3,label="valid")
plt.plot(red_line,y,"r")
plt.plot(red_line2,y,"r")
plt.text(step3[0], 0.2, "lr=1e-3", size = 15, alpha = 0.8,color="r")
plt.text(step3[100], 0.2, "lr=1e-4", size = 15, alpha = 0.8,color="r")
plt.legend()
plt.show()