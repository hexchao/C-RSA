import torchvision as tv
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
import random
from torch import optim
from torch.autograd  import Variable


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

device = 'cuda:0'
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = (0.5,0.5,0.5),std = (0.5,0.5,0.5))])
testset = tv.datasets.CIFAR10('../data1/',train=False,download=True,transform=transform)#10000
trainset = tv.datasets.CIFAR10(root='../data1/',train = True,download=True,transform=transform)#50000

Node = 10
Label = 10
batchsize = 10
stepsize = 0.001
penal = 0.001
regu = 0.01
mmm = 0.9
Q_ratio = 0.5
iteration = 100
criterion  = nn.CrossEntropyLoss()#定义交叉熵损失函数


#测试集数据
data_test_temp = []
label_test_temp = []
for i in range(len(testset)):
    data_test_temp.append(testset[i][0])
    label_test_temp.append(testset[i][1])

data_test = []
label_test = []
for i in range(int(len(data_test_temp)/batchsize)):
    data_batchtemp = []
    label_batchtemp = []
    for bs in range(batchsize):
        data_batchtemp.append(data_test_temp[i*batchsize+bs].numpy().tolist())
        label_batchtemp.append(label_test_temp[i*batchsize+bs])
    data_test.append(torch.tensor(data_batchtemp, device=device, requires_grad=False))
    label_test.append(torch.tensor(np.array(label_batchtemp), device=device, requires_grad=False))



#数据在生成时已经做了混合打乱，程序主体中每一轮迭代相当于做无放回抽样
data_len = len(trainset)
data_temp = [[] for _ in range(Label)]
label_temp = [[] for _ in range(Label)]
for i in range(data_len):
    label_temp[trainset[i][1]].append(trainset[i][1])
    data_temp[trainset[i][1]].append(trainset[i][0])

#——————iid数据集——————
#先从每个标签中随机抽样放进每个节点里
data_iid_temp = [[] for _ in range(Node)]
label_iid_temp = [[] for _ in range(Node)]
rand_index = list(range(len(data_temp[0])))#每个标签下的数据个数，默认5000
random.shuffle(rand_index)
for k in range(Node):
    for l in range(Label):
        for j in range(data_len // (Label*Node)):#250
            data_iid_temp[k].append(data_temp[l][rand_index[j + k*(data_len//(Label*Node))]].to(device))
            label_iid_temp[k].append(label_temp[l][rand_index[j + k*(data_len//(Label*Node))]])
#打乱，按batchsize分组打包
data_iid = [[] for _ in range(Node)]
label_iid = [[] for _ in range(Node)]
for k in range(Node):
    rand_index = list(range(len(data_iid_temp[0])))#依次取随机index的十个段
    random.shuffle(rand_index)
    for j in range(len(data_iid_temp[0])//batchsize):#组数
        data_batchtemp = []
        label_batchtemp = []
        for bs in range(batchsize):
            data_batchtemp.append(data_iid_temp[k][rand_index[j*batchsize+bs]][None])
            label_batchtemp.append(label_iid_temp[k][rand_index[j*batchsize+bs]])
        data_iid[k].append(torch.cat(data_batchtemp, dim=0))
        label_iid[k].append(torch.from_numpy(np.array(label_batchtemp)).to(device))

#——————noniid数据集——————
#从每个标签中随机抽样放进每个节点里
data_noniid_temp = [[] for _ in range(Node)]
label_noniid_temp = [[] for _ in range(Node)]
#先放一半
for k in range(Node):
    for j in range(len(data_temp[0])//2):#2500
        data_noniid_temp[k].append(data_temp[k][j].to(device))
        label_noniid_temp[k].append(label_temp[k][j])
    #再放一半
data_noniid_mix = []
label_noniid_mix = []
for k in range(Node):
    for j in range(len(data_temp[0])//2):              #先混合
        data_noniid_mix.append(data_temp[k][len(data_temp)//2+j])
        label_noniid_mix.append(label_temp[k][len(data_temp)//2+j])
rand_index = list(range(len(data_noniid_mix)))      #打乱
random.shuffle(rand_index)
for k in range(10):
    for i in range(len(data_temp[0])//2):               #分配
        data_noniid_temp[k].append(data_noniid_mix[rand_index[(len(data_temp[0])//2)*k + i]].to(device))
        label_noniid_temp[k].append(label_noniid_mix[rand_index[(len(label_temp[0])//2)*k + i]])

#打乱，按batchsize分组打包
data_noniid = [[] for _ in range(Node)]
label_noniid = [[] for _ in range(Node)]
for k in range(Node):
    rand_index = list(range(len(data_noniid_temp[0])))#依次取随机index的十个段,5000
    random.shuffle(rand_index)
    for j in range(len(data_noniid_temp[0])//batchsize):#组数
        data_batchtemp = []
        label_batchtemp = []
        for bs in range(batchsize):
            data_batchtemp.append(data_noniid_temp[k][rand_index[j*batchsize+bs]][None])
            label_batchtemp.append(label_noniid_temp[k][rand_index[j*batchsize+bs]])
        data_noniid[k].append(torch.cat(data_batchtemp, dim=0))
        label_noniid[k].append(torch.from_numpy(np.array(label_batchtemp)).to(device))




#Predefine
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = torch.nn.MaxPool2d(kernel_size = 3,stride = 2)
        self.conv2 = torch.nn.Conv2d(64,64,5)
        self.fc1 = torch.nn.Linear(64*4*4,384)
        self.fc2 = torch.nn.Linear(384,192)
        self.fc3 = torch.nn.Linear(192,10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def Accuracy(net, data_test, label_test):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(data_test)):
            images = data_test[i]
            labels = label_test[i]
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1) # 第一个变量不重要，用-代替
            correct +=(predicted == labels).sum()      # 更新正确分类的图片的数量
            total += len(data_test[i])                  # 更新测试图片的数量
    return correct/total



def randk_w(wk,w0):
    w_size = w0.size()
    w1 = wk-w0
    w1 = w1.view([-1])
    ratio = Q_ratio
    paralength = len(w1)
    index_probs = torch.ones(paralength) * (1-ratio)
    index_distriution = torch.distributions.bernoulli.Bernoulli(index_probs)
    index = index_distriution.sample().to(device)
    w1 = w1.mul(index)
    w1 = w1.view(w_size)
    w1 = w1.sign()
    return w1

def randk_b(wk,w0):
    w1 = wk-w0
    ratio = Q_ratio
    paralength = len(w1)
    index_probs = torch.ones(paralength) * (1-ratio)
    index_distriution = torch.distributions.bernoulli.Bernoulli(index_probs)
    index = index_distriution.sample().to(device)
    w1 = w1.mul(index)
    w1 = w1.sign()
    return w1



#Initialization
net0 = Net().to(device)

net = []
for k in range(Node):
    net.append(Net().to(device))
for k in range(Node):
    net[k].load_state_dict(net0.state_dict())

optimizer0 = optim.SGD(net0.parameters(),lr = stepsize,momentum=mmm)
optimizer = []
for k in range(Node):
    optimizer.append(optim.SGD(net[k].parameters(),lr = stepsize,momentum=mmm))


#开始训练
outputs = []
loss = []
running_loss = []
inputs = []
labels = []
accuracy_list = []
sign_fc1_w = []
sign_fc2_w = []
sign_fc3_w = []
sign_conv1_w = []
sign_conv2_w = []
sign_fc1_b = []
sign_fc2_b = []
sign_fc3_b = []
sign_conv1_b = []
sign_conv2_b = []
for k in range(Node):
    outputs.append([])
    loss.append([])
    inputs.append([])
    labels.append([])
    sign_fc1_w.append([])
    sign_fc2_w.append([])
    sign_fc3_w.append([])
    sign_conv1_w.append([])
    sign_conv2_w.append([])
    sign_fc1_b.append([])
    sign_fc2_b.append([])
    sign_fc3_b.append([])
    sign_conv1_b.append([])
    sign_conv2_b.append([])
steps = len(data_iid[0])
for epoch in range(iteration):
    rand_i = list(range(steps))
    random.shuffle(rand_i)
    for i in range(steps):#500
        for k in range(Node):
            inputs[k] = data_iid[k][rand_i[i]]
            labels[k] = label_iid[k][rand_i[i]]
            #——————————noniid——————————
            #inputs[k] = data_noniid[k][rand_i[i]]
            #labels[k] = label_noniid[k][rand_i[i]]
        optimizer0.zero_grad()
        for k in range(Node):
            optimizer[k].zero_grad()
            outputs[k] = net[k](inputs[k])
            loss[k] = criterion(outputs[k], labels[k])
            loss[k].backward()
        #先用旧的计算（epoch）
        net_fc1_weight_grad = []
        net_fc2_weight_grad = []
        net_fc3_weight_grad = []
        net_conv1_weight_grad = []
        net_conv2_weight_grad = []
        net_fc1_bias_grad = []
        net_fc2_bias_grad = []
        net_fc3_bias_grad = []
        net_conv1_bias_grad = []
        net_conv2_bias_grad = []
        #renew grad: weight&bias
        for k in range(Node):
            sign_fc1_w[k] = randk_w(net[k].fc1.weight.data.clone(),net0.fc1.weight.data.clone())
            sign_fc2_w[k] = randk_w(net[k].fc2.weight.data.clone(),net0.fc2.weight.data.clone())
            sign_fc3_w[k] = randk_w(net[k].fc3.weight.data.clone(),net0.fc3.weight.data.clone())
            sign_conv1_w[k] = randk_w(net[k].conv1.weight.data.clone(),net0.conv1.weight.data.clone())
            sign_conv2_w[k] = randk_w(net[k].conv2.weight.data.clone(),net0.conv2.weight.data.clone())
            sign_fc1_b[k] = randk_b(net[k].fc1.bias.data.clone(),net0.fc1.bias.data.clone())
            sign_fc2_b[k] = randk_b(net[k].fc2.bias.data.clone(),net0.fc2.bias.data.clone())
            sign_fc3_b[k] = randk_b(net[k].fc3.bias.data.clone(),net0.fc3.bias.data.clone())
            sign_conv1_b[k] =  randk_b(net[k].conv1.bias.data.clone(),net0.conv1.bias.data.clone())
            sign_conv2_b[k] =  randk_b(net[k].conv2.bias.data.clone(),net0.conv2.bias.data.clone())
        for k in range(Node):
            net_fc1_weight_grad.append(penal * sign_fc1_w[k])
            net_fc2_weight_grad.append(penal * sign_fc2_w[k])
            net_fc3_weight_grad.append(penal * sign_fc3_w[k])
            net_conv1_weight_grad.append(penal * sign_conv1_w[k])
            net_conv2_weight_grad.append(penal * sign_conv2_w[k])
            net_fc1_bias_grad.append(penal * sign_fc1_b[k])
            net_fc2_bias_grad.append(penal * sign_fc2_b[k])
            net_fc3_bias_grad.append(penal * sign_fc3_b[k])
            net_conv1_bias_grad.append(penal *sign_conv1_b[k])
            net_conv2_bias_grad.append(penal *sign_conv2_b[k])
        net0_fc1_weight_grad = regu * net0.fc1.weight.data.clone() + penal * torch.stack([(-1)*sign_fc1_w[k] for k in range(10)],2).sum(axis=2)
        net0_fc2_weight_grad = regu * net0.fc2.weight.data.clone() + penal * torch.stack([(-1)*sign_fc2_w[k] for k in range(10)],2).sum(axis=2)
        net0_fc3_weight_grad = regu * net0.fc3.weight.data.clone() + penal * torch.stack([(-1)*sign_fc3_w[k] for k in range(10)],2).sum(axis=2)
        net0_conv1_weight_grad = regu * net0.conv1.weight.data.clone() + penal * torch.stack([(-1)*sign_conv1_w[k] for k in range(10)],4).sum(axis=4)
        net0_conv2_weight_grad = regu * net0.conv2.weight.data.clone() + penal * torch.stack([(-1)*sign_conv2_w[k] for k in range(10)],4).sum(axis=4)
        net0_fc1_bias_grad = regu * net0.fc1.bias.data.clone() + penal * torch.stack([(-1)*sign_fc1_b[k] for k in range(10)],1).sum(axis=1)
        net0_fc2_bias_grad = regu * net0.fc2.bias.data.clone() + penal * torch.stack([(-1)*sign_fc2_b[k] for k in range(10)],1).sum(axis=1)
        net0_fc3_bias_grad = regu * net0.fc3.bias.data.clone() + penal * torch.stack([(-1)*sign_fc3_b[k] for k in range(10)],1).sum(axis=1)
        net0_conv1_bias_grad = regu * net0.conv1.bias.data.clone() + penal * torch.stack([(-1)*sign_conv1_b[k] for k in range(10)],1).sum(axis=1)
        net0_conv2_bias_grad = regu * net0.conv2.bias.data.clone() + penal * torch.stack([(-1)*sign_conv2_b[k] for k in range(10)],1).sum(axis=1)
        #再更新（epoch+1）
        for k in range(Node):
            net[k].fc1.weight.grad += net_fc1_weight_grad[k]
            net[k].fc2.weight.grad += net_fc2_weight_grad[k]
            net[k].fc3.weight.grad += net_fc3_weight_grad[k]
            net[k].conv1.weight.grad += net_conv1_weight_grad[k]
            net[k].conv2.weight.grad += net_conv2_weight_grad[k]
            net[k].fc1.bias.grad += net_fc1_bias_grad[k]
            net[k].fc2.bias.grad += net_fc2_bias_grad[k]
            net[k].fc3.bias.grad += net_fc3_bias_grad[k]
            net[k].conv1.bias.grad += net_conv1_bias_grad[k]
            net[k].conv2.bias.grad += net_conv2_bias_grad[k]
        net0.fc1.weight.grad = net0_fc1_weight_grad
        net0.fc2.weight.grad = net0_fc2_weight_grad
        net0.fc3.weight.grad = net0_fc3_weight_grad
        net0.conv1.weight.grad = net0_conv1_weight_grad
        net0.conv2.weight.grad = net0_conv2_weight_grad
        net0.fc1.bias.grad = net0_fc1_bias_grad
        net0.fc2.bias.grad = net0_fc2_bias_grad
        net0.fc3.bias.grad = net0_fc3_bias_grad
        net0.conv1.bias.grad = net0_conv1_bias_grad
        net0.conv2.bias.grad = net0_conv2_bias_grad
        # 对所有蹭的梯度做聚合替换
        optimizer0.step()
        for k in range(Node):
            optimizer[k].step()
    accur = Accuracy(net0, data_test, label_test)
    accuracy_list.append(accur.cpu().numpy().tolist())
    print('net0',epoch+1,accuracy_list[epoch])
print("----------finished training---------")



pd.DataFrame(accuracy_list).to_csv('./accuracy_list_CRSA1000.csv')



