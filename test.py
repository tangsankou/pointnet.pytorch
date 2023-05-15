import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import json

a =np.array([[[1,2,3],[4,5,6]],[[7,8,9][10,11,12]],[[2,4,6],[1,3,5]]])
print(a[:,[0,2]])

""" fname_train = '/root/pointnet.pytorch/scripts/shapenetcore_partanno_segmentation_benchmark_v0/train_test_split/shuffled_train_file_list.json' 
filelist_train = json.load(open(fname_train, 'r'))
i=0
for file in filelist_train:
    i=i+1
    _, category, uuid = file.split('/')
print(i)

fname_test = '/root/pointnet.pytorch/scripts/shapenetcore_partanno_segmentation_benchmark_v0/train_test_split/shuffled_test_file_list.json' 
filelist_test = json.load(open(fname_test, 'r'))
j=0
for file in filelist_test:
    j=j+1
    __, categoryy, uuidd = file.split('/')
print(j) """

""" x = torch.rand(6,4,4)
print("x.size:",x.size())
print("x:",x)
x=x.topk(2, dim=2, largest=True, sorted=False)[0]
# y=x.topk(2,dim=2,largest=False,sorted=False)[0]
print("x.size:",x.size())
print("x=:",x)
# print("y.size:",y.size())
# print("y=:",y)

a,b= torch.chunk(x,2,dim=2)
d = x.view(-1, 8)
# print("a.size:",a.size())#(6,4,1)
# print("a=:",a)
# print("b.size:",b.size())#(6,4,1)
# print("b=:",b)
c = torch.cat([a,b],dim=1)#(6,8,1)
print("c.size:",c.size())
print("c=",c)
c=c.view(-1,8)
print("c.view.size:",c.size())
print("c.view=",c)

print("d.size:",d.size())
print("d:",d)

 values0=x.topk(k=1, dim=0, sorted=False)[0]
print("values0.size:",values0.size())
print("values0=:",values0)

values1=x.topk(k=1, dim=1, sorted=False)[0]
print("values1.size:",values1.size())
print("values1=:",values1)

values2=x.topk(k=1, dim=2, sorted=False)[0]
print("values2.size:",values2.size())
print("values2=:",values2) 
print("no problem") """