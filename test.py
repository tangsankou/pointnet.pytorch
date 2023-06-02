import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import json

batchsize = 5
x=torch.rand(batchsize,9)
print("x:",x)
iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
print("iden.shape:",iden.shape)
print("iden:",iden)
x=x+iden
print("x+iden:",x)
x=x.view(-1,3,3)
print("x.view:",x)

""" k=5
x=torch.rand(1,3,8)
print(type(x))
print("x:",x)
y=x.topk(k, dim=2, largest=True, sorted=False)[0]
print("y:",y)
z=torch.mean(x.topk(k, dim=2, largest=True, sorted=False)[0], dim=2)
print("z:",z) """

""" names = ["A","B","C","D","E","F","G","H","I","J","K"]
for epoch in range(10):
    print("-----epoch------:",epoch)
    for index,value in enumerate(names,0):
        print(f'{index}: {value}')
        if index%3 ==0:
            print("---next---")
            i,v = next(enumerate(names,0))
            print(f'{i}: {v}')
            print("---end next---") """

""" a =np.array([[[1,2,3],[4,5,6]],[[7,8,9][10,11,12]],[[2,4,6],[1,3,5]]])
print(a[:,[0,2]])

fname_train = '/root/autodl-tmp/shapenetcore_partanno_segmentation_benchmark_v0/train_test_split/shuffled_train_file_list.json' 
filelist_train = json.load(open(fname_train, 'r'))
i=0
for file in filelist_train:
    i=i+1
    _, category, uuid = file.split('/')
print("len(train):",i)#12137

fname_test = '/root/autodl-tmp/shapenetcore_partanno_segmentation_benchmark_v0/train_test_split/shuffled_test_file_list.json' 
filelist_test = json.load(open(fname_test, 'r'))
j=0
for file in filelist_test:
    j=j+1
    _, category, uidd = file.split('/')
print("len(test):",j)#2874 """

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