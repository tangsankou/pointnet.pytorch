from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

# opt.manualSeed = random.randint(1, 10000)  # fix seed
opt.manualSeed = 5323
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice])
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

test_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))#shapenet 2658 704 cap 39 11
num_classes = dataset.num_seg_classes
print('classes', num_classes)#shapenet chair 4 cap 2
try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'

classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize#2658/32
print("num_batch:",num_batch)

for epoch in range(opt.nepoch):
#scheduler.step()  
    for i, data in enumerate(dataloader, 0):#当前batch的第i个样本数据
    # dataloader迭代遍历数据（每次遍历一个batch）
        points, target = data#(32,2500,3) (32,2500)
        # print("points.size:",points.size())
        # print("target.size:",target.size())
        points = points.transpose(2, 1)#为什么要交换dim 1和2(32,3,2500)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)#pred:(32,2500,4)chair
        # print("pred.size:",pred.size())
        pred = pred.view(-1, num_classes)#(80000,4)
        # print("pred1.size:",pred.size())
        target = target.view(-1, 1)[:, 0] - 1#(80000)32*2500
        # print("target1.size:",target.size())
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]#按照dim=1？行这个维度求最大值，返回索引
        # print("pred_choice.size:",pred_choice.size())#80000
        correct = pred_choice.eq(target.data).cpu().sum()#预测正确的个数
        # print("correct.type:",type(correct))
        # print("correct:",correct.data)
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item()/float(opt.batchSize * 2500)))

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))#next返回一个元组
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            # print("pred.size:",pred.size())#(11,2500,2)cap:len(test)=11
            # print("pred.data.size:",pred.data.size())
            
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1)[:, 0] - 1
            # print("target.data.size:",target.data.size())
            loss = F.nll_loss(pred, target)
            # print("testpred1.data.size:",pred.data.size())#cap:(27500,2)11*2500
            pred_choice = pred.data.max(1)[1]
            # print("testpred_choice.data.size:",pred_choice.data.size())#cap:(27500)
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize * 2500)))   
    torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))
scheduler.step()
    
## benchmark mIOU
""" shape_ious = []
# a=0
for i,data in tqdm(enumerate(testdataloader, 0)):
    # a=a+1
    points, target = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(2)[1]#为什么这里取max(2)，而上面是max(1)

    pred_np = pred_choice.cpu().data.numpy()
    # print("pred_np.type:",type(pred_np))
    # print("pred_np.size",pred_np.shape)#cap:(11,2500) chair(32,2500)
    target_np = target.cpu().data.numpy() - 1
    # print("target_np.type:",type(target_np))
    # print("target_np.size",target_np.shape)#cap:(11,2500) chair(32,2500)

    # print(target_np.shape[0])#这是什么，好像是测试集长度
    # j=0
    for shape_idx in range(target_np.shape[0]):#cap：外层遍历11
        parts = range(num_classes)#np.unique(target_np[shape_idx])cap:0到10
        part_ious = []
        for part in parts:#遍历类别，cap：内层遍历2
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            j=j+1
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    # print("j:",j)#chair:128 4*32 cap:22 2*11 num_classes*(len(test)/batchsize)

# print("a:",a)#chair:704/32 cap:11/32?取一次a=1
print("mIOU for class {}: {}".format(opt.class_choice, np.mean(shape_ious)))
"""

for epoch in range(opt.nepoch):
    checkpoint=torch.load('%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))
    # print("type(checkpoint):",type(checkpoint))
    # for i in checkpoint:
        # print(i)
    model = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)
    model.load_state_dict(checkpoint)
    model.cuda()
    shape_ious = []
    for i,data in enumerate(testdataloader, 0):
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        model = model.eval()
        pred, _, _ = model(points)
        # print("mioupred.data.size:",pred.data.size())#(11,2500,2)
        pred_choice = pred.data.max(2)[1]#为什么这里取max(2)，而上面是max(1)
        # print("mioupred_choice.data.size:",pred_choice.data.size())#(11,2500)
        pred_np = pred_choice.cpu().data.numpy()
        target_np = target.cpu().data.numpy() - 1
    
        for shape_idx in range(target_np.shape[0]):
            parts = range(num_classes)#np.unique(target_np[shape_idx])
            part_ious = []
            for part in parts:
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                if U == 0:
                    iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
                else:
                    iou = I / float(U)
                part_ious.append(iou)
            shape_ious.append(np.mean(part_ious))
    print("mIOU for class {} in epoch {}: {}".format(opt.class_choice, epoch, np.mean(shape_ious)))