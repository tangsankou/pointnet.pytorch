from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
from utils.io import IOStream, load_model, save_model


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
# parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

#add
parser.add_argument('--gpu_id', type=int, default=0, help='choose the gpu')
parser.add_argument('--exp_name', type=str, default='exp', metavar='N', help='Name of the experiment')
#end

opt = parser.parse_args()
# print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

# opt.manualSeed = random.randint(1, 10000)  # fix seed
opt.manualSeed = 8543
random.seed(opt.manualSeed)
# torch.manual_seed(opt.manualSeed)

#add
if not os.path.exists('checkpoints_cls'):
    os.makedirs('checkpoints_cls')
if not os.path.exists('checkpoints_cls/' + opt.exp_name):
    os.makedirs('checkpoints_cls/' + opt.exp_name)
if not os.path.exists('checkpoints_cls/' + opt.exp_name + '/' + 'models'):
    os.makedirs('checkpoints_cls/' + opt.exp_name + '/' + 'models')

io = IOStream('checkpoints_cls/' + opt.exp_name + '/run.log')
opt.cuda = torch.cuda.is_available()
if opt.cuda:
    torch.cuda.set_device(opt.gpu_id)
    io.cprint(
       'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    torch.cuda.manual_seed(opt.manualSeed)
else:
    io.cprint('Using CPU')
#end

io.cprint("Random Seed: ", opt.manualSeed)

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='trainval')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

# print("len(dataset):",len(dataset))
# print("len(test_dataset):", len(test_dataset))
num_classes = len(dataset.classes)
io.cprint('classes:', num_classes)

""" try:
    os.makedirs(opt.outf)
except OSError:
    pass """

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier = load_model(args, classifier)
    # classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
# classifier.cuda()

num_batch = len(dataset) / opt.batchSize
io.cprint("num_batch",num_batch)

for epoch in range(opt.nepoch):
    #scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        if i != 0 and i % 30 == 0:
            io.cprint('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))
    scheduler.step()

        if epoch % 9 == 0:
            total_correct = 0
            total_testset = 0
            classifier = classifier.eval()
            with torch.no_grad():
                j, data = enumerate(testdataloader, 0)
                points, target = data
                target = target[:, 0]
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                optimizer.step()
                pred, _, _ = classifier(points)
                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                total_correct += correct.item()
                total_testset += points.size()[0]
            io.cprint('[%d: ] %s loss: %f accuracy: %f' % (epoch, blue('test'), loss.item(), total_correct / float(total_testset)))

    # torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
    #add
    if epoch % 9 == 0:
        save_model(classifier, opt, 'model')
save_model(classifier, opt, 'model_final')
#end
    
""" total_correct = 0
total_testset = 0
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

io.cprint("final accuracy {}".format(total_correct / float(total_testset))) """