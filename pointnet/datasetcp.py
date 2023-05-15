from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement

def get_segmentation_classes(root):
    catfile = os.path.join(root, 'synsetoffset2category.txt')
    cat = {}
    meta = {}

    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]

    for item in cat:
        dir_seg = os.path.join(root, cat[item], 'points_label')
        dir_point = os.path.join(root, cat[item], 'points')
        fns = sorted(os.listdir(dir_point))
        meta[item] = []
        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0])
            meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))
    
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'w') as f:
        for item in cat:
            datapath = []
            num_seg_classes = 0
            for fn in meta[item]:
                datapath.append((item, fn[0], fn[1]))

            for i in tqdm(range(len(datapath))):
                l = len(np.unique(np.loadtxt(datapath[i][-1]).astype(np.uint8)))
                if l > num_seg_classes:
                    num_seg_classes = l

            print("category {} num segmentation classes {}".format(item, num_seg_classes))
            f.write("{}\t{}\n".format(item, num_seg_classes))

def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))

class ShapeNetDataset(data.Dataset):
    """ def __init__(self, root, npoints=2500, classification=False,
                 class_choice=None, split="train", data_augmentation=False):
        
        self.root = root           # 数据集路径
        self.npoints = npoints     # 采样点数
        self.data_augmentation = data_augmentation     # 是否使用法线信息
        self.category = {}         # 类别所对应文件夹
        # shapenet有16个大类，每个大类有一些部件，
        # 例如飞机 'Airplane': [0, 1, 2, 3] 其中标签为0 1 2 3 的四个小类都属于飞机这个大类
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
        
        
        # 读取 类别所对应的文件夹信息，即该文件synsetoffset2category.txt
        with open(self.root+"/synsetoffset2category.txt") as f:
            for line in f.readlines():
                cate,file = line.strip().split()
                self.category[cate] = file
        # print(self.category)   # {'Airplane': '02691156', 'Bag': '02773838', 'Cap': '02954340', 'Car': '02958343', 'Chair': '03001627', 'Earphone': '03261776', 'Guitar': '03467517', 'Knife': '03624134', 'Lamp': '03636649', 'Laptop': '03642806', 'Motorbike': '03790512', 'Mug': '03797390', 'Pistol': '03948459', 'Rocket': '04099429', 'Skateboard': '04225987', 'Table': '04379243'}
        
        # 将类别字符串与数字对应
        self.category2id = {}
        i = 0
        for item in self.category:
            self.category2id[item] = i
            i = i + 1
        
        
        # class_choice进行类别选择
        if class_choice:     # class_choice 是 list类型
            for item in self.category:
                if item not in class_choice:     # 若 类别 不在class_choice中，则删除
                    self.category.pop(item)
        
        
        # 存储类别对应的点云数据文件
        self.datapath = []           # 存储形式：[ (类别, 数据路径), (类别, 数据路径), ... ]
        
        # 遍历点云文件，进行存储
        for item in self.category:
            filesName = [f[:-4] for f in os.listdir(self.root+"/"+self.category[item])]    # 把该类别文件夹下的所有文件遍历出来，之后对其进行判断（属于训练集、验证集、测试集、）
            
            # 抓取部分数据（训练集、验证集、测试集）
            if split=="train":
                with open(self.root+"/"+"train_test_split"+"/"+"shuffled_train_file_list.json") as f:
                    filename = [f.split("/")[-1] for f in json.load(f)]
                    for file in filesName:
                        if file in filename:   # 若该类别文件夹中的数据在训练集中，则存储
                            self.datapath.append((item, self.root+"/"+self.category[item]+"/"+file+".txt"))
            elif split=="test":
                with open(self.root+"/"+"train_test_split"+"/"+"shuffled_test_file_list.json") as f:
                    filename = [f.split("/")[-1] for f in json.load(f)]
                    for file in filesName:
                        if file in filename:   # 若该类别文件夹中的数据在测试集中，则存储
                            self.datapath.append((item, self.root+"/"+self.category[item]+"/"+file+".txt"))
        
        
    def __getitem__(self, index):
        '''
            :return: 点云数据, 大类别, 每个点的语义（大类别中的小类别）
        '''
        cls = self.datapath[index][0]     # 类别字符串
        cls_index = self.category2id[cls] # 类被字符串所对应的数字
        path = self.datapath[index][1]    # 点云数据存储的路径
        data = np.loadtxt(path)           # 点云数据
        
        point_data = None  
        if self.normal_use:   # 是否使用法线信息
            point_data = data[:, 0:-1]
        else:
            point_data = data[:, 0:3]
        
        seg = data[:, -1]     # 语义信息
        
        
        # 对数据进行重新采样
        choice = np.random.choice(len(seg), self.npoints)
        point_data = point_data[choice, :]
        seg = seg[choice]
        
        if self.classification:
            return point_data, cls_index
        else:
            return point_data, seg
 
 
    def __len__(self):
        return len(self.datapath) """

    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        # print("self.cat:",self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        # print("self.cat when class_choice is not None:",self.cat)

        self.id2cat = {v: k for k, v in self.cat.items()}# 反过来
        # print(self.id2cat)

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        print("splitfile: ",splitfile)
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')# category:03001627
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid+'.pts'),
                                        os.path.join(self.root, category, 'points_label', uuid+'.seg')))
        """ for k,v in self.meta.items():
            print("dict:",k,v)
            break 
        for category in self.cat.keys():
            print("len(self.meta(class))",len(self.meta[category])) """
        # print("len(self.meta): ",len(self.meta))#16
        # print("len(cat): ",len(self.cat))#16
        self.datapath = []
        for item in self.cat:#Airplane
            for fn in self.meta[item]:#fn[0]:points,fn[1]:points_label
                self.datapath.append((item, fn[0], fn[1]))
        # print("self.datapath[0]",self.datapath[0])
        print("len(self.datapath): ",len(self.datapath))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print("self.classes:",self.classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        # print("self.seg_classes and num_seg_classes:",self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]#fn[0]:class name,fn[1]:points path,fn[2]points_label path
        # print("type(fn):",type(fn))#tuple
        cls = self.classes[self.datapath[index][0]]#classification class label 0-15
        # print("type(cls):",type(cls))#int
        point_set = np.loadtxt(fn[1]).astype(np.float32)#array(n,3)
        seg = np.loadtxt(fn[2]).astype(np.int64)#array(n,)
        # print("point_set[0][0] and seg[0]:",point_set[1258], seg[1258])

        choice = np.random.choice(len(seg), self.npoints, replace=True)#0-len(seg)选取npoints个点，replace=True:可重复
        # print("type(choice):",type(choice))#np.array
        #resample
        point_set = point_set[choice, :]#采样

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # 去中心化
        #计算到原点的最远距离
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        #归一化
        point_set = point_set / dist #scale

        #默认False  开启旋转任意角度并加上一个bias,增强数据的抗干扰能力
        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation随机旋转;[:,[0,2]]第0列和第2列
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter随机抖动

        seg = seg[choice]#array(2500,)，采样后的
        # print("seg.shape:",seg.shape)
        point_set = torch.from_numpy(point_set)#转换成tensor
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))#cls为对应的代号,比如Airplane对应0
        print("cls:",cls)
        # print("type(seg):",type(seg))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)

class ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.fns.append(line.strip())

        self.cat = {}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        print(self.cat)
        self.classes = list(self.cat.keys())

    def __getitem__(self, index):
        fn = self.fns[index]
        cls = self.cat[fn.split('/')[0]]
        with open(os.path.join(self.root, fn), 'rb') as f:
            plydata = PlyData.read(f)
        pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls


    def __len__(self):
        return len(self.fns)

if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]

    if dataset == 'shapenet':
        d = ShapeNetDataset(root = datapath, class_choice = ['Chair'])
        print("seglen(d):",len(d))
        ps, seg = d[0]
        print("ps.s ps.t and seg.s seg.t",ps.size(), ps.type(), seg.size(),seg.type())

        d = ShapeNetDataset(root = datapath, classification = True)
        print("classlen(d):",len(d))
        ps, cls = d[0]
        print("ps.s ps.t and cls.s cls.t",ps.size(), ps.type(), cls.size(),cls.type())
        # get_segmentation_classes(datapath)

    if dataset == 'modelnet':
        gen_modelnet_id(datapath)
        d = ModelNetDataset(root=datapath)
        print(len(d))
        print(d[0])

