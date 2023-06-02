import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import json

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = torch.zeros(batch_data.shape)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        # print("rotation_angle:",rotation_angle)
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = torch.from_numpy(np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])).to(torch.float32)
        # print("rotation_matrix:",rotation_matrix)
        shape_pc = batch_data[k, ...]
        # print("shape_pc(batch_data[k, ...]):",shape_pc)
        # print(rotated_data[0][0][0].dtype,shape_pc[0][0].dtype,rotation_matrix[0][0].dtype)
        rotated_data[k, ...] = torch.mm(shape_pc.reshape((-1, 3)), rotation_matrix)
        print("rd:",type(rotated_data))
        # print("rotated_data[k, ...]:",rotated_data[k,...])
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = torch.zeros(batch_data.shape)
    # rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi #?
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = torch.from_numpy(np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])).to(torch.float32)
        """ rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]]) """
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = torch.mm(shape_pc.reshape((-1, 3)), rotation_matrix)
        # rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * torch.randn(B, N, C), -1*clip, clip)
    # jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    # print("jittered_data:",jittered_data)
    jittered_data += batch_data
    return jittered_data

if __name__ == '__main__':
    x=torch.rand(2, 4, 3)
    # x=np.random.random((2, 4, 3))
    print("x:",x)
    # y=jitter_point_cloud(x)
    y = rotate_point_cloud_by_angle(x, 0.3)
    print("y:",y)