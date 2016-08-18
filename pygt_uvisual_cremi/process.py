from __future__ import print_function
import h5py
import numpy as np
from numpy import float32, int32, uint8, dtype
import sys

from cremi import Annotations, Volume
from cremi.io import CremiFile
import random

# Relative path to where PyGreentea resides
pygt_path = '../../PyGreentea'


import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), pygt_path))

# Other python modules
import math

# Load PyGreentea
import PyGreentea as pygt
import glob
from PIL import Image


test_net_file = 'net_test_softmax.prototxt'
test_device = 0

pygt.caffe.set_devices((test_device,))

caffemodels = pygt.getCaffeModels('net');

test_net = pygt.init_testnet(test_net_file, trained_model=caffemodels[-1][1], test_device=test_device)
'''
Input option 1
Load the datasets - hdf5
'''
'''
hdf5_raw_file = '../dataset_06/fibsem_medulla_7col/tstvol-520-1-h5/img_normalized.h5'
hdf5_gt_file = '../dataset_06/fibsem_medulla_7col/tstvol-520-1-h5/groundtruth_seg.h5'
# hdf5_aff_file - affinity ground truth. if you train with euclid, ignore this one
# hdf5_aff_file = '../dataset_06/fibsem_medulla_7col/tstvol-520-1-h5/groundtruth_aff.h5'

# read hdf5 files
hdf5_raw = h5py.File(hdf5_raw_file, 'r')
hdf5_gt = h5py.File(hdf5_gt_file, 'r')
# ignore for eucledian distance
# hdf5_aff = h5py.File(hdf5_aff_file, 'r')

# from this point, the data is in numpy arrays
hdf5_raw_ds = pygt.normalize(np.asarray(hdf5_raw[hdf5_raw.keys()[0]]).astype(float32), -1, 1)
hdf5_gt_ds = np.asarray(hdf5_gt[hdf5_gt.keys()[0]]).astype(float32)
# ignore for eucledian distance
# hdf5_aff_ds = np.asarray(hdf5_aff[hdf5_aff.keys()[0]]).astype(float32)
'''

###############################################################
'''
Input option 2
Load the datasets - individual tiff files in a directory
'''
# Load the datasets
'''
rawInputDir = '/home/thanuja/projects/data/dataset_01/train/raw';
labelInputDir = '/home/thanuja/projects/data/dataset_01/train/labels'

rawImagePath = sorted(glob.glob(rawInputDir+'/*.tif'))
print("rawImagePath: {0}".format(rawImagePath))
labelImagePath = sorted(glob.glob(labelInputDir+'/*.png'))
numFiles = len(rawImagePath)

raw_ds = [np.expand_dims(pygt.normalize(np.array(Image.open(rawImagePath[i]).convert('L'), 'f')),0) for i in range(0,1)]
gt_ds = [np.array(Image.open(labelImagePath[i]).convert('L'), 'f') for i in range(0,1)]
gt_ds_scaled = [np.expand_dims(np.floor(label/31),0) for label in gt_ds]
print(gt_ds_scaled[0].shape)
print(raw_ds[0].shape)
'''

'''
Input format: cremi challenge
'''
hdf5_fileName = '/home/thanuja/DATA/cremi/train/hdf/sample_A_20160501.hdf'
file = CremiFile(hdf5_fileName, "r")

# raw_ds = pygt.normalize(np.array(file.read_raw().data))
raw_ds = pygt.normalize(np.array(file.read_raw().data, dtype=float32))
# print(raw_ds[0].shape)

datasets = []
# for i in range(0,len(raw_ds)):
for i in range(0,3):
    dataset = {}
    dataset['data'] = np.expand_dims(raw_ds[i],0)
    # dataset['label'] = gt_ds_scaled[i]
    datasets += [dataset]

print(len(dataset)) # expected 125
print(datasets[0]['data'].shape) # expected (1,1250,1250)


pred_array = pygt.process(test_net, datasets)

#pygt.dump_tikzgraph_maps(test_net, 'dump')

outhdf5 = h5py.File('test_out.h5', 'w')
outdset = outhdf5.create_dataset('main', np.shape(pred_array), np.float32, data=pred_array)
outhdf5.close()

