from __future__ import print_function
import h5py
import numpy as np
from numpy import float32, int32, uint8, dtype
import sys
from PIL import Image
import glob

from cremi import Annotations, Volume
from cremi.io import CremiFile
import random

# Relative path to where PyGreentea resides
pygt_path = '../../PyGreentea'


import sys, os
sys.path.append(os.path.join(os.path.dirname('__file__'), pygt_path))

# Other python modules
import math
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# Load PyGreentea
import PyGreentea as pygt


# npv = np.version.version
# print("numpy version: {0}".format(npv))

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

'''
Input format: cremi challenge
'''
'''
maxNumImagesToTrain = 100
hdf5_fileName_original = '/home/thanuja/DATA/cremi/train/hdf/sample_A_20160501.hdf'
hdf5_fileName_membranes = '/home/thanuja/DATA/cremi/train/hdf/sample_A_20160501_membranes.hdf'
rawfile = CremiFile(hdf5_fileName_original, "r")
labelfile = CremiFile(hdf5_fileName_membranes, "r")
#raw_ds = pygt.normalize(np.array(rawfile.read_raw().data))
raw_ds = pygt.normalize(np.array(rawfile.read_raw().data, dtype=float32))
# label_ds is a np array with neuronIDs as elements. 0 is for membranes. 
# rescale labels s.t. all non zero labels are given 1.
label_ds = 1*(np.array(labelfile.read_neuron_ids().data)>1)

print('max label:')
print(np.max(label_ds))
print('min label:')
print(np.min(label_ds))
'''

###############################################################
'''
Input option 2
Load the datasets - individual tiff files in a directory
'''
maxNumImagesToTrain = 30
rawInputDir = '/home/thanuja/DATA/ISBI2012/train-volume'
labelInputDir = '/home/thanuja/DATA/ISBI2012/train-labels'

rawImagePath = sorted(glob.glob(rawInputDir+'/*.tif'))
print("rawImagePath: {0}".format(rawImagePath))
labelImagePath = sorted(glob.glob(labelInputDir+'/*.tif'))
numFiles = len(rawImagePath)

# raw_ds = [np.expand_dims(pygt.normalize(np.array(Image.open(rawImagePath[i]).convert('L'), 'f')),0) for i in range(0,numFiles)]
raw_ds = [np.expand_dims(pygt.normalize(np.array(Image.open(rawImagePath[i]).convert('L'), 'f')),0) for i in range(0,numFiles)]
label_ds = [np.expand_dims(np.floor(np.array(Image.open(labelImagePath[i]).convert('L'), 'f')/255),0) for i in range(0,numFiles)]
# gt_ds_scaled = [np.expand_dims(np.floor(label/31),0) for label in gt_ds]
'''
print(label_ds[0].shape)
print(raw_ds[0].shape)

print(np.maximum(raw_ds[0]))
print(np.minimum(raw_ds[0]))
print(np.maximum(label_ds[0]))
print(np.minimum(label_ds[0]))
'''
print("len(raw_ds): " + str(len(raw_ds)))
print("len(label_ds): " + str(len(label_ds)))

print("raw_ds[0].shape: " + str(raw_ds[0].shape))
print("label_ds[0].shape: " + str(label_ds[0].shape))

print("raw max: " + str(np.amax(raw_ds[0])))
print("raw min: " + str(np.amin(raw_ds[0])))
print("label max: " + str(np.amax(label_ds[0])))
print("label min: " + str(np.amin(label_ds[0])))

# plt.imshow(label_ds[0][0,:,:])
# plt.show()

'''
print("os.listdir(rawInputDir):{0}".format(os.listdir(rawInputDir)))
raw_arrays =  []
print("Start william's log message...")
print("numFiles: {0}".format(numFiles))
for i in range(numFiles):
    path = rawImagePath[i]
    image_object = Image.open(path)
    image_object.convert('L')
    image_array = np.array(image_object, 'f')
    print(i, image_array.shape, image_array.dtype, image_object.size, image_object.info)
    raw_arrays.append(image_array)
hdf5_raw_ds = np.array(raw_arrays)
print(hdf5_raw_ds.shape, hdf5_raw_ds.dtype)
print("Stop william's log message.")
'''
'''
print("raw_ds.shape = {0}".format(raw_ds.shape))
print("label_ds.shape = {0}".format(label_ds.shape))
# print("hdf5_gt_ds.shape = {0}".format(gt_ds.shape))
'''

###############################################################
'''
Input option 3
Load the datasets - multipage tiff containing all the images
'''
'''
tiff_raw_dir = '/home/thanuja/DATA/ISBI2012/train-volume/'
tiff_gt_dir = '/home/thanuja/DATA/ISBI2012/train-labels/'
# ignore for eucledian distance
tiff_aff_file = ''
# read all files into 3D numpy arrays
tiff_raw = Image.open(tiff_raw_file)
tiff_gt = Image.open(tiff_gt_file)
tiff_aff = Image.open(tiff_aff_file)
# read into np arrays
# tiff
'''
###############################################################
'''
datasets = []
# for i in range(hdf5_raw_ds.shape[1]):
for i in range(hdf5_raw_ds.shape[0]):
    # print(i)
    dataset = {}
    dataset['data'] = hdf5_raw_ds[None, i, :]
    dataset['components'] = hdf5_gt_ds[None, i, :]
    # ignore for eucledian distance
    # dataset['label'] = hdf5_aff_ds[0:3, i, :]
    dataset['nhood'] = pygt.malis.mknhood2d()
    datasets += [dataset]
'''

###############################
# create dataset for training
###############################



print('Using {} images out of the available {} images'.format(maxNumImagesToTrain,len(raw_ds)))

datasets = []
for i in range(0,min(len(raw_ds),maxNumImagesToTrain)):
    dataset = {}
    # dataset['data'] = np.expand_dims(raw_ds[i],0)
    # dataset['label'] = np.expand_dims(label_ds[i],0)
    dataset['data'] = raw_ds[i]
    dataset['label'] = label_ds[i]
    datasets += [dataset]

#test_dataset = {}
#test_dataset['data'] = hdf5_raw_ds
#test_dataset['label'] = hdf5_aff_ds

print(len(datasets)) # expected 30
print(datasets[0]['data'].shape) # expected (1,512,512)
print(datasets[0]['label'].shape) # expected (1,512,512)


# Set train options
class TrainOptions:
    loss_function = "softmax"
    loss_snapshot = 100
    test_interval = 4000
    scale_error = False
    train_device = 0
    test_device = 0
    test_net=None #'net_test.prototxt'

options = TrainOptions()

# Set solver options
solver_config = pygt.caffe.SolverParameter()
solver_config.train_net = 'net_train_softmax.prototxt'
solver_config.base_lr = 0.00005
solver_config.momentum = 0.99
solver_config.weight_decay = 0.000005
solver_config.lr_policy = 'inv'
solver_config.gamma = 0.0001
solver_config.power = 0.75
solver_config.max_iter = 20000
solver_config.snapshot = 2000
solver_config.snapshot_prefix = 'net'
solver_config.type = 'Adam'
solver_config.display = 1

# Set devices
# pygt.caffe.enumerate_devices(False)
pygt.caffe.set_devices((options.train_device,))

solverstates = pygt.getSolverStates(solver_config.snapshot_prefix);

# First training method
if (len(solverstates) == 0 or solverstates[-1][0] < solver_config.max_iter):
    solver, test_net = pygt.init_solver(solver_config, options)
    if (len(solverstates) > 0):
        solver.restore(solverstates[-1][1])
    pygt.train(solver, test_net, datasets, [], options)


   
