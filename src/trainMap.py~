#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import cPickle as pickle
from mini_batch_loader_revert import MiniBatchLoader
from chainer import serializers
from myfcn import MyFcn
from chainer import cuda, optimizers, Variable
import sys
import numpy as np
import scipy.io as sio
#from skimage import measure
import cv2

TEST_BATCH_SIZE = 1
IMAGE_DIR_PATH  = "data/LSP/images/"

if __name__ == '__main__':

    test_fn = "data/LSP/test_joints.csv"
    test_dl = np.array([l.strip() for l in open(test_fn).readlines()])

    mini_batch_loader = MiniBatchLoader(IMAGE_DIR_PATH, TEST_BATCH_SIZE, MyFcn.IN_SIZE)

    # get model
    myfcn = pickle.load(open('result/rot40/myfcn_epoch_200.model', 'rb'))
    myfcn = myfcn.to_gpu()

    sum_accuracy = 0
    sum_loss     = 0
    test_data_size = 1000
    ests = np.zeros((test_data_size, 14, 2)).astype(np.float32)

    for i in range(0, test_data_size, TEST_BATCH_SIZE):
        raw_x, raw_t, crop = mini_batch_loader.load_data(test_dl[i:i+TEST_BATCH_SIZE])
        x = Variable(cuda.to_gpu(raw_x))
        t = Variable(cuda.to_gpu(raw_t))
        myfcn.train = False
        pred = myfcn(x, t)
        sum_loss     += myfcn.loss.data * TEST_BATCH_SIZE
        #sum_accuracy += myfcn.accuracy * TEST_BATCH_SIZE

        #_/_/_/ max location _/_/_/
        hmap = cuda.to_cpu(pred.data[0])
        joints = np.zeros((14,2))
        for j in range(1,15):
            one_joint_map = hmap[j,:,:]
            maxi = np.argmax(one_joint_map)
            joints[j-1,:] = np.unravel_index(maxi,(224,224))

        #_/_/_/ revert to original coordinate _/_/_/
        joints[:,0] = joints[:,0] * crop[3] / 224
        joints[:,1] = joints[:,1] * crop[2] / 224
        joints[:,0] = joints[:,0] + crop[1]
        joints[:,1] = joints[:,1] + crop[0]
        ests[i,:,:] = joints
        
        '''
        joints[:,[0,1]] = joints[:,[1,0]]
        joints = joints.astype(np.int32)
        joints = [tuple(p) for p in joints]
        img = cv2.imread('data/LSP/images/'+test_dl[i].split(',')[0])
        for j, joint in enumerate(joints):
            cv2.circle(img, joint, 5, (0, 0, 255), -1)
        cv2.imwrite('mat/'+str(i)+'.jpg',img)
        '''

    sio.savemat('mat/ests.mat', {'ests':ests})
 
    print("test mean loss {a}, accuracy {b}".format(a=sum_loss/test_data_size, b=test_data_size))
    sys.stdout.flush()



    
