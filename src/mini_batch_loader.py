#!/usr/bin/env python2.7
# coding:utf-8
 
import os
import sys
import numpy as np
import cv2
import cPickle
import math
import time
from scipy import ndimage
import scipy.misc
 
class MiniBatchLoader(object):
 
    def __init__(self, args, train):
 
        self.data_dir = args.img_dir
        self.batch_size = args.batchsize
        self.classes = args.n_joints
        self.mean = np.array([113.970, 110.130, 103.804]) 
        self.in_size = args.inputRes
        self.outputRes = args.outputRes
        self.scale = args.scale
        self.rotate = args.rotate
        self.train = train
 
    def load_data(self, lines):
        mini_batch_size = self.batch_size
        in_channels = 3
        xs = np.zeros((mini_batch_size, in_channels,  self.in_size, self.in_size)).astype(np.float32)
        ys = np.zeros((mini_batch_size, self.classes, self.outputRes, self.outputRes)).astype(np.float32)
 
        for i, line in enumerate(lines):
            delete = []
            datum = line.split(',')
            img_fn = '%s%s' % (self.data_dir, datum[0])
            r = 0
            
            # read image & joint
            img = cv2.imread(img_fn)
            joints = np.asarray([int(float(p)) for p in datum[1:29]])
            joints = joints.reshape((len(joints) / 2, 2))
            
            if self.train:
                s = float(datum[-3])
                c = np.asarray([float(datum[-2]),float(datum[-1])])
                c[1] += 15*s

                # scale
                s *= 1.25
                s *= (2 ** self.rnd(self.scale))

                # rotate 
                if np.random.rand() > 0.6:
                    r = self.rnd(self.rotate)

                # flip, must apply at begining, or label in delete will be wrong
                if np.random.randint(2) == 1:
                    img, joints = self.flip(img, joints)
                    c[0] = img.shape[1] - c[0]

                for v in range(len(joints)):
	                if  joints[v,0]<0 or joints[v,1]<0 or joints[v,0]>img.shape[1] or joints[v,1]>img.shape[0]:
		                delete.append(v)

            else:
                h,w,_ = img.shape
                s = 0.8*h/200
                c = np.asarray([w/2, h/2])
                c[1] += 15*s
                s *= 1.25

            img, joints = self.crop(img, joints, c, s, r)
                
            if self.train:
                img[0] = np.clip(img[0]*np.random.uniform(0.6,1.4), 0, 255)
                img[1] = np.clip(img[1]*np.random.uniform(0.6,1.4), 0, 255)
                img[2] = np.clip(img[2]*np.random.uniform(0.6,1.4), 0, 255)


            xs[i, :, :, :] = ((img - self.mean)/255).transpose(2, 0, 1)

            # heatmap 
            ksize = 7
            joints_outRes = joints.copy().astype(np.int32)
            h, w, _ = img.shape
            joints_outRes[:,0] = joints_outRes[:,0] / float(w) * self.outputRes
            joints_outRes[:,1] = joints_outRes[:,1] / float(h) * self.outputRes
            heatmap = np.zeros((self.classes,self.outputRes,self.outputRes)).astype(np.float32)
            gaussian = self.gauss2D(ksize)
            for j in range(len(joints_outRes)):
                if j in delete:
                    continue            
                if joints_outRes[j,0]<0 or joints_outRes[j,1]<0 or \
                   joints_outRes[j,0] >= self.outputRes or joints_outRes[j,1] >= self.outputRes:
                    continue
                x = joints_outRes[j,0]-int(ksize/2)
                xp = joints_outRes[j,0]+int(ksize/2)+1
                y = joints_outRes[j,1]-int(ksize/2)
                yp = joints_outRes[j,1]+int(ksize/2)+1
                l = np.clip(x, 0, self.outputRes)
                r = np.clip(xp, 0, self.outputRes)
                u = np.clip(y, 0, self.outputRes)
                d = np.clip(yp, 0, self.outputRes)

                clipped = gaussian[u-y:ksize-(yp-d), l-x:ksize-(xp-r)]
                heatmap[j,u:d,l:r] = clipped

            ys[i, :, :, :] = heatmap

        return xs, ys


    def flip(self, img, joints):
        img = np.fliplr(img)
        joints[:,0] = img.shape[1] - joints[:,0]
        joints = list(zip(joints[:,0], joints[:,1]))

        joints[0], joints[5] = joints[5], joints[0] #ankle
        joints[1], joints[4] = joints[4], joints[1] #knee
        joints[2], joints[3] = joints[3], joints[2] #hip
        joints[6], joints[11] = joints[11], joints[6] #wrist
        joints[7], joints[10] = joints[10], joints[7] #elbow
        joints[8], joints[9] = joints[9], joints[8] #shoulder

        joints = np.array(joints).flatten()
        joints = joints.reshape((len(joints) / 2, 2))

        return img, joints

    def gauss2D(self,shape):
        if shape==7:
            h = np.array([[0.0529,  0.1197,  0.1954,  0.2301,  0.1954,  0.1197 , 0.0529],
                 [0.0529,  0.1197,  0.1954,  0.2301,  0.1954,  0.1197,  0.0529],
                 [0.1197,  0.2709,  0.4421,  0.5205,  0.4421,  0.2709,  0.1197],
                 [0.1954,  0.4421,  0.7214,  0.8494,  0.7214,  0.4421,  0.1954],
                 [0.2301,  0.5205,  0.8494,  1.0000,  0.8494,  0.5205,  0.2301],
                 [0.1954,  0.4421,  0.7214,  0.8494,  0.7214,  0.4421,  0.1954],
                 [0.1197,  0.2709,  0.4421,  0.5205,  0.4421,  0.2709,  0.1197],
                 [0.0529,  0.1197,  0.1954,  0.2301,  0.1954,  0.1197,  0.0529]])
        return h

    def rnd(self, x):
        return max(-2*x, min(2*x, np.random.randn()*x))

    def get_transform(self, center, scale, res, rot=0):
        # Generate transformation matrix
        h = 200 * scale
        t = np.zeros((3, 3))
        t[0, 0] = float(res[1]) / h
        t[1, 1] = float(res[0]) / h
        t[0, 2] = res[1] * (-float(center[0]) / h + .5)
        t[1, 2] = res[0] * (-float(center[1]) / h + .5)
        t[2, 2] = 1
        if not rot == 0:
            rot = -rot # To match direction of rotation from cropping
            rot_mat = np.zeros((3,3))
            rot_rad = rot * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
            rot_mat[2,2] = 1
            # Need to rotate around center
            t_mat = np.eye(3)
            t_mat[0,2] = -res[1]/2
            t_mat[1,2] = -res[0]/2
            t_inv = t_mat.copy()
            t_inv[:2,2] *= -1
            t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
        return t

    def transform(self, pt, center, scale, res, invert=0, rot=0):
        # Transform pixel location to different reference
        t = self.get_transform(center, scale, res, rot=rot)
        if invert:
            t = np.linalg.inv(t)
        new_pt = np.array([pt[0], pt[1], 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2].astype(int)

    def crop(self, img, joints, center, scale, rot):
        res = self.in_size

        ht, wd, _ = img.shape
        
        ul = np.array(self.transform([0, 0], center, scale, (res, res), 1, 0))
        br = np.array(self.transform([res,res], center, scale, (res, res), 1, 0))

        size = np.asarray([br[1] - ul[1], br[0] - ul[0]])
        pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
        if not rot == 0:
            ul -= pad
            br += pad

        # Range to fill new array
        new_x = max(0, -ul[0]), min(br[0], wd) - ul[0]
        new_y = max(0, -ul[1]), min(br[1], ht) - ul[1]
        # Range to sample from original image
        old_x = max(0, ul[0]), min(wd, br[0])
        old_y = max(0, ul[1]), min(ht, br[1])

        newImg = np.zeros((br[1] - ul[1], br[0] - ul[0],3))
        newImg[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]           

        joints = np.asarray([(j[0] - ul[0], j[1] - ul[1]) for j in joints])     

        if not rot == 0:
            newImg, joints = self.rot(newImg, joints, rot, size)
  
        orig_h, orig_w, _ = newImg.shape 
        joints[:,0] = joints[:,0] / float(orig_w) * res
        joints[:,1] = joints[:,1] / float(orig_h) * res
        newImg = cv2.resize(newImg, (res, res),interpolation=cv2.INTER_CUBIC)

        return newImg, joints



    def rot(self, img, joints, r, shape):
        h, w, _ = img.shape
        theta = np.radians(r)
        cos, sin = np.cos(theta), np.sin(theta)
        R = np.asarray([[cos,-sin],[sin,cos]])
        joints[:,0] = joints[:,0]-w/2
        joints[:,1] = joints[:,1]-h/2
        joints = np.dot(joints,R)
        joints[:,0] = joints[:,0]+w/2
        joints[:,1] = joints[:,1]+h/2
                                
        img = ndimage.rotate(img, r, reshape=True)
        joints[:,0] = joints[:,0]+(img.shape[1]-w)/2
        joints[:,1] = joints[:,1]+(img.shape[0]-h)/2

        h, w, _ = img.shape
        img = img[h/2-shape[0]/2:h/2 + shape[0]/2, w/2-shape[1]/2:w/2 + shape[1]/2]
        joints = np.asarray([(j[0] - (w/2-shape[1]/2), j[1] - (h/2-shape[0]/2)) for j in joints])      

        return img, joints
