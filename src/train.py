#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import cPickle as pickle
import logging
from mini_batch_loader import MiniBatchLoader
from chainer import serializers
from hourglass import Hourglass
from chainer import cuda, optimizers, Variable
import sys, os
import math
import time
import numpy as np
import opt
 
def test(args, model, test_dl):
    sum_accuracy = 0
    sum_loss     = 0

    mini_batch_loader = MiniBatchLoader(args, False)
    for i in range(0, test_data_size, args.batchsize):
        raw_x, raw_t = mini_batch_loader.load_data(test_dl[i:i+args.batchsize])
        x = Variable(cuda.to_gpu(raw_x))
        t = Variable(cuda.to_gpu(raw_t))
        model.train = False
        model(x, t)
        sum_loss += model.loss.data * args.batchsize
 
    msg = "test mean loss {a}".format(a=sum_loss/test_data_size)
    print(msg)
    sys.stdout.flush()
    logging.info(msg)
 
 
if __name__ == '__main__':

    args = opt.get_arguments()

    # log 
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    pickle_dump_path = args.result_dir + "epoch_{i}.model"
    fp = open(args.result_dir+'log', 'w')
    logging.basicConfig(filename=args.result_dir+'log', level=logging.DEBUG)
    
    # load dataset 
    train_dl = np.array([l.strip() for l in open(args.train_csv_fn).readlines()])
    test_dl = np.array([l.strip() for l in open(args.test_csv_fn).readlines()])
    train_data_size = len(train_dl)
    test_data_size = len(test_dl)

    mini_batch_loader = MiniBatchLoader(args, True)
 
    # load model 
    cuda.get_device(args.gpu).use()
    hourglass = Hourglass(args)
    
    # setup 
    hourglass = hourglass.to_gpu()
    optimizer = optimizers.Adam()
    #optimizer = optimizers.RMSprop(lr=2.5e-4)
    optimizer.setup(hourglass)
 
    # training 
    test(args, hourglass,test_dl)
    for epoch in range(1, args.epoch+1):
        st = time.time()
        sys.stdout.flush()
        indices = np.random.permutation(train_data_size)
        sum_accuracy = 0
        sum_loss     = 0
 
        for i in range(0, train_data_size, args.batchsize):
            sys.stdout.write('Iteration '+str(i)+'\r')
            sys.stdout.flush()
            r = indices[i:i+args.batchsize]
            raw_x, raw_y = mini_batch_loader.load_data(train_dl[r])
            x = Variable(cuda.to_gpu(raw_x))
            y = Variable(cuda.to_gpu(raw_y))
            hourglass.zerograds()
            hourglass.train = True
            loss = hourglass(x, y)
            loss.backward()
            optimizer.update()
 
            sum_loss += loss.data * args.batchsize
 
        end = time.time()
        msg = "epoch:{x} training loss:{a}, time {c}".format(x=epoch,a=sum_loss/train_data_size, c=end-st)
        print(msg)
        logging.info(msg)
        
        sys.stdout.flush()
         
        if epoch == 1 or epoch % args.snapshot == 0:
            pickle.dump(hourglass, open(pickle_dump_path.format(i=epoch), "wb"))
 
        if epoch == 1 or epoch % args.test_freq == 0:
            test(args, hourglass,test_dl)
     
