#!/usr/bin/env python
# coding:utf-8
 
import chainer
 
class Add(chainer.Function):
    def forward_cpu(self, inputs):
        x, y, z = inputs
        w = x + y + z
        return w,
 
    def backward_cpu(self, inputs, grad_outputs):
        x, y, z = inputs
        gw, = grad_outputs
        gx = gw
        gy = gw
        gz = gw
        return gx, gy, gz
 
    def forward_gpu(self, inputs):
        x, y, z = inputs
        w = chainer.cuda.elementwise(
            'float32 x, float32 y, float32 z',
            'float32 w',
            'w = x + y + z',
            'add_fwd')(x, y, z)
 
        return w,
 
    def backward_gpu(self, inputs, grad_outputs):
        x, y, z = inputs
        gw, = grad_outputs
        gx, gy, gz = chainer.cuda.elementwise(
            'float32 x, float32 y, float32 z, float32 gw',
            'float32 gx, float32 gy, float32 gz',
            '''
                gx = gw;
                gy = gw;
                gz = gw;
            ''',
            'add_bwd')(x, y, z, gw)
        return gx, gy, gz
 
def add(x, y, z):
    return Add()(x, y, z)
