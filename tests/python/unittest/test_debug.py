from __future__ import print_function
import numpy as np
import mxnet as mx
import math
import random
import itertools
from numpy.testing import assert_allclose, assert_array_equal
from mxnet.test_utils import *
from mxnet.base import py_str, MXNetError
from common import setup_module, with_seed
import unittest
import time
import cPickle

def check_concat_with_shape():

    with open('d1.pkl', 'r') as f:
        d1 = cPickle.load(f)
    with open('d2.pkl', 'r') as f:
        d2 = cPickle.load(f)
    with open('d3.pkl', 'r') as f:
        d3 = cPickle.load(f)
    with open('d4.pkl', 'r') as f:
        d4 = cPickle.load(f)
    with open('d5.pkl', 'r') as f:
        d5 = cPickle.load(f)

    inputs = [mx.symbol.Variable('cls_prob'), mx.symbol.Variable('bbox_pred'), mx.symbol.Variable('im_info'), mx.symbol.Variable('gt_boxes'), mx.symbol.Variable('valid_ranges')]
    sym = mx.symbol.MultiProposalTarget(*inputs, name='multi_proposal_target')
    d1 = mx.nd.array(d1, ctx=mx.cpu(0))
    d2 = mx.nd.array(d2, ctx=mx.cpu(0))
    d3 = mx.nd.array(d3, ctx=mx.cpu(0))
    d4 = mx.nd.array(d4, ctx=mx.cpu(0))
    d5 = mx.nd.array(d5, ctx=mx.cpu(0))

    exe = sym.bind(ctx=mx.cpu(0), args={'cls_prob': d1, 'bbox_pred': d2, 'im_info': d5, 'gt_boxes': d3, 'valid_ranges': d4})    
    import time
    t1 = time.time()
    outputs = exe.forward(is_train=True)
    print (outputs[1])
    t2 = time.time()
    print (t2 - t1)

    #import pdb
    #pdb.set_trace()
    
    print('Done!')

if __name__=='__main__':
	check_concat_with_shape()
