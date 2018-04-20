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
def check_concat_with_shape():
    inputs = [mx.symbol.Variable('input')]
    sym = mx.symbol.Debug(*inputs, name='debug')
    x_nd = mx.nd.array([[[[1, 6, 4, 2]]]], ctx=mx.cpu(0))
    #data_npy = np.random.normal(size=(10,20,3,3))
    exe = sym.bind(ctx=mx.cpu(0), args={'input': x_nd})
    
    outputs = exe.forward(is_train=True)
    print('Done!')

if __name__=='__main__':
	check_concat_with_shape()