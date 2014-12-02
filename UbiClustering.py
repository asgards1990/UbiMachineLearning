# -*- coding: utf-8 -*-
"""
Created on Mon Dec 01 12:24:59 2014

@author: su.yang
"""
print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


np.random.seed(42)

digits = load_digits()
data = scale(digits.data)