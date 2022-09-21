import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from nltk.util import ngrams
import dpkt
import math
import DataCollection_IDS
import warnings
warnings.filterwarnings(action='ignore')

path = '/home/hbb/pythonProject/CNN_LSTM_AE_Unknow_Protocol_Classification/IDS17_v3/'
class_list = ['dns', 'ftp', 'http', 'smb']






