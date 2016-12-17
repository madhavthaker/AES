from __future__ import print_function
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.stats import spearmanr
import nltk.data
import re
from nltk.corpus import stopwords
import tensorflow as tf
import logging
from gensim.models import word2vec
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc
from time import time
from matplotlib.pyplot import cm 

