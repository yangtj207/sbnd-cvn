"""
This is the dataset generator module.
"""

__version__ = '1.0'
__author__ = 'Saul Alonso-Monsalve'
__email__ = "saul.alonso.monsalve@cern.ch"

import numpy as np
import glob
import ast
import ntpath
import pickle
import configparser
import logging
import sys
import time
import random
import zlib

from sklearn.utils import class_weight
from collections import Counter

'''
****************************************
************** PARAMETERS **************
****************************************
'''

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

config = configparser.ConfigParser()
config.read('config/config.ini')

# random

SEED = int(config['random']['seed'])

if SEED == -1:
    SEED = int(time.time())

np.random.seed(SEED)

# images

IMAGES_PATH = config['images']['path']
VIEWS = int(config['images']['views'])
PLANES = int(config['images']['planes'])
CELLS = int(config['images']['cells'])

# dataset

DATASET_PATH = config['dataset']['path']
PARTITION_PREFIX = config['dataset']['partition_prefix']
LABELS_PREFIX = config['dataset']['labels_prefix']
UNIFORM = ast.literal_eval(config['dataset']['uniform'])

# model

OUTPUTS = int(config['model']['outputs'])

# train

TRAIN_FRACTION = float(config['train']['fraction'])
WEIGHTED_LOSS_FUNCTION = ast.literal_eval(config['train']['weighted_loss_function'])
CLASS_WEIGHTS_PREFIX = config['train']['class_weights_prefix']

# validation

VALIDATION_FRACTION = float(config['validation']['fraction'])

# test

TEST_FRACTION = float(config['test']['fraction'])

if((TRAIN_FRACTION + VALIDATION_FRACTION + TEST_FRACTION) > 1):
    logging.error('(TRAIN_FRACTION + VALIDATION_FRACTION + TEST_FRACTION) must be <= 1')
    exit(-1)

# Return 3 if value > 2
def normalize(value):
    if value > 2:
        return 3
    return value 

# Return 1 if N < 0 else 0
def normalize2(value):
    if value < 0:
        return 1
    return 0 

count_flavour = [0]*5
count_category = [0]*15

'''
****************************************
*************** DATASETS ***************
****************************************
'''

partition = {'train' : [], 'validation' : [], 'test' : []} # Train, validation, and test IDs
labels = {}                                                # ID : label
y_train = []
y1_class_weights = []
y2_class_weights = []

only_train = ['nutau2', 'nutau3']

if UNIFORM:
    pass

# Iterate through label folders

logging.info('Filling datasets...')

count_neutrinos = 0
count_antineutrinos = 0
count_empty_views = 0
count_empty_events = 0
count_less_10nonzero_views = 0
count_less_10nonzero_events = 0

for images_path in glob.iglob(IMAGES_PATH + '/*/training'):
    count_train, count_val, count_test = (0, 0, 0)
    print(images_path)
    if 'nutau2' in images_path or 'nutau3' in images_path:
        continue
    files = list(glob.iglob(images_path + "/*.gz"))
    random.shuffle(files)
    for imagefile in files:
        file_ID = imagefile.split("/")[-1][:-3]
        infofile = images_path + '/' + file_ID + '.info'
        ID = imagefile.split("/")[-3] + "/" + imagefile.split("/")[-2] + "/" + file_ID
        try:
           info = open(infofile, 'r').readlines()
        except FileNotFoundError:
           continue
        if not len(info):
           print("No lines to read.")
           continue

        fInt = int(info[0].strip())
	
        flavour = fInt // 4
        interaction = fInt % 4

        fNuEnergy = float(info[1].strip())
        fLepEnergy = float(info[2].strip())
        fRecoNueEnergy = float(info[3].strip())
        fRecoNumuEnergy = float(info[4].strip())
        fRecoNutauEnergy = float(info[5].strip())
        fEventWeight = float(info[6].strip())

        fNuPDG = normalize2(int(info[7].strip()))
        fNProton = normalize(int(info[8].strip()))
        fNPion = normalize(int(info[9].strip()))
        fNPizero = normalize(int(info[10].strip()))
        fNNeutron = normalize(int(info[11].strip()))
	
	# special case: NC
        if fInt == 13:
            fNuPDG = -1
            flavour = 3
            interaction = -1
	
	# special case: cosmic    
        if fInt == 14:
            fNuPDG = -1
            flavour = 4
            interaction = -2

        if fNuPDG == 0:
            count_neutrinos+=1
        elif fNuPDG == 1:
            count_antineutrinos+=1

        random_value = np.random.uniform(0,1)
	
        with open(imagefile, 'rb') as image_file:
             try: 
                pixels = np.frombuffer(zlib.decompress(image_file.read()), dtype=np.uint8).reshape(VIEWS, PLANES, CELLS)
             except zlib.error:
                continue
		
             views = [None]*VIEWS
             empty_view = [0,0,0]
             non_empty_view = [0,0,0]
	     
             count_empty = 0
             count_less_10nonzero = 0
	     
             for i in range(len(views)):
                 views[i] = pixels[i, :, :].reshape(PLANES, CELLS, 1)
                 maxi = np.max(views[i]) 
                 mini = np.min(views[i])
                 nonzero = np.count_nonzero(views[i])
                 total = np.sum(views[i])
                 avg = np.mean(views[i])
		 
                 if nonzero == 0:
                    count_empty+=1
                    count_empty_views+=1
                 if nonzero < 10:
                    count_less_10nonzero+=1
                    count_less_10nonzero_views+=1
             
             if count_empty == len(views):
                count_empty_events+=1
		
             if count_less_10nonzero > 0:
                count_less_10nonzero_events+=1
                print("********* Filtered out **********")
                continue
                
                 
		
           
