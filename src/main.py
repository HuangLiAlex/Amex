# https://www.kaggle.com/code/alexlee127/tensorflow-transformer-0-790/edit
from src.feature_eng import feature_engineer
from src.infer_test import infer_test
from src.process_data import process_train_data, process_test_data
from src.data_io import read_train_data, read_test_data
from src.submission import save_sub_csv, plot_predict
from src.train_model import train_model_GRU
from hyper_param import *
import torch
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


print('Using PyTorch version', torch.__version__)

'''
    Process Data
'''
PROCESS_TRAIN_DATA = False
if PROCESS_TRAIN_DATA:
    print('Start to process train data')
    train, train_ID, targets, customers, T_COLS, rows = read_train_data()

    # this methods divides train data into 10 pqt files
    # as pqt files already given in dataset, we shall skip this step
    process_train_data(train, targets, rows, T_COLS, NUM_FILES=10)

else:
    print('Skipped process train data')

PROCESS_TEST_DATA = False
if PROCESS_TEST_DATA:
    print('Start to process test data')
    test, test_ID, customers, T_COLS, rows = read_test_data()

    process_test_data(test, rows, T_COLS, NUM_FILES=20)
else:
    print('Skipped process test data')

'''
    Train Model
'''
TRAIN_MODEL = True
if TRAIN_MODEL:
    print('Start to train model')
    train_model_GRU()
else:
    print('Skipped train model')


'''
    Infer Test
'''
INFER_TEST = False
if INFER_TEST:
    print('Start to infer test data')
#     submit = infer_test()
#
#     print('Start to save result')
#     save_sub_csv(submit)
#     # plot_predict(submit)
else:
    print('Skipped infer test data')
