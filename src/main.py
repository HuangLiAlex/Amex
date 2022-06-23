# https://www.kaggle.com/code/alexlee127/tensorflow-transformer-0-790/edit
from src.analyse import get_rows
from src.feature_eng import feature_engineer
from src.infer_test import infer_test
from src.process_data import read_train_data, process_train_data, read_test_data, process_test_data
from src.submission import save_sub_csv, plot_predict
from src.train_model import train_model
from hyper_param import *
import torch


print('Using PyTorch version', torch.__version__)


if PROCESS_DATA:
    print('Start to process train data')
    train, targets, customers, T_COLS = read_train_data()

    # this is to divide full train data into 10 files
    # rows is number of rows for each file
    # we shall skip
    # NUM_FILES = 10
    # rows = get_rows(customers, train, NUM_FILES = NUM_FILES, verbose ='train')

    # this methods divides train data into 10 pqt files
    # as pqt files already given in dataset, we shall skip this step
    # process_train_data(train, targets, rows, T_COLS, NUM_FILES)

    # do feature engineering
    # this will pad customers into 13 rows
    # shape: n_customers * 13 * 188_features
    train = feature_engineer(train, targets=targets)

    # print('Start to process test data')
    # test, customers, T_COLS = read_test_data()
    # NUM_FILES = 20
    # rows = get_rows(customers, test, NUM_FILES=NUM_FILES, verbose='test')
    #
    # process_test_data(test, rows, T_COLS, NUM_FILES)

# if TRAIN_MODEL:
#     print('Start to train model')
#     train_model()
#
# if INFER_TEST:
#     print('Start to infer test data')
#     submit = infer_test()
#
#     print('Start to save result')
#     save_sub_csv(submit)
#     # plot_predict(submit)
