# import cupy, cudf  # GPU LIBRARIES
import os, gc
import numpy as np, pandas as pd  # CPU LIBRARIES
from src.feature_eng import feature_engineer
from src.hyper_param import PATH_TO_DATA, TRAIN_DATA_PATH, TEST_DATA_PATH


def process_train_data(trainX, targets, rows, T_COLS, NUM_FILES):
    # CREATE PROCESSED TRAIN FILES AND SAVE TO DISK
    for k in range(NUM_FILES):

        # READ CHUNK OF TRAIN CSV FILE
        skip = int(np.sum(rows[:k]))
        if 'feather' in TRAIN_DATA_PATH or 'parquet' in TRAIN_DATA_PATH:
            # print("skip: ", skip)
            # print("rows[k]: ", rows[k])
            train = trainX.loc[skip:skip+rows[k]-1]
        else:
            train = pd.read_csv(TRAIN_DATA_PATH, nrows=rows[k],
                                skiprows=skip, header=None, names=T_COLS)

        # FEATURE ENGINEER DATAFRAME
        # return 1/k train data
        train = feature_engineer(train, targets=targets)
        print("train shape: ", train.shape)

        # SAVE FILES
        print(f'Train_File_{k + 1} has {train.customer_ID.nunique()} customers and shape', train.shape)
        tar = train[['customer_ID', 'target']].drop_duplicates().sort_index()
        # tar.info()
        if not os.path.exists(PATH_TO_DATA): os.makedirs(PATH_TO_DATA)
        tar.to_parquet(f'{PATH_TO_DATA}trans_targets_{k + 1}.pqt', index=False)

        data = train.iloc[:, 1:-1].values.reshape((-1, 13, 189))
        # print(data[0:5])
        np.save(f'{PATH_TO_DATA}trans_data_{k + 1}', data.astype('float32'))

    # CLEAN MEMORY
    del train, tar, data
    del targets
    gc.collect()


def process_test_data(testX, rows, T_COLS, NUM_FILES):
    # SAVE TEST CUSTOMERS INDEX
    test_customer_hashes = np.array([], dtype='int64')
    # CREATE PROCESSED TEST FILES AND SAVE TO DISK
    for k in range(NUM_FILES):
        # READ CHUNK OF TEST CSV FILE
        skip = int(np.sum(rows[:k]))  # the plus one is for skipping header
        if 'feather' in TRAIN_DATA_PATH or 'parquet' in TRAIN_DATA_PATH:
            test = testX.loc[skip:skip+rows[k]-1]
        else:
            test = pd.read_csv(TEST_DATA_PATH, nrows=rows[k],
                           skiprows=skip, header=None, names=T_COLS)

        # FEATURE ENGINEER DATAFRAME
        test = feature_engineer(test, targets=None)

        # SAVE TEST CUSTOMERS INDEX
        cust = test[['customer_ID']].drop_duplicates().sort_index().values.flatten()
        test_customer_hashes = np.concatenate([test_customer_hashes, cust])

        # SAVE FILES
        print(f'Test_File_{k + 1} has {test.customer_ID.nunique()} customers and shape', test.shape)
        data = test.iloc[:, 1:].values.reshape((-1, 13, 189))
        np.save(f'{PATH_TO_DATA}trans_test_data_{k + 1}', data.astype('float32'))

    # SAVE CUSTOMER INDEX OF ALL TEST FILES
    np.save(f'{PATH_TO_DATA}test_hashes_data', test_customer_hashes)

    # CLEAN MEMORY
    del test, data
    gc.collect()
