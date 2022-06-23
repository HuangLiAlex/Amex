# import cupy, cudf  # GPU LIBRARIES
import os, gc
import numpy as np, pandas as pd  # CPU LIBRARIES
from src.feature_eng import feature_engineer
from src.hyper_param import PATH_TO_CUSTOMER_HASHES, PATH_TO_DATA, TRAIN_LABELS_CSV, TRAIN_DATA_CSV, TEST_DATA_CSV


def read_train_data():
    # LOAD TARGETS
    targets = pd.read_csv(TRAIN_LABELS_CSV)
    # targets['customer_ID'] = targets['customer_ID'].str[-16:].astype('int64')
    # pandas way
    targets['customer_ID'] = targets['customer_ID'].apply(lambda x: int(x[-16:], 16)).astype('int64')
    print(f'There are {targets.shape[0]} train targets')

    # GET TRAIN COLUMN NAMES
    if 'feather' in TRAIN_DATA_CSV:
        train = pd.read_feather(TRAIN_DATA_CSV)
    else:
        train = pd.read_csv(TRAIN_DATA_CSV, nrows=1)

    T_COLS = train.columns
    print(f'There are {len(T_COLS)} train dataframe columns')

    # GET TRAIN CUSTOMER NAMES (use pandas to avoid memory error)
    train_customers = pd.read_parquet(f'{PATH_TO_CUSTOMER_HASHES}train_customer_hashes.pqt')

    customers = train_customers.drop_duplicates().sort_index().values.flatten()
    print(f'There are {len(customers)} unique customers in train.')

    return train, targets, customers, T_COLS


def process_train_data(train, targets, rows, T_COLS, NUM_FILES):
    # CREATE PROCESSED TRAIN FILES AND SAVE TO DISK
    for k in range(NUM_FILES):

        # READ CHUNK OF TRAIN CSV FILE
        skip = int(np.sum(rows[:k]) + 1)  # the plus one is for skipping header
        train = pd.read_csv(TRAIN_DATA_CSV, nrows=rows[k],
                            skiprows=skip, header=None, names=T_COLS)

        # FEATURE ENGINEER DATAFRAME
        train = feature_engineer(train, targets=targets)

        # SAVE FILES
        print(f'Train_File_{k + 1} has {train.customer_ID.nunique()} customers and shape', train.shape)
        tar = train[['customer_ID', 'target']].drop_duplicates().sort_index()
        if not os.path.exists(PATH_TO_DATA): os.makedirs(PATH_TO_DATA)
        tar.to_parquet(f'{PATH_TO_DATA}targets_{k + 1}.pqt', index=False)
        data = train.iloc[:, 1:-1].values.reshape((-1, 13, 188))
        pd.save(f'{PATH_TO_DATA}data_{k + 1}', data.astype('float32'))

    # CLEAN MEMORY
    del train, tar, data
    del targets
    gc.collect()


def read_test_data():
    test = pd.read_csv(TEST_DATA_CSV, nrows=1)
    T_COLS = test.columns
    print(f'There are {len(T_COLS)} test dataframe columns')

    # GET TEST CUSTOMER NAMES (use pandas to avoid memory error)
    if PATH_TO_CUSTOMER_HASHES:
        test = pd.read_parquet(f'{PATH_TO_CUSTOMER_HASHES}test_customer_hashes.pqt')
    else:
        test = pd.read_csv('/raid/Kaggle/amex/test_data.csv', usecols=['customer_ID'])
        test['customer_ID'] = test['customer_ID'].apply(lambda x: int(x[-16:], 16)).astype('int64')
    customers = test.drop_duplicates().sort_index().values.flatten()
    print(f'There are {len(customers)} unique customers in test.')

    return test, customers, T_COLS


def process_test_data(rows, T_COLS, NUM_FILES):
    # SAVE TEST CUSTOMERS INDEX
    test_customer_hashes = np.array([], dtype='int64')
    # CREATE PROCESSED TEST FILES AND SAVE TO DISK
    for k in range(NUM_FILES):
        # READ CHUNK OF TEST CSV FILE
        skip = int(np.sum(rows[:k]) + 1)  # the plus one is for skipping header
        test = pd.read_csv(TEST_DATA_CSV, nrows=rows[k],
                           skiprows=skip, header=None, names=T_COLS)

        # FEATURE ENGINEER DATAFRAME
        test = feature_engineer(test, targets=None)

        # SAVE TEST CUSTOMERS INDEX
        cust = test[['customer_ID']].drop_duplicates().sort_index().values.flatten()
        test_customer_hashes = np.concatenate([test_customer_hashes, cust])

        # SAVE FILES
        print(f'Test_File_{k + 1} has {test.customer_ID.nunique()} customers and shape', test.shape)
        data = test.iloc[:, 1:].values.reshape((-1, 13, 188))
        pd.save(f'{PATH_TO_DATA}test_data_{k + 1}', data.astype('float32'))

    # SAVE CUSTOMER INDEX OF ALL TEST FILES
    pd.save(f'{PATH_TO_DATA}test_hashes_data', test_customer_hashes)

    # CLEAN MEMORY
    del test, data
    gc.collect()
