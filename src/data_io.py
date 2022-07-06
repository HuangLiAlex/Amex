import pandas as pd

from src.hyper_param import TRAIN_LABELS_CSV, TRAIN_DATA_PATH, PATH_TO_CUSTOMER_HASHES, TEST_DATA_PATH


def get_rows(customers, train, NUM_FILES = 10, verbose =''):
    chunk = len(customers)//NUM_FILES
    if verbose != '':
        print(f'We will split {verbose} data into {NUM_FILES} separate files.')
        print(f'There will be {chunk} customers in each file (except the last file).')
        print('Below are number of rows in each file:')
    rows = []

    for k in range(NUM_FILES):
        if k==NUM_FILES-1: cc = customers[k*chunk:]
        else: cc = customers[k*chunk:(k+1)*chunk]
        s = train.loc[train.customer_ID.isin(cc)].shape[0]
        rows.append(s)
    if verbose != '': print( rows )
    return rows


def read_train_data():
    # LOAD TARGETS
    targets = pd.read_csv(TRAIN_LABELS_CSV)
    # targets['customer_ID'] = targets['customer_ID'].str[-16:].astype('int64')
    # pandas way
    targets['customer_ID'] = targets['customer_ID'].apply(lambda x: int(x[-16:], 16)).astype('int64')
    print(f'There are {targets.shape[0]} train targets')

    # Read data
    if 'feather' in TRAIN_DATA_PATH:
        train = pd.read_feather(TRAIN_DATA_PATH)
    elif 'parquet' in TRAIN_DATA_PATH:
        train = pd.read_parquet(TRAIN_DATA_PATH)
    else:
        train = pd.read_csv(TRAIN_DATA_PATH)

    # Convert customer ID to int64
    train_ID = pd.DataFrame()
    train['customer_ID'] = train['customer_ID'].apply(lambda x: int(x[-16:], 16)).astype('int64')
    train_ID['customer_ID'] = train['customer_ID']
    # train_ID.info()

    # Print column number
    T_COLS = train.columns
    print(f'There are {len(T_COLS)} train dataframe columns')

    # GET DISTINCT TRAIN CUSTOMER NAMES (use pandas to avoid memory error)
    train_customers = pd.read_parquet(f'{PATH_TO_CUSTOMER_HASHES}train_customer_hashes.pqt')
    customers = train_customers.drop_duplicates().sort_index().values.flatten()
    print(f'There are {len(customers)} unique customers in train.')

    '''
        this is to divide full train data into 10 files
        rows is number of rows for each file
        we shall skip
    '''
    NUM_FILES = 10
    rows = get_rows(customers, train_ID, NUM_FILES=NUM_FILES, verbose='train')

    return train, train_ID, targets, customers, T_COLS, rows


def read_test_data():
    # Read Data
    if 'feather' in TEST_DATA_PATH:
        test = pd.read_feather(TEST_DATA_PATH)
    elif 'parquet' in TEST_DATA_PATH:
        test = pd.read_parquet(TEST_DATA_PATH)
    else:
        test = pd.read_csv(TEST_DATA_PATH)

    # Convert customer ID to int64
    test['customer_ID'] = test['customer_ID'].apply(lambda x: int(x[-16:], 16)).astype('int64')

    # Print column number
    T_COLS = test.columns
    print(f'There are {len(T_COLS)} test dataframe columns')

    # GET TEST CUSTOMER NAMES (use pandas to avoid memory error)
    test_ID = pd.read_parquet(f'{PATH_TO_CUSTOMER_HASHES}test_customer_hashes.pqt')

    customers = test_ID.drop_duplicates().sort_index().values.flatten()
    print(f'There are {len(customers)} unique customers in test.')

    # Calc number of rows for each file
    NUM_FILES = 20
    rows = get_rows(customers, test_ID, NUM_FILES=NUM_FILES, verbose='test')

    return test, test_ID, customers, T_COLS, rows

