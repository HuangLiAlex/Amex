#
TRAIN_LABELS_CSV = '../input/amex-default-prediction/train_labels.csv'
# TRAIN_DATA_PATH = '../input/amex-default-prediction/train_data.csv'
# TEST_DATA_PATH = '../input/amex-default-prediction/test_data.csv'
TRAIN_DATA_PATH = '../input/amex_data_feather/train.feather'
TEST_DATA_PATH = '../input/amex_data_feather/test.feather'
# TRAIN_DATA_PATH = '../input/radar/train.parquet'
# TEST_DATA_PATH = '../input/radar/test.parquet'
#
PATH_TO_CUSTOMER_HASHES = '../input/amex-data-files/'

#


PATH_TO_DATA = '../input/data3d/'

#

PATH_TO_MODEL = '../saved_model/'
# PATH_TO_MODEL = '../input/amex-data-for-transformers-and-rnns/model/'



OUTPUT_SAMPLE_CSV = '../output/sample_submission.csv'
OUTPUT_CSV = '../output/submission.csv'

params = {
    'model': 'GRU',
    'batch_size': 512,
    'lr': 0.002,
    'wd': 1e-5,
    'device': 'cpu',
    # 'device': 'cuda:0',
    'early_stopping': 4,
    'n_fold': 5,
    'seed': 2021,
    'max_epochs': 20,
}

