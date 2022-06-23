from matplotlib import pyplot as plt

from src.hyper_param import OUTPUT_CSV


def save_sub_csv(submit):
    submit.to_csv(OUTPUT_CSV, index=False)
    print('Submission file shape is', submit.shape)
    print(submit.head())


def plot_predict(submit):
    # DISPLAY SUBMISSION PREDICTIONS
    plt.hist(submit.prediction, bins=100)
    plt.title('Test Predictions')
    plt.show()
