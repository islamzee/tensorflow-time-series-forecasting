import enum

SPLIT_FRACTION = 0.95
LOOK_BACK = 12
EPOCH_SIZE = 10
BATCH_SIZE = 30

YEARS_INPUT = [2017, 2018, 2019, 2020]

DIRECTORY_NAME_ARIMA = 'arima'
DIRECTORY_NAME_ROLLING_LSTM = 'rolling_lstm'
DIRECTORY_DATASET_DEC_1 = 'DEC-1'
DIRECTORY_DATASET_DEC_1999_2019 = 'DEC-1999-2019'

FILE_NAME_LSTM_MODEL = 'lstm_pred.h5'
FILE_NAME_ROLLING_LSTM_MODEL = 'rolling_lstm_pred.h5'
FILENAME_PREPARED_DATASET = 'dataset.csv'
FILENAME_PRICE_DATASET_JULY_2019 = 'dataset_07-2019.csv'
FILENAME_PRICE_DATASET_MAY_JUN_2019 = 'dataset_05-06-2019.csv'
FILENAME_ARIMA_PREDICTION = 'arima_prediction.csv'
FILENAME_ROLLING_LSTM_PREDICTION = 'prediction.csv'

SUB_FILE_NAME_PLOT_DATASET = '_plot_dataset.pdf'
SUB_FILE_NAME_PLOT_FUTURE = '_plot_future.pdf'

TIMESTAMP_PREVIOUS_DAY_START = '2020-11-30 23:00'
TIMESTAMP_PREVIOUS_DAY_END = '2020-12-01 00:00'

class PredictionType(enum.Enum):
    ARIMA = 1
    ROLLING_LSTM = 3

class ISO(enum.Enum):
    caiso = 'CAISO'
    nyiso = 'NYISO'
    isone = 'ISONE'
    pjm = 'PJM'
