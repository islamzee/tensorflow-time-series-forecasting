import enum

SPLIT_FRACTION = 0.67
LOOK_BACK = 12
EPOCH_SIZE = 1
BATCH_SIZE = 1

YEARS_INPUT = [2017, 2018, 2019, 2020]

DIRECTORY_NAME_ARIMA = 'arima'
DIRECTORY_NAME_ROLLING_LSTM = 'rolling_lstm'

FILE_NAME_LSTM_MODEL = 'lstm_pred.h5'
FILE_NAME_ROLLING_LSTM_MODEL = 'rolling_lstm_pred.h5'
FILENAME_PREPARED_DATASET = 'dataset.csv'
FILENAME_ARIMA_PREDICTION = 'arima_prediction.csv'
FILENAME_ROLLING_LSTM_PREDICTION = 'prediction.csv'

SUB_FILE_NAME_PLOT_DATASET = '_plot_dataset.pdf'
SUB_FILE_NAME_PLOT_FUTURE = '_plot_future.pdf'


class PredictionType(enum.Enum):
    ARIMA = 1
    ROLLING_LSTM = 3

class ISO(enum.Enum):
    caiso = 'CAISO'
    nyiso = 'NYISO'
    isone = 'ISONE'
    pjm = 'PJM'
