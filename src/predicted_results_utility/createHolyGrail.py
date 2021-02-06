import numpy as np
from predicted_results_utility.utility import *
from constants import *


def caseArima(regional_ISO_name):
    file_path = os.path.join(getOutputPathForISO(regional_ISO_name), DIRECTORY_NAME_ARIMA, FILENAME_ARIMA_PREDICTION)
    return pd.read_csv(file_path)


def caseRollingLSTM(regional_ISO_name):
    file_path = os.path.join(getOutputPathForISO(regional_ISO_name), DIRECTORY_NAME_ROLLING_LSTM,
                             FILENAME_ROLLING_LSTM_PREDICTION)
    return pd.read_csv(file_path, header=None)


def getHolyGrail(regional_ISO_name, prediction_type):
    switcher = {
        PredictionType.ARIMA: caseArima(regional_ISO_name),
        PredictionType.ROLLING_LSTM: caseRollingLSTM(regional_ISO_name),
    }

    predictions = switcher.get(prediction_type)
    predictions_values = predictions.values
    holyGrail = []
    # process 'predictions' in 288 * 12 format
    for i in range(len(predictions_values)):
        row = predictions_values[(i + 1):(i + 12 + 1)].flatten()
        holyGrail.append(row)

    return pd.DataFrame(holyGrail)


def create_holygrail(regional_ISO_name, predictionType):
    holyGrail = getHolyGrail(regional_ISO_name, predictionType)
    dirNameSwitcher = {
        PredictionType.ARIMA: DIRECTORY_NAME_ARIMA,
        PredictionType.ROLLING_LSTM: DIRECTORY_NAME_ROLLING_LSTM
    }

    filepath = os.path.join(getOutputPathForISO(regional_ISO_name), dirNameSwitcher.get(predictionType),
                            'holyGrail_12.csv')
    holyGrail.to_csv(filepath, index=None, header=None)


for iso in ISO:
    for predType in PredictionType:
        create_holygrail(iso.value, predType)
        print('--- DONE: ', iso.value, ' --- ', predType)
