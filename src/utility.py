from importlibs import *

switcher = {
    'NYISO': ['Time Stamp', 'Name', 'LONGIL', 'LBMP ($/MWHr)'],
    'PJM': ['Time Stamp', 'Name', 'PJM', 'LBMP ($/MWHr)'],
    'CAISO': ['Date', 'hub', 'TH_NP15', 'price'],
    'ISONE': ['Time', None, None, '$/MWh']
}


def fetchData(isLabel: bool, regionalISOName):
    root_dir = '/dataset/' + regionalISOName
    filename = FILENAME_PRICE_DATASET_MAY_JUN_2019
    if isLabel:
        root_dir = '/labels/' + regionalISOName
        filename = FILENAME_PRICE_DATASET_JULY_2019

    fullpath = os.getcwd()
    projectPath = Path(fullpath).parents[0]
    datasetDirectoryPath = Path(str(projectPath) + root_dir)
    return pd.read_csv(os.path.join(datasetDirectoryPath, filename),
                       index_col=switcher[regionalISOName][0])


def prepareDataFor(isLabel: bool, regionalISOName, dateTimeColumnName, zoneFilterColumnName, zoneName, lbmpColumnName):
    innerDirPath = regionalISOName
    root_dir = '/dataset/' + innerDirPath
    if isLabel:
        root_dir = '/labels/' + regionalISOName

    fullpath = os.getcwd()
    projectPath = Path(fullpath).parents[0]
    datasetDirectoryPath = Path(str(projectPath) + root_dir)

    if Path(datasetDirectoryPath, FILENAME_PREPARED_DATASET).exists():
        return pd.read_csv(os.path.join(datasetDirectoryPath, FILENAME_PREPARED_DATASET), index_col=dateTimeColumnName)

    data = []
    totalDataSize = 0
    for file in sorted(datasetDirectoryPath.iterdir()):
        if Path.is_dir(file) and file.name == DIRECTORY_DATASET_DEC_1999_2019:
            for f in sorted(file.iterdir()):
                if f.name.endswith('.csv'):
                    df = pd.read_csv(f, parse_dates=True, index_col=dateTimeColumnName, header=0)
                    df.index = pd.to_datetime(df.index, format="%Y-%m-%d-%H-%M-%S")

                    df_filtered = df
                    if not (zoneFilterColumnName is None) and not (zoneName is None):
                        df_filtered = df.loc[df[zoneFilterColumnName] == zoneName]
                        df_filtered = df_filtered[lbmpColumnName]

                    data.append(df_filtered)
                    totalDataSize = totalDataSize + df_filtered.size
                    print('--- Size of new data after append: ', totalDataSize, ' File: ', f)

    df = pd.concat(data)
    df.sort_index().to_csv(os.path.join(datasetDirectoryPath, FILENAME_PREPARED_DATASET))
    return df


def prepareDataFromSpecificFolder(isLabel: bool, regionalISOName, directoryName, dateTimeColumnName, zoneFilterColumnName, zoneName, lbmpColumnName):
    innerDirPath = regionalISOName
    root_dir = '/dataset/' + innerDirPath
    if isLabel:
        root_dir = '/labels/' + regionalISOName

    fullpath = os.getcwd()
    projectPath = Path(fullpath).parents[0]
    datasetDirectoryPath = Path(str(projectPath) + root_dir)

    data = []
    totalDataSize = 0
    for file in sorted(datasetDirectoryPath.iterdir()):
        if Path.is_dir(file) and file.name == directoryName:
            for f in sorted(file.iterdir()):
                if f.name.endswith('.csv'):
                    df = pd.read_csv(f, parse_dates=True, index_col=dateTimeColumnName, header=0)
                    df.index = pd.to_datetime(df.index, format="%Y-%m-%d-%H-%M-%S")

                    df_filtered = df
                    if not (zoneFilterColumnName is None) and not (zoneName is None):
                        df_filtered = df.loc[df[zoneFilterColumnName] == zoneName]
                        df_filtered = df_filtered[lbmpColumnName]

                    data.append(df_filtered)
                    totalDataSize = totalDataSize + df_filtered.size
                    print('--- Size of new data after append: ', totalDataSize, ' File: ', f)

    df = pd.concat(data)
    df.sort_index().to_csv(os.path.join(datasetDirectoryPath, 'dataset_'+directoryName+'.csv'))
    return df

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append([dataset[i + look_back, 0]])
    return np.array(dataX), np.array(dataY)


def createHolyGrail(dataset, look_back):
    dataX= pd.DataFrame()
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        dataX = dataX.append(pd.Series(a[:,0]), ignore_index=True)
        # dataY.append([dataset[i + look_back, 0]])
    return dataX

def lstmModelFileExists(regional_ISO_name):
    root_dir = '/output/' + regional_ISO_name
    fullpath = os.getcwd()
    projectPath = Path(fullpath).parents[0]
    outputDirPath = Path(str(projectPath) + root_dir)
    return os.path.exists(os.path.join(outputDirPath, FILE_NAME_LSTM_MODEL))


def loadModelFile(regional_ISO_name):
    root_dir = '/output/' + regional_ISO_name
    fullpath = os.getcwd()
    projectPath = Path(fullpath).parents[0]
    outputDirPath = Path(str(projectPath) + root_dir)
    return load_model(os.path.join(outputDirPath, FILE_NAME_LSTM_MODEL))


def savePredictionsFor(regional_ISO_name, isArima: bool, predictionArr: list):
    root_dir = '/output/' + regional_ISO_name
    if isArima:
        root_dir = root_dir + '/' + DIRECTORY_NAME_ARIMA

    fullpath = os.getcwd()
    projectPath = Path(fullpath).parents[0]
    outputDirPath = Path(str(projectPath) + root_dir)
    pdSeries = pd.Series(predictionArr)
    with open(os.path.join(outputDirPath, FILENAME_ARIMA_PREDICTION), "w") as file1:
        pdSeries.to_csv(file1, index=False, header=False)


def arimaPredictionExistsFor(regional_ISO_name):
    root_dir = '/output/' + regional_ISO_name
    fullpath = os.getcwd()
    projectPath = Path(fullpath).parents[0]
    outputDirPath = Path(str(projectPath) + root_dir)

    return os.path.exists(os.path.join(outputDirPath, FILENAME_ARIMA_PREDICTION))


def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


def getOutputPathForISO(regional_ISO_name):
    root_dir = '/output/' + regional_ISO_name
    fullpath = os.getcwd()
    project_path = Path(fullpath).parents[0]
    return os.path.join(Path(str(project_path) + root_dir))


def getOutputPathForModelFile(regional_ISO_name, filename):
    return os.path.join(getOutputPathForISO(regional_ISO_name), filename)


def moving_test_window_preds(model, futureX, n_future_preds=288):
    ''' n_future_preds - Represents the number of future predictions we want to make
                         This coincides with the number of windows that we will move forward
                         on the test data
    '''
    preds_moving = []  # Use this to store the prediction made on each test window
    moving_test_window = [futureX[0,:].tolist()]  # Creating the first test window
    moving_test_window = np.array(moving_test_window)  # Making it an numpy array

    for i in range(n_future_preds):
        preds_one_step = model.predict(
            moving_test_window)  # Note that this is already a scaled prediction so no need to rescale this
        preds_moving.append(preds_one_step[0, 0])  # get the value from the numpy 2D array and append to predictions
        preds_one_step = preds_one_step.reshape(1, 1)  # Reshaping the prediction to 3D array for concatenation with moving test window
        moving_test_window = np.append(moving_test_window[:,1:], preds_one_step) # This is the new moving test window, where the first element from the window has been removed and the prediction  has been appended to the end
        moving_test_window = moving_test_window.reshape(1, len(moving_test_window))

    # preds_moving = scaler.inverse_transform(preds_moving)

    return preds_moving


def verify_and_clean_dataset_datetime_wise(df, timestampColumnName, start, end, freq):
    datetime_arr = pd.date_range(start, end, freq=freq)

    clean_df = pd.DataFrame()
    df_datetime_series = pd.to_datetime(df.iloc[:,0])
    df[timestampColumnName] = df_datetime_series
    is_df_range_correct = False
    missingDates = []
    for dt in datetime_arr:
        dt_formatted = pd.to_datetime(dt, format="%Y-%m-%d-%H-%M-%S")
        if dt_formatted in df[timestampColumnName].values:
            clean_df = clean_df.append(df.loc[df[timestampColumnName]==dt])
            # print(clean_df.shape)
        else:
            print('Data missing for: ',dt_formatted)
            missingDates.append(dt_formatted)

    if(len(clean_df.values) == len(datetime_arr)):
        is_df_range_correct = True
    clean_df.index = clean_df[timestampColumnName]
    clean_df = clean_df.reset_index(drop=True)
    return is_df_range_correct, clean_df, missingDates