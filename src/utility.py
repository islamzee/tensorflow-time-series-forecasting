from importlibs import *

switcher = {
    'NYISO': ['Time Stamp', 'Name', 'LONGIL', 'LBMP ($/MWHr)'],
    'PJM': ['Time Stamp', 'Name', 'PJM', 'LBMP ($/MWHr)'],
    'CAISO': ['Date', 'hub', 'TH_NP15', 'price'],
    'ISONE': ['Time', None, None, '$/MWh']
}


def fetchData(isLabel: bool, regionalISOName):
    root_dir = '/dataset/' + regionalISOName
    if isLabel:
        root_dir = '/labels/' + regionalISOName

    fullpath = os.getcwd()
    projectPath = Path(fullpath).parents[0]
    datasetDirectoryPath = Path(str(projectPath) + root_dir)
    return pd.read_csv(os.path.join(datasetDirectoryPath, FILENAME_PREPARED_DATASET),
                       index_col=switcher[regionalISOName][0])


def prepareDataFor(isLabel: bool, regionalISOName, dateTimeColumnName, zoneFilterColumnName, zoneName, lbmpColumnName):
    root_dir = '/dataset/' + regionalISOName
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
        if Path.is_dir(file):
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


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append([dataset[i + look_back, 0]])
    return np.array(dataX), np.array(dataY)


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


def getOutputPathForModelFile(regional_ISO_name, filename):
    root_dir = '/output/' + regional_ISO_name
    fullpath = os.getcwd()
    projectPath = Path(fullpath).parents[0]
    return os.path.join(Path(str(projectPath) + root_dir, filename))