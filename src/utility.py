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
    return pd.read_csv(os.path.join(datasetDirectoryPath, FILENAME_PREPARED_DATASET), index_col=switcher[regionalISOName][0])


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
                    if not(zoneFilterColumnName is None) and not(zoneName is None):
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