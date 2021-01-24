from importlibs import *


def prepareDataFor(regionalISOName, dateTimeColumnName, zoneFilterColumnName, zoneName, lbmpColumnName):
    root_dir = '/dataset/' + regionalISOName
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

                    edited_df = df.loc[df[zoneFilterColumnName] == zoneName]
                    edited_df = edited_df[lbmpColumnName]
                    data.append(edited_df)
                    totalDataSize = totalDataSize + edited_df.size
                    print('--- Size of new data after append: ', totalDataSize, ' File: ', f)

    df = pd.concat(data)
    df.to_csv(os.path.join(datasetDirectoryPath, FILENAME_PREPARED_DATASET))
    return df


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append([dataset[i + look_back, 0]])
    return np.array(dataX), np.array(dataY)
