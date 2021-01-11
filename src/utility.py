from importlibs import *


def prepareDataForNYISO(columnName, zoneName):
    root_dir = '/dataset/NYISO'
    fullpath = os.getcwd()

    projectPath = Path(fullpath).parents[0]
    datasetDirectoryPath = Path(str(projectPath) + root_dir)

    data = []
    totalDataSize = 0
    for file in sorted(datasetDirectoryPath.iterdir()):
        if Path.is_dir(file):
            for f in sorted(file.iterdir()):
                df = pd.read_csv(f, index_col=None, header=0)

                edited_df = df.loc[df[columnName] == zoneName]
                # print('editedDfSize: ', edited_df.size)
                data.append(edited_df)
                totalDataSize = totalDataSize + edited_df.size
                # print('--- Size of new data after append: ', totalDataSize)

    df = pd.concat(data)
    return df.reset_index(drop=True)


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)
