from importlibs import *
import glob
from utility import *

input = prepareDataForNYISO('Name', 'LONGIL')
print(input.shape)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = input.iloc[:,3].values.astype('float32')
dataset = scaler.fit_transform(dataset.reshape(-1,1))

# split into train and test sets
train_size = int(len(dataset) * SPLIT_FRACTION)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

# reshape into X=t and Y=t+1
look_back = LOOK_BACK
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)