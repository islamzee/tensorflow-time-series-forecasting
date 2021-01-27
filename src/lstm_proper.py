from importlibs import *
import glob
from utility import *


def run_LSTM(regional_ISO_name):
    input = fetchData(False, regional_ISO_name)
    print('Dataset total size: ', input.shape)

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = input.astype('float32')
    # dataset = pd.DataFrame(scaler.fit_transform(dataset.values.reshape(-1, 1)), index=dataset.index)

    # split into train and test sets
    train_size = int(len(dataset) * SPLIT_FRACTION)
    # train_size = 1000
    # test_size = 800
    train, test = dataset.iloc[0:train_size, :], dataset.iloc[train_size:len(dataset), :]
    # train, test = dataset.iloc[0:train_size,:], dataset.iloc[train_size:1800,:]

    train = pd.DataFrame(scaler.fit_transform(train.values.reshape(-1, 1)), index=train.index)
    test = pd.DataFrame(scaler.fit_transform(test.values.reshape(-1, 1)), index=test.index)
    print(len(train), len(test))

    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train.values, LOOK_BACK)  # trainSize * lookbackSize, trainSize * 1
    testX, testY = create_dataset(test.values, LOOK_BACK)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    # create and fit the LSTM network
    model = Sequential()
    # if not os.path.exists(FILE_NAME_LSTM_MODEL):
    if not lstmModelFileExists(regional_ISO_name):
        model.add(LSTM(units=4, input_shape=(LOOK_BACK, 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY,
                  epochs=EPOCH_SIZE, batch_size=BATCH_SIZE,
                  validation_data=(testX, testY),
                  verbose=1)

        model.save(FILE_NAME_LSTM_MODEL)  # creates a HDF5 file 'my_model.h5'

    # returns a compiled model identical to the previous one
    model = load_model(FILE_NAME_LSTM_MODEL)
    # ------------------------------------------------------------

    label_dataset = fetchData(True, regional_ISO_name)

    # normalize label_dataset
    label_dataset = pd.DataFrame(scaler.fit_transform(label_dataset.values.reshape(-1, 1)), index=label_dataset.index)

    futureX, futureY = create_dataset(label_dataset)
    futureX = np.reshape(futureX, (futureX.shape[0], futureX.shape[1], 1))

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    futurePredict = model.predict(futureX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)
    futurePredict = scaler.inverse_transform(futurePredict)
    futureY = scaler.inverse_transform(futureY)


    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY, testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))
    futureScore = math.sqrt(mean_squared_error(futureY, futurePredict[:, 0]))
    print('Future Predict Score: %.2f RMSE' % (futureScore))


    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[LOOK_BACK:len(trainPredict) + LOOK_BACK, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (LOOK_BACK * 2) + 1:len(dataset) - 1, :] = testPredict
    # testPredictPlot[len(trainPredict)+(LOOK_BACK*2)+1:1800-1, :] = testPredict

    #shift future predictions for plotting

    # plot baseline and predictions
    xAxis = dataset.index.to_numpy()
    df = pd.DataFrame(index=dataset.index)
    df['dataset'] = scaler.inverse_transform(dataset)
    df['train'] = trainPredictPlot
    df['test'] = testPredictPlot

    fig = df.plot(y=['dataset', 'train', 'test'], use_index=True, x_compat=True).get_figure()
    fig.savefig('Plot_LSTM.pdf')


# ==========================================================================================================

run_LSTM('ISONE')
