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
    train, test = dataset.iloc[0:train_size, :], dataset.iloc[train_size:len(dataset), :]

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
    if not lstmModelFileExists(regional_ISO_name):
        model.add(LSTM(units=4, input_shape=(LOOK_BACK, 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY,
                  epochs=EPOCH_SIZE, batch_size=BATCH_SIZE,
                  validation_data=(testX, testY),
                  verbose=1)
        root_dir = '/output/' + regional_ISO_name
        fullpath = os.getcwd()
        projectPath = Path(fullpath).parents[0]
        outputDirPath = Path(str(projectPath) + root_dir)
        model.save(os.path.join(outputDirPath, FILE_NAME_LSTM_MODEL))  # creates a HDF5 file 'my_model.h5'
    else:
        # returns a compiled model identical to the previous one
        directoryPath = '../output/' + regional_ISO_name + '/'
        model = load_model(directoryPath + FILE_NAME_LSTM_MODEL)
    # ------------------------------------------------------------

    label_dataset = fetchData(True, regional_ISO_name)

    # normalize label_dataset
    label_dataset = pd.DataFrame(scaler.fit_transform(label_dataset.values.reshape(-1, 1)), index=label_dataset.index)

    futureX, futureY = create_dataset(label_dataset.values, LOOK_BACK)
    futureX = np.reshape(futureX, (futureX.shape[0], futureX.shape[1], 1))

    # make predictions
    # trainPredict = model.predict(trainX)
    # testPredict = model.predict(testX)
    futurePredict = model.predict(futureX)

    # invert predictions
    # trainPredict = scaler.inverse_transform(trainPredict)
    # trainY = scaler.inverse_transform(trainY)
    # testPredict = scaler.inverse_transform(testPredict)
    # testY = scaler.inverse_transform(testY)
    futurePredict = scaler.inverse_transform(futurePredict)
    futureY = scaler.inverse_transform(futureY)

    # invert label_dataset
    label_dataset = pd.DataFrame(scaler.inverse_transform(label_dataset), index=label_dataset.index)

    # calculate root mean squared error
    # trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:, 0]))
    # print('Train Score: %.2f RMSE' % (trainScore))
    # testScore = math.sqrt(mean_squared_error(testY, testPredict[:, 0]))
    # print('Test Score: %.2f RMSE' % (testScore))
    futureScore = sqrt(mean_squared_error(futureY, futurePredict[:, 0]))
    print('Future Predict Score: %.2f RMSE' % (futureScore))

    # ------

    # plot baseline and predictions
    df = pd.DataFrame(index=dataset.index.append(label_dataset.index), columns=('dataset', 'train', 'test', 'future'))
    fullData = np.append(dataset.values, label_dataset)
    df['dataset'][:] = pd.Series(fullData.flatten())

    # df['train'][0:len(trainPredict)] = pd.Series(trainPredict.flatten())
    # df['test'][0:len(testPredict)] = pd.Series(testPredict.flatten())
    # df['future'][0:len(futurePredict)] = pd.Series(futurePredict.flatten())

    future_datetime_arr = pd.date_range(start='2020-12-01 00:00:00', freq='5T', periods=len(futurePredict))
    df_future = pd.DataFrame({'Predicted':futurePredict[:,0],'Actual':futureY[:,0]},
                            index=future_datetime_arr)

    # ========= PLOT =======
    xAxis = label_dataset.index.to_numpy()

    # f1 = plt.figure()
    df.plot(y=['dataset', 'train', 'test'], use_index=True, x_compat=True)
    plt.savefig('../output/'
                + regional_ISO_name + '/'
                + regional_ISO_name + SUB_FILE_NAME_PLOT_DATASET)

    # f2 = plt.figure()
    # plt.plot(xAxis, label_dataset.values, label='Actual', color="blue")
    # y = np.concatenate([futurePredict[:, 0], np.zeros(13)])
    # plt.plot(xAxis, y, label='Predicted', color="green")
    # plt.legend()
    # plt.title(regional_ISO_name + ': 1 Dec 2020')
    # plt.savefig('../output/'
    #             + regional_ISO_name + '/'
    #             + regional_ISO_name + SUB_FILE_NAME_PLOT_FUTURE)


    # f3 = plt.figure()
    df_future.plot(use_index=True, x_compat=True)
    plt.savefig('../output/'
                + regional_ISO_name + '/'
                + regional_ISO_name + SUB_FILE_NAME_PLOT_FUTURE)
    plt.show()

# ==========================================================================================================

run_LSTM('PJM')
