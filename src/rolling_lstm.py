from importlibs import *
import glob
from utility import *


def lstmModelFileExists(regional_ISO_name):
    root_dir = '/output/' + regional_ISO_name
    fullpath = os.getcwd()
    projectPath = Path(fullpath).parents[0]
    outputDirPath = Path(str(projectPath) + root_dir)
    return os.path.exists(os.path.join(outputDirPath, FILE_NAME_ROLLING_LSTM_MODEL))


def loadModelFile(regional_ISO_name):
    root_dir = '/output/' + regional_ISO_name
    fullpath = os.getcwd()
    projectPath = Path(fullpath).parents[0]
    outputDirPath = Path(str(projectPath) + root_dir)
    return load_model(os.path.join(outputDirPath, FILE_NAME_ROLLING_LSTM_MODEL))


def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        model.reset_states()
    return model


def forecast_lstm(model, batch_size, row):
    X = row.reshape(1, 1, len(row))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


def rolling_lstm(regional_ISO_name):
    # load dataset
    input = fetchData(False, regional_ISO_name)
    print('Dataset total size: ', input.shape)
    label_dataset = fetchData(True, regional_ISO_name)

    # transform data to be stationary
    raw_values = input.values
    # --- diff_values = difference(raw_values, 1)
    diff_values = raw_values
    label_values = label_dataset.values

    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values

    test_supervised = timeseries_to_supervised(label_values, 1)
    test_supervised_values = test_supervised.values
    # split data to train and test-sets
    train, test = supervised_values[0:-288], test_supervised_values[:]

    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)

    # fit the model
    lstm_model = Sequential()
    if not lstmModelFileExists(regional_ISO_name):
        lstm_model = fit_lstm(train_scaled, BATCH_SIZE, EPOCH_SIZE, 4)
        lstm_model.save(getOutputPathForModelFile(regional_ISO_name, FILE_NAME_ROLLING_LSTM_MODEL))
    else:
        lstm_model = loadModelFile(regional_ISO_name)

    # forecast the entire training dataset to build up state for forecasting
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    # lstm_model.predict(train_reshaped, batch_size=1)

    # walk-forward validation
    predictions = list()
    for i in range(len(test_scaled)):
        # make one-step forecast
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]

        yhat = forecast_lstm(lstm_model, 1, X)
        # invert scaling
        yhat = invert_scale(scaler, X, yhat)
        # inverse differencing
        # --- yhat = inverse_difference(raw_values, yhat, len(test_scaled))
        # store forecast
        predictions.append(yhat)
        expected = label_values[i]
        print('Iteration=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))

    # report performance
    rmse = sqrt(mean_squared_error(label_values, predictions))
    print('RMSE: %.3f' % rmse)

    # line plot of observed VS predicted
    plt.plot(label_values, label='Test')
    plt.plot(predictions, label='Predicted')
    plt.legend()

    filepath = getOutputPathForModelFile(regional_ISO_name, 'rolling_lstm/prediction.csv')
    np.savetxt(filepath, predictions, delimiter=',')
    plt.savefig(getOutputPathForModelFile(regional_ISO_name, 'rolling_lstm/plot.pdf'))

    plt.show()

rolling_lstm('NYISO')
rolling_lstm('PJM')

