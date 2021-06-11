from importlibs import *
import glob

from utility import *

isoList = ['CAISO', 'NYISO', 'PJM', 'ISONE']
# prepareDataFor(False, regional_ISO_name, switcher[regional_ISO_name][0], switcher[regional_ISO_name][1], switcher[regional_ISO_name][2], switcher[regional_ISO_name][3] )
# prepareDataFromSpecificFolder(False, regional_ISO_name, '05-06-2019', switcher[regional_ISO_name][0], switcher[regional_ISO_name][1], switcher[regional_ISO_name][2], switcher[regional_ISO_name][3] )

for regional_ISO_name in isoList:
    print('======================================================================')
    print('       ISO: ', regional_ISO_name)
    print('======================================================================')

    priceDatasetPath = '../dataset/'+regional_ISO_name+'/dataset_07-2019.csv'
    df = pd.read_csv(priceDatasetPath)
    isCorrect, clean_df, missingDates = verify_and_clean_dataset_datetime_wise(df, switcher[regional_ISO_name][0],
                                                        '2019-07-01 00:00:00', '2019-07-31 23:55:00', '5T')
    print('isCorrect = ', isCorrect)
    print(missingDates)
    # clean_df.to_csv(priceDatasetPath, index=False)

# for regional_ISO_name in isoList:
#     arimaPredictionFilePath = '../output/' + regional_ISO_name + '/JUL2019_arima_prediction.csv'
#     df_raw = pd.read_csv(arimaPredictionFilePath, header=None)
#     result = createHolyGrail(df_raw.values, 12)
#     result.to_csv('../output/' + regional_ISO_name + '/holyGrailJuly2019.csv', header=None, index=None)
