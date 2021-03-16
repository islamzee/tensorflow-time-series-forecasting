from importlibs import *
import glob

from utility import *

regional_ISO_name = 'ISONE'
# prepareDataFor(False, regional_ISO_name, switcher[regional_ISO_name][0], switcher[regional_ISO_name][1], switcher[regional_ISO_name][2], switcher[regional_ISO_name][3] )
prepareDataFromSpecificFolder(False, regional_ISO_name, '05-06-2019', switcher[regional_ISO_name][0], switcher[regional_ISO_name][1], switcher[regional_ISO_name][2], switcher[regional_ISO_name][3] )

priceDatasetPath = '../dataset/'+regional_ISO_name+'/dataset_05-06-2019.csv'
df = pd.read_csv(priceDatasetPath)
isCorrect, clean_df = verify_and_clean_dataset_datetime_wise(df, switcher[regional_ISO_name][0],
                                                    '2019-05-01 00:00:00', '2019-06-30 23:55:00', '5T')
print('isCorrect = ', isCorrect)
clean_df.to_csv(priceDatasetPath, index=False)

