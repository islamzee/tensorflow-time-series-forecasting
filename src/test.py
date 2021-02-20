from importlibs import *
import glob

from utility import *

regional_ISO_name = 'PJM'
# prepareDataFor(False, regional_ISO_name, switcher[regional_ISO_name][0], switcher[regional_ISO_name][1], switcher[regional_ISO_name][2], switcher[regional_ISO_name][3] )


df = pd.read_csv('../labels/PJM/dataset.csv')
isCorrect, clean_df = verify_and_clean_dataset_datetime_wise(df, '2020-12-01 00:00', '2020-12-02 00:00', '5T')
print('iscorrect = ', isCorrect)
print(clean_df)

