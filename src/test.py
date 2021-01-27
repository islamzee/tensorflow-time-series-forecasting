from importlibs import *
import glob

from lstm_proper import lstmModelFileExists
from utility import *

switcher = {
        'NYISO' :  ['Time Stamp', 'Name', 'LONGIL', 'LBMP ($/MWHr)'],
        'PJM' : ['Time Stamp', 'Name', 'PJM', 'LBMP ($/MWHr)'],
        'CAISO' : ['Date', 'hub', 'TH_NP15', 'price'],
        'ISONE': ['Time', None, None, '$/MWh']
    }
    # in case of ISONE, just append the csv's

regional_ISO_name = 'ISONE'
# input = prepareDataFor(True, regional_ISO_name, switcher[regional_ISO_name][0], switcher[regional_ISO_name][1], switcher[regional_ISO_name][2], switcher[regional_ISO_name][3])

print('model file exists: ', lstmModelFileExists(regional_ISO_name))