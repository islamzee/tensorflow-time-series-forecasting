from importlibs import *
import glob
from utility import *

switcher = {
        'NYISO' :  ['Time Stamp', 'Name', 'LONGIL', NYISO_LBMP_COL_NAME],
        'PJM' : ['Time Stamp', 'Name', 'PJM', NYISO_LBMP_COL_NAME],
        'CAISO' : ['Date', 'hub', 'TH_NP15', 'price' ]
    }
    # in case of ISONE, just append the csv's

regionName = 'CAISO'
input = prepareDataFor(regionName,
                       switcher[regionName][0], switcher[regionName][1], switcher[regionName][2], switcher[regionName][3])
