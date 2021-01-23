from importlibs import *
import glob
from utility import *

switcher = {
        'NYISO' :  ['Name', 'LONGIL'],
        'PJM' : ['Name', 'PJM'],
        'CAISO' : ['hub', 'TH_NP15']
    }
    # in case of ISONE, just append the csv's

regionName = 'CAISO'
input = prepareDataFor(regionName, switcher[regionName][0], switcher[regionName][1])