from importlibs import *
import glob

from utility import *

# regional_ISO_name = 'ISONE'
# print(regional_ISO_name + SUB_PICKLE_NAME_PLOT_DATASET)
# f1 = pickle.load(open(regional_ISO_name + SUB_PICKLE_NAME_PLOT_DATASET, 'rb'))
# f1.show()
# plt.show()

regionalISOName='PJM'
prepareDataFor(True, regionalISOName, switcher[regionalISOName][0],
                        switcher[regionalISOName][1],
                        switcher[regionalISOName][2],
                        switcher[regionalISOName][3])
