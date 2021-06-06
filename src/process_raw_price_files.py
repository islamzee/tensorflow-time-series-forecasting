import pandas as pd
from utility import *
from constants import *
import csv
import datetime
import os
from pathlib import *
import numpy as np

def getDaywiseDatasetFor(regionalISOName):
    root_dir = '/dataset/' + regionalISOName
    filename = FILENAME_PRICE_DATASET_JULY_2019

    fullpath = os.getcwd()
    projectPath = Path(fullpath).parents[0]
    datasetDirectoryPath = Path(str(projectPath) + root_dir)
    df = pd.read_csv(os.path.join(datasetDirectoryPath, filename), index_col=switcher[regionalISOName][0])
    df['day'] = pd.to_datetime(df.index).day
    listOfDfDaywise = [x for i, x in df.groupby(df['day'])]

    output_dir = '/output/' + regionalISOName + '/processed_raw_prices_by_day'
    dest_dir = Path(str(projectPath) + output_dir)
    if not os.path.exists(dest_dir):
        dest_dir.mkdir(exist_ok=True)
    for dfx in listOfDfDaywise:
        filename = 'day' + str(dfx.day[0]) + '.csv'
        dfx.to_csv(dest_dir / filename)


getDaywiseDatasetFor('CAISO')
