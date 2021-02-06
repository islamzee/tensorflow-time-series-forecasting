import os
from pathlib import Path
import pandas as pd


def getOutputPathForISO(regional_ISO_name):
    root_dir = '/output/' + regional_ISO_name
    fullpath = os.getcwd()
    project_path = Path(fullpath).parents[1]
    return os.path.join(Path(str(project_path) + root_dir))