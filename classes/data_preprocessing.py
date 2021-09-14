import pandas as pd 
import numpy as np 
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas
import ast
from skimage import data, color


def _geographic_subset_data():
    '''This function returns a subset of the Oxford traffic data for Oxford by indexing based on a lon/lat bbox.
    '''

    # loop thru and convert lat/lon strings to python lists
    loni = []
    lati=[]
    for i in range(len(traffic_matrix)):
        lati.append(ast.literal_eval(traffic_matrix.lats.iloc[i]))
        loni.append(ast.literal_eval(traffic_matrix.lons.iloc[i]))

    traffic_matrix['lons'] = loni
    traffic_matrix['lats'] = lati

    # index the data based on the lat/lon bounding coordinates provided in the initialization of the function.
    bbox_data=[]
    for i in range(len(traffic_matrix)):
        lonmin = np.array(traffic_matrix.lons.iloc[i]).min()
        lonmax = np.array(traffic_matrix.lons.iloc[i]).max()
        latmin = np.array(traffic_matrix.lats.iloc[i]).min()
        latmax = np.array(traffic_matrix.lats.iloc[i]).max()
        if latmax<=top and latmin >=bottom and lonmin >= left and lonmax <=right:
            bbox_data.append(traffic_matrix.iloc[i].values)
    bbox_data = pd.DataFrame(np.array(bbox_data))
    bbox_data.columns = traffic_matrix.columns

    return bbox_data

