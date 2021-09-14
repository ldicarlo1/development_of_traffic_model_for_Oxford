import pandas as pd 
import numpy as np 
import ast
from datetime import datetime
import os 



class Import_Traffic_Data():
    ''' This class of functions returns collected traffic data as a numpy array with dims (time, number of indexed road segments, columns).

        Args:

        top (float) : top latitude of bounding box for Oxford location

        bottom (float) :  bottom latitude of bounding box for Oxford location

        right (float) : right longitude of bounding box

        left (float) : left longitude of bounding box
    '''
    def __init__(self,top,bottom,right,left):
        self.top = top
        self.bottom = bottom
        self.right = right
        self.left = left
        self.os = os.getcwd()

    def _get_road_segments(self,top,bottom, right, left):
        # locate a random traffic file and load in using pandas
        dir = f"{self.os}/data_collection/data/traffic_data/"
        filename = f"oxford_traffic_202106230000.csv"
        file = pd.read_csv(dir+filename)

        # Peartree Roundabout bounding-box
        #top=51.798433
        #bottom=51.791451
        #right=-1.281979
        #left=-1.289524

        # convert lats/lons columns to actual python lists
        loni = []
        lati=[]
        for i in range(len(file)):
            lati.append(ast.literal_eval(file.lats.iloc[i]))
            loni.append(ast.literal_eval(file.lons.iloc[i]))

        # create new column with actual python lists of lats/lons
        file['loni'] = loni
        file['lati'] = lati

        # loop thru each row and subset based on the lat/lon boundings, and store the index of each row that is within the specified area
        road_indexes=[]
        for i in range(len(file)):
            lonmin = np.array(file.loni.iloc[i]).min()
            lonmax = np.array(file.loni.iloc[i]).max()
            latmin = np.array(file.lati.iloc[i]).min()
            latmax = np.array(file.lati.iloc[i]).max()
            if latmax<=top and latmin >=bottom and lonmin >= left and lonmax <=right:
                road_indexes.append(file.iloc[i]["Unnamed: 0"])


        # print the number of road segments within the bounding box
        print("The number of road segments within the area of interest is: ",len(road_indexes))

        return road_indexes



    def load_traffic_data(self,datetime_start, datetime_end, verbose=False):
        
        '''
        This function collects the road network traffic data for Oxford, UK on 5-minute intervals from 
        the specified start datetime, end datetime, and index of the data based on a latitude/longitide bounding box.
        
        Args: 
        
        datetime_start (datetime object): specifies the start datetime for creating a range of datetimes
        
        datetime_end (datetime object): specifies the end datetime for creating a range of datetimes
        
        indexes (list-object): specifies the indexes to select from the road network based on a subset of the network
        
        verbose (Bool) : If True print output otherwise returns nothing.
        
        Returns:
        
        traffic matrix: 3d-array (array_obj): dims (time, number of indexed road segments, columns)
        
        time: 1d-array (datetime array-obj): dims (time)

        
        '''
        self.indexes = self._get_road_segments(self.top, self.bottom, self.right, self.left)

        # empty traffic/time dataframe to be filled
        traffic_data = []

        # create date-time range for all file times
        time = pd.date_range(datetime_start,datetime_end,freq="5min")
        datetimestamps = time.strftime("%Y%m%d%H%M")
        
        # create array with shape (time, number of road segments, columns)
        for datetimestamp in datetimestamps:
            if verbose==False:
                pass
            else:
                print(datetimestamp)
            
            dir = f"{self.os}/data_collection/data/traffic_data/"
            filename = f"oxford_traffic_{datetimestamp}.csv"
            traffic_matrix = pd.read_csv(dir+filename)
            
            
            # subset data based on indexes
            traffic_matrix_subset = traffic_matrix[traffic_matrix["Unnamed: 0"].isin(self.indexes)]
            print(traffic_matrix_subset.shape)
            traffic_data.append(traffic_matrix_subset)
        traffic_data = np.array(traffic_data)
        
        return traffic_data,time

