from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import seaborn as sns
import ast
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import seaborn as sns
import ast
import matplotlib as mpl
import matplotlib.pyplot as plt

class spatio_temporal_traffic_model():
    def __init__(self, model=lr):
        ''' Input machine learning algorithm to produce model perfomance statistics for spatial traffic data. Default is linear regression.
        '''
        self.model = model
        
    def fit(self, X_train, y_train, X_test, y_test):
        '''This function collects the training and testing data for the modeling and performs modeling.
        
        Args:
        
            X_train (array-obj) : training features // dims (time, road-segments, sequence length)
            
            y_train (array-obj) : training target values // dims (time, nodes)
            
            X_test (array-obj) : testing features // dims (time, road-segments, sequence length)
            
            y_test (array-obj) : test target values // dims (time, nodes)
            
        '''
        
        self.y_test = y_test
        
        # get the number of road segments
        num_road_seg = X_train[0,:,0]
        
        # make empty list to populate with predictions from each road segment
        predictions = []
        
        # loop thru each road segment and train the ML algorithm
        for road_seg in range(len(num_road_seg)):
            
            # model the ML algorithm with the training/testing data
            self.model.fit(X_train[:,road_seg,:], y_train[:,road_seg])
            
            # make predictions on test data
            predictions.append(self.model.predict(X_test[:,road_seg,:]))
            
        self.predictions = np.array(predictions)
            
      
    def r2(self,true, predictions):
        '''Return the r2 score across the entire road network.
        '''
        
        r2=[]
        for road_seg in range(predictions.shape[1]):

            # r2-score calculation using sklearn
            r2.append(r2_score(true[road_seg,:], predictions[:,road_seg]))
            
        r2 = np.array(r2)
        
        return r2
            
    def mae(self, true, predictions):
        '''Return the mean-absolute error across the entire road network.
        '''
        
        mae=[]
        for road_seg in range(predictions.shape[1]):

            # mae calculation using sklearn
            mae.append(mean_absolute_error(true[road_seg,:], predictions[:,road_seg]))
            
        mae = np.array(mae)
        
        return mae
        
    def mse(self, true, predictions):
        '''Return the root mean squared error across the entire road network.
        '''
        
        mse=[]
        for road_seg in range(predictions.shape[1]):

            # rmse calculation using sklearn
            mse.append(mean_squared_error(true[road_seg,:], predictions[:,road_seg]))
            
        mse = np.array(mse)
        
        return mse       
    
    def plot_mean_road_speed_performance(self, machine_learning_method, time,predictions=None,true=None,window=None):
        '''This function produces a plot of the performance of the model by comparing the average predictions with the
        average of the true values.
        
        Args:
            machine_learning_method (str) : simply adds the ML method to the plot title
            
            time (datetime-arr) : array of all datetimes

            predictions (array-obj) : allows to input predictions and truths that are calculated outside of this class
            of functions and produce the same plots. Default is None.
            
            true (array-obj) : allows to input predictions and truths that are calculated outside of this class
            of functions and produce the same plots. Default is None.

            window (int) : allows the users to specify a time window to focus the plot on to gauge more pinpoint results.
            The default is the entire testing time period. Otherwise an integer can be input to represent the number of 5-minute
            time steps into the future to plot.
            
        '''
        
        # if no external values are specified, the plots are generated with the data originally declared in this class of funcs

        if (type(predictions) == type(None)) & (type(true) == type(None)):
            # get the mean predictions for the road network and mean true values
            predictions = self.predictions
            true = self.y_test
            
            mean_pred = np.mean(predictions,axis=0)
            mean_true = np.mean(true,axis=1)
        else:
            mean_pred = np.mean(predictions,axis=0)
            mean_true = np.mean(true,axis=1)
        
        # select window of time to plot, and index the plots accordingly
        if window==None:
            end_index = None
        else:
            end_index = (-len(mean_true)+window)
        
        # produce plot
        fig = plt.figure(figsize=(18, 8))
        sns.set_theme()
        plt.plot(time[-len(mean_true):end_index],mean_true[:end_index],c="violet",label="truth")
        plt.plot(time[-len(mean_true):end_index],mean_pred[:end_index],c="darkmagenta",label="predictions")
        plt.xlabel("Datetime")
        plt.ylabel("Speed (Normalized)")
        plt.legend(loc="upper right")
        plt.title(f"Mean Model Performance for the Road Network: {machine_learning_method}")
        plt.axis('on')
        plt.savefig(f"figures/mean_performance_{machine_learning_method}")
        plt.show()
        
        # return metrics
        print(f"MSE: {self.mse(true, predictions).mean()}")
        print(f"MAE: {self.mae(true, predictions).mean()}")
        print(f"R2: {self.r2(true, predictions).mean()}")
        
        return None
    
    def plot_specific_road_speed_performance(self, machine_learning_method, time, road_seg_num, true=None,predictions=None,window=None):
        '''This function produces a plot of the performance of the model for one specific road segment.
        
        Args:
            machine_learning_method (str) : simply adds the ML method to the plot title
            
            time (datetime-arr) : array of all datetimes

            road_seg_num (int) : The road segment to plot (0-70).
            
            predictions (array-obj) : allows to input predictions and truths that are calculated outside of this class
            of functions and produce the same plots. Default is None.
            
            true (array-obj) : allows to input predictions and truths that are calculated outside of this class
            of functions and produce the same plots. Default is None.
            
            window (int) : allows the users to specify a time window to focus the plot on to gauge more pinpoint results.
            The default is the entire testing time period. Otherwise an integer can be input to represent the number of 5-minute
            time steps into the future to plot.
        '''
        
        # if no external values are specified, the plots are generated with the data originally declared in this class of funcs
        if (type(predictions) ==type(None)) & (type(true) ==type(None)):
            # get the mean predictions for the road network and mean true values
            predictions = self.predictions
            true = self.y_test
        else:
            pass
        
        # select window of time to plot, and index the plots accordingly
        if window==None:
            end_index = None
        else:
            end_index = (-len(true)+window)
        # produce plot
        fig = plt.figure(figsize=(18, 8))
        sns.set_theme()
        plt.plot(time[-len(true):end_index],true[:end_index,road_seg_num],c="violet",label="truth")
        plt.plot(time[-len(true):end_index],predictions[road_seg_num,:end_index],c="purple",label="predictions")
        plt.xlabel("Datetime")
        plt.ylabel("Speed (Normalized)")
        plt.legend(loc="upper right")
        plt.title(f"Road Segment {road_seg_num} Model Performance for the Road Network: {machine_learning_method}")
        plt.axis('on')
        plt.savefig(f"figures/{machine_learning_method}_performance_road_segment_{road_seg_num}")
        plt.show()
        
        # return metrics for the road segment
        print(f"MSE for Road Segment {road_seg_num}: {self.mse(true,predictions)[road_seg_num]}")
        print(f"MAE for Road Segment {road_seg_num}: {self.mae(true,predictions)[road_seg_num]}")
        print(f"R2 for Road Segment {road_seg_num}: {self.r2(true,predictions)[road_seg_num]}")
        
        return None
    
    def geoplot_metric(self, machine_learning_method, true=None, predictions=None, method="MSE"):
        '''This function creates a geoplot of the Peartree roundabout mean-squared error by road segment.
        '''
        
        # if no external values are specified, the plots are generated with the data originally declared in this class of funcs
        if (type(predictions) ==type(None)) & (type(true) ==type(None)):
            # get the mean predictions for the road network and mean true values
            predictions = self.predictions
            true = self.y_test
        else:
            pass
        
        # determine metric
        if method == "MSE":
            metric = self.mse(true, predictions)
        elif method == "MAE":
            metric = self.mae(true, predictions)
        elif method == "R2":
            metric = self.r2(true, predictions)
        else:
            print("Please re-enter a valid metric.")
        
        
        # generate color scheme for mse
        colors = ["maroon","darkred","red","orangered","peru","orange","gold","yellow","green","lightgreen","greenyellow"]

        # if metric is MSE/MAE reverse the color order 
        if (method == "MSE") | (method=="MAE"):
            colors = colors[::-1]
        else:
            pass
        
        print(colors)
        # create linearly spaced intervals for 10 colors
        color_vals = np.linspace(metric.min(),metric.max(),10)
        
        # Create geographic plot of road network 
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.grid(False)
        for i in range(0,len(metric)):
            plt.xlim(-1.281979,-1.289524)
            
            # values
            color_val= metric[i]
            if color_val <color_vals[0]:
                color = colors[0]
            elif (color_val >=color_vals[0]) & (color_val <color_vals[1]):
                color = colors[1]
            elif (color_val >=color_vals[1]) & (color_val <color_vals[2]):
                color = colors[2]
            elif (color_val >=color_vals[2]) & (color_val <color_vals[3]):
                color = colors[3]
            elif (color_val >=color_vals[3]) & (color_val <color_vals[4]):
                color = colors[4]
            elif (color_val >=color_vals[4]) & (color_val <color_vals[5]):
                color = colors[5]
            elif (color_val >=color_vals[5]) & (color_val <color_vals[6]):
                color = colors[6]
            elif (color_val >=color_vals[6]) & (color_val <color_vals[7]):
                color = colors[7]
            elif (color_val >=color_vals[7]) & (color_val <color_vals[8]):
                color = colors[8]
            elif (color_val >=color_vals[8]) & (color_val <color_vals[9]):
                color = colors[9]
            else:
                color = colors[10]

            plt.plot(ast.literal_eval(lons[i]),ast.literal_eval(lats[i]), c=color,linewidth=4)

        plt.axis('off')
        plt.title(f"{machine_learning_method}: Peartree Roundabout {method}")
        plt.gca().invert_xaxis()
        plt.style.use('dark_background')
        plt.style.use('dark_background')
        plt.show()
        
        # COLORBAR

        fig, ax = plt.subplots(figsize=(10, 1))
        fig.subplots_adjust(bottom=0.5)
        cmap = mpl.colors.ListedColormap(colors)
        bounds = color_vals
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cb3 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                        norm=norm,
                                        extend='both',
                                        extendfrac='auto',
                                        ticks=bounds,
                                        spacing='uniform',
                                        orientation='horizontal')
        cb3.set_label(method)
        fig.show()
        
        