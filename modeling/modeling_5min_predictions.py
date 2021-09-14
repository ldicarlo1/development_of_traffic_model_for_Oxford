# change directory to parent folder to access all folders
import os
path = os.path.dirname(os.getcwd())
os.chdir(path)
from data_preprocessing.classes.load_traffic_data import Import_Traffic_Data

import networkx as nx
import pandas as pd
import numpy as np
import ast
import math
import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense,concatenate
from stellargraph import StellarGraph, StellarDiGraph
from stellargraph.layer import GCN_LSTM
import stellargraph as sg
from datetime import datetime
from keras.models import Sequential, load_model
from keras.layers import LSTM
from pmdarima.model_selection import train_test_split
import seaborn as sns
from classes import model_performance,preprocessing
import pickle
import time as timex


################################################
# Load traffic data
################################################
# Peartree roundabout bbox and datetimes of interest
top=51.798433
bottom=51.791451
right=-1.281979
left=-1.289524
datetime_start=datetime(2021,6,23,0,0)
datetime_end=datetime(2021,7,13,10,50)

# load in traffic data
traffic_data,time = Import_Traffic_Data(top,bottom,right,left).load_traffic_data(datetime_start,datetime_end)


################################################
# Load weather data
################################################
# load in 5min wx data from csv
wx_df = pd.read_csv("data_collection/data/wx_data/oxfordcity_wx_variables_5min_intervals.csv")

# collect variables of significance
wx_vars = wx_df[['precipitationIntensity','temperature','humidity','weatherCode']]

wx_vars_scaled = np.zeros_like(wx_vars)

# normalize between 0 and 1
for i in range(4):
    norm = (wx_vars.iloc[:,i] - wx_vars.iloc[:,i].min())/(wx_vars.iloc[:,i].max() - wx_vars.iloc[:,i].min())
    wx_vars_scaled[:,i] = norm
    #print(wx_vars_scaled[i].max())
    
    
# transpose data to be in proper format for preprocessing   
wx_vars_scaled = wx_vars_scaled.T




################################################
# Create graph
################################################

# load in csv of node connections
connections = pd.read_csv(f"{path}/data_preprocessing/peartree_roundabout.csv")
connections.head(5)

# convert feeding roads to integers
for i in range(len(connections)):
#for i in range(4):
    try:
        connections.feeding_roads.iloc[i] = ast.literal_eval(connections.feeding_roads.iloc[i])
    except ValueError:
        connections.feeding_roads.iloc[i] = np.nan

# node connections
nodes = connections["Unnamed: 0"]
roads = connections.feeding_roads

# replace nans with 0's
connections.feeding_roads = connections.feeding_roads.fillna(0)

# loop thru and establish edges
edge_list = []
for row in range(len(roads)):
    node1 = connections["Unnamed: 0"].iloc[row]
    node2 = connections.feeding_roads.iloc[row]
    try:
        for i in range(len(node2)):
            edge_list.append([node2[i], node1])
        #node2 = connections.feeding_roads.iloc[row]
    except TypeError:
        edge_list.append([node2, node1])
        
# remove 0's
edges = []
for edge in edge_list:
    if edge[0]==0:
        pass
    else:
        edges.append(edge)  

#build the graph
G = nx.Graph()
for i in range(len(nodes)):
    G.add_node(nodes[i],spd=sp[:,i])
G.add_edges_from(edges)

# get adjacency matrix 
A = nx.to_numpy_array(G)

# convert graph to stellargraph object for modeling
square = StellarGraph.from_networkx(G,node_features="spd")

# get feature matrix
X = square.node_features()

################################################
# Preprocess data
################################################
# specify the training rate
train_rate = 0.8

# replace missing values with nans
X = np.where(X<0,0,X)

# split train/test
train_data, test_data = preprocessing.train_test_split(X, train_rate)
wx_train_data, wx_test_data = preprocessing.train_test_split(wx_vars_scaled, train_rate)

print("Train data: ", train_data.shape)
print("Test data: ", test_data.shape)

# scale data based on max/min
train_scaled, test_scaled = preprocessing.scale_data(train_data, test_data)

# create new train/test variables for wx variables
wx_train_data_ = []
wx_test_data_ = []

# loop thru and assign the wx data to each node
for i in range(70):
    wx_train_data_.append(wx_train_data.T)
    wx_test_data_.append(wx_test_data.T)

# convert data to correct shape
wx_train_data_ = np.array(wx_train_data_)
wx_test_data_ = np.array(wx_test_data_)

################################################
# 5-min prediction length modeling preprocessing
################################################
# the number of timesteps up to the prediction that we will feed to the model (5-minute intervals)
seq_len = 12

# the amount of time in advance we want to predict (5-minute intervals)
pre_len = 1

# preprocess traffic data
traffic_trainX, trainY, traffic_testX, testY = preprocessing.sequence_data_preparation(
    seq_len, pre_len, train_scaled, test_scaled
)

# preprocessing weather data
wx_trainX, wx_trainY, wx_testX, wx_testY = preprocessing.sequence_data_preparation(
    seq_len, pre_len, wx_train_data_, wx_test_data_
)

#Combine the weather variables w/ traffic matrix to create feature matrix
trainX = np.empty((len(wx_trainX[:,0,0,0]),len(wx_trainX[0,:,0,0]),len(wx_trainX[0,0,:,0]),5) )
testX = np.empty((len(wx_testX[:,0,0,0]),len(wx_testX[0,:,0,0]),len(wx_testX[0,0,:,0]),5) )

trainX[:,:,:,0] = traffic_trainX
trainX[:,:,:,1:5] = wx_trainX
testX[:,:,:,0] = traffic_testX
testX[:,:,:,1:5] = wx_testX

################################################
# Generate models
################################################


############## Linear Regression ##################
# define linear regression model
lr = LinearRegression()

# get the number of road segments
num_road_seg = trainX[0,:,0]

# make empty list to populate with predictions from each road segment
predictions = []

# loop thru each road segment and train the ML algorithm
for road_seg in range(len(num_road_seg)):

    # model the ML algorithm with the training/testing data
    lr.fit(trainX[:,road_seg,:], trainY[:,road_seg])

# save the model to disk
filename = 'modeling/models/lr-5min.sav'
pickle.dump(lr, open(filename, 'wb'))


############## LSTM ##################
# define model
model_lstm = Sequential()
model_lstm.add(LSTM(200, activation='linear', input_shape=(70, 12,)))
model_lstm.add(Dense(70))

# compile model
optimizer = keras.optimizers.Adam(lr=0.001)
model_lstm.compile(optimizer=optimizer, loss="mse", metrics=["mse"])

# print summary
model_lstm.summary()
sg.utils.plot_history(history)

# save model to folder
model_lstm.save('modeling/models/lstm-5min')

############## T-GCN ##################
gcn_lstm = GCN_LSTM(
    seq_len=seq_len,
    adj=A,
    gc_layer_sizes=[10],
    gc_activations=["linear"],
    lstm_layer_sizes=[200],
    lstm_activations=["linear"],
    dropout=0.0,
)
# model architecture with keras
x_input, x_output = gcn_lstm.in_out_tensors()
model_tgcn = Model(inputs=x_input, outputs=x_output)

# compile model
optimizer = keras.optimizers.Adam(lr=0.001)
model_tgcn.compile(optimizer=optimizer, loss="mse", metrics=["mse"])

history = model_tgcn.fit(
    x=trainX,
    y=trainY,
    epochs=75,
    batch_size=64,
    shuffle=True,
    verbose=1,
    validation_data=(testX,testY)
)
# plot loss 
sg.utils.plot_history(history)

# save model to folder
model_tgcn.save('modeling/models/tgcn-5min')

############## T-GCN-wx ##################
gcn_lstm = GCN_LSTM(
    seq_len=seq_len,
    adj=A,
    gc_layer_sizes=[10],
    gc_activations=["linear"],
    lstm_layer_sizes=[200],
    lstm_activations=["linear"],
    dropout=0.1
)
# build data fusion layer which will merge wx/traffic feature matrix (length 5) into one array to be fed into the t-gcn model
input_layer = Input(shape=(70,12,5))
layer1 = Dense(1, activation='linear')(input_layer)
layer2 = BatchNormalization()(layer1)
layer3 = Dropout(0.1)(layer1)
output_layer = Reshape((70,12))(layer3)
data_fusion_model = Model(input_layer,output_layer)

# recall tgcn model
x_input, x_output = gcn_lstm.in_out_tensors()
tgcn_model = Model(inputs=x_input, outputs=x_output)


# of data fusion model feeds into t-gcn
output = tgcn_model(data_fusion_model.output)

# define entire model 
tgcn_wx_model = Model(data_fusion_model.input,output, name="T-GCN-WX")
tgcn_wx_model.summary()

# compile model
optimizer = keras.optimizers.Adam(lr=0.001)
tgcn_wx_model.compile(optimizer=optimizer, loss="mse", metrics=["mse"])

history = tgcn_wx_model.fit(
    x=trainX,
    y=trainY,
    epochs=100,
    batch_size=64,
    shuffle=True,
    verbose=1,
    validation_data=(testX,testY)
)

sg.utils.plot_history(history)

# save model to folder
tgcn_wx_model.save('modeling/models/tgcn_wx-5min')




