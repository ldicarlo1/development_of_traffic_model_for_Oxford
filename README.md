# Development of a predictive traffic model for Oxford, UK using a Temporal Graph Convolutional Neural Network and weather variables (T-GCN-wx)
##### Author: Luca Di Carlo
##### Date: September 2021
##### Location: Oxford, UK
##### Organization: Oxford Brookes University

## Project Overview
A proof of concept predictive traffic model was developed for a locality of Oxford, UK using HERE® Traffic API data and Tomorrow.io API weather observations. 
The traffic model was developed using a deep hybrid neural network, and was compared against a three other simpler machine learning algorithms.

## File Structure
##### analysis/  
  ###### Contains Jupyter Notebooks for analyzing both the weather and traffic data collected. Also a graph representation of the road network locality of Oxford (Peartree Roundabout) and the excel sheet of the road connections necessary for the graph representation.
  
##### classes/  
  ###### Contains classes of functions necessary for data collection, preprocessing, and modeling.
  
##### data_collection/  
  ###### Contains scripts necessary for data collection. DATA FOLDER DOES NOT CONTAIN ENTIRE DATASET USED, BECAUSE FULL DATASET WAS TOO LARGE TO UPLOAD (>16 GB). ONLY SUBSET OF DATA AVAILABLE IN THIS REPOSITORY.
  
##### data_collection_full_data/  
  ###### This folder contains the complete data for this project, and if it is downloaded, then remove the folder 'data_collection/' and rename this folder 'data_collection/' for the python notebooks and scripts to work.
 
##### data_preprocessing/  
  ###### Contains preprocessing functions and excel sheet of the Peartree roundabout road connections.
  
##### figures/  
  ###### Contains figures for the dissertation research report.
  
##### modeling/  
  ###### Contains Jupyter Notebooks, classes of functions, models, and hyperparameter tuning results all necessary for the development of the traffic models for varying temporal prediction lengths. Four models were developed and the Jupyter Notebooks containing the development of the Linear Regression, LSTM,  T-GCN, and T-GCN-wx models are all in their respective folders. The script 'modeling_5min_predictions.py' is an example script of all of the code combined  necessary to produce the four models to predict traffic 5-minutes into the future. The code in this script is pulled from the Jupyter Notebooks.

##### performance_analysis/  
  ###### Contains the Jupyter Notebooks necessary for the analysis of the traffic models.
  
  


## Methodology
A hybrid deep neural network was employed to make short-term traffic predictions across a variety of prediction horizons using both traffic and weather data. This 
complex architecture (T-GCN-wx) was compared against three other modeling approaches, linear regression, LSTM, and T-GCN to critically analyze the candidate model. 
Only the T-GCN-wx method involved weather and traffic data, with the other remaining three only using traffic data alone. This allowed for a fair comparison against 
the effectiveness of weather data. 

## Development of the T-GCN-wx Hybrid Traffic Prediction Model
##### T-GCN Concept
The T-GCN-wx is the candidate model and focus for this research, as it incorporates both traffic and weather data using a deep hybrid neural network structure. This 
structure is heavily based on the T-GCN architecture developed by Zhao et al. (2019) which involved using a GCN to process the graph structure of the road network, 
and a GRU to process the temporal features of the data. Stellargraph (Data61, 2018) is a Python library that provides a built-in function which is inspired off of 
the paper by Zhao, using a GCN combined with an LSTM to make traffic predictions. As mentioned above, LSTM and GRU have similar performance and therefore and using 
LSTM would not pose a performance issue.

The structure of a 1-layer GCN can be represented as the following:

![test image size](https://github.com/ldicarlo1/development_of_traffic_model_for_Oxford/blob/main/photos/Screen%20Shot%202021-10-01%20at%204.03.37%20PM.png)


  
In addition to the feature matrix being input to the GCN layer, the adjacency matrix also must be input for the GCN to understand the connections in the temporal 
graph. The output of the GCN layers then passes to the LSTM, where it is stored in both short and long-term memory as it propagates though each cell. The result 
produces a traffic prediction N steps into the future.

##### Incorporating Weather Data
A method of data fusion was implemented to incorporate weather data. A review of data fusion techniques resulted in choosing a data-in data-out (DAI DAO) data 
fusion technique (Castanedo, 2013). This was also adopted by Essien et al. (2021) where a traffic model combined weather variables and traffic speeds in the first
layer of a deep neural network. Using a multidimensional input, the traffic data and weather variables were fused in the second layer. This tends to lead to more 
reliable outputs as the errors introduced at the prediction level are avoided.
The traffic data and weather data were combined into a four-dimensional input matrix, with the extra dimension being the five features (four weather features one 
traffic feature). Using an input layer with five dimensions, this multidimensional array can be fed into a neural network. The second layer reduces the five 
dimensions to one, hence “fusing” the data which can be better represented in Figure 16.


![test image size](https://github.com/ldicarlo1/development_of_traffic_model_for_Oxford/blob/main/photos/Screen%20Shot%202021-10-01%20at%204.03.47%20PM.png)


##### Model Architecture
The complete model architecture is visualized in Figure 17 which includes the data fusion layer and the complete T-GCN. Data enters the model with four dimensions: time, number of nodes, input sequence length, and the number of features. The data is dimensionally reduced after passing through a dense layer of neurons, and batch normalization and a dropout layer are applied to prepare the data to be fed into the T- GCN. At this phase, the traffic data is fused together with the weather data. The adjacency matrix also is input into the T-GCN model at this phase. Upon entering the T- GCN, the data passes through a GCN layer which spatially filters the data before it is reshaped to enter the LSTM layer. After the LSTM layer the data passes through a dropout layer and one more neuron layer before the final forecast is produced.

![test image size](https://github.com/ldicarlo1/development_of_traffic_model_for_Oxford/blob/main/photos/Screen%20Shot%202021-10-01%20at%204.03.23%20PM.png)
 
## Performance
Model performance boasted low MSE, MAE, and high R2 scores for all models, however the deep learning model performed best. There was little difference between 
the T-GCN and T-GCN-wx models, as the weather data offered no significant improvement in traffic prediction. 

![test image size](https://github.com/ldicarlo1/development_of_traffic_model_for_Oxford/blob/main/photos/Screen%20Shot%202021-10-01%20at%204.10.59%20PM.png)


However there was an evident lag in predicted traffic speeds that became less evident the further in time the prediction was.


![test image size](https://github.com/ldicarlo1/development_of_traffic_model_for_Oxford/blob/main/photos/Screen%20Shot%202021-10-01%20at%204.11.27%20PM.png)
