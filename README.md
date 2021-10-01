# Development of a predictive traffic model for Oxford, UK using a Temporal Graph Convolutional Neural Network and Weather Data (T-GCN-wx)
##### Author: Luca Di Carlo
##### Date: September 2021
##### Location: Oxford, UK
##### Organization: Oxford Brookes University

## Project Overview
A proof of concept predictive traffic model was developed for a locality of Oxford, UK using HERE® Traffic API data and Tomorrow.io API weather observations. 
The traffic model was developed using a deep hybrid neural network, and was compared against a three other simpler machine learning algorithms.

## File Structure
###### analysis/  
  Contains Jupyter Notebooks for analyzing both the weather and traffic data collected. Also a graph representation of the road network locality of
  Oxford (Peartree Roundabout) and the excel sheet of the road connections necessary for the graph representation.
  
###### classes/  
  Contains classes of functions necessary for data collection, preprocessing, and modeling.
  
###### data_collection/  
  Contains scripts necessary for data collection. DATA FOLDER DOES NOT CONTAIN ENTIRE DATASET USED, BECAUSE FULL DATASET WAS TOO LARGE TO UPLOAD (>16 GB).
  ONLY SUBSET OF DATA AVAILABLE IN THIS REPOSITORY.
  
###### data_collection_full_data/  
  This folder contains the complete data for this project, and if it is downloaded, then remove the folder 'data_collection/' and rename this folder 'data_collection/' for the python notebooks and scripts to work.
 
###### data_preprocessing/  
  Contains preprocessing functions and excel sheet of the Peartree roundabout road connections.
  
###### figures/  
  Contains figures for the dissertation research report.
  
###### modeling/  
  Contains Jupyter Notebooks, classes of functions, models, and hyperparameter tuning results all necessary for the development of the traffic models for 
  varying temporal prediction lengths. Four models were developed and the Jupyter Notebooks containing the development of the Linear Regression, LSTM, 
  T-GCN, and T-GCN-wx models are all in their respective folders. The script 'modeling_5min_predictions.py' is an example script of all of the code combined 
  necessary to produce the four models to predict traffic 5-minutes into the future. The code in this script is pulled from the Jupyter Notebooks.

###### performance_analysis/  
  Contains the Jupyter Notebooks necessary for the analysis of the traffic models.
  
  


## Methodology
A hybrid deep neural network was employed to make short-term traffic predictions across a variety of prediction horizons using both traffic and weather data. This 
complex architecture (T-GCN-wx) was compared against three other modeling approaches, linear regression, LSTM, and T-GCN to critically analyze the candidate model. 
Only the T-GCN-wx method involved weather and traffic data, with the other remaining three only using traffic data alone. This allowed for a fair comparison against 
the effectiveness of weather data. In this section first the workflow of the project will be presented, followed by a discussion of the architectures of the 
candidate model and the three benchmark methods, along with justifications of the methods, the training and testing of each model, hyperparameters, and the metrics 
that will be used in the analysis.
  
  
   
 
    
 
