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