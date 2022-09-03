# Disaster-Response-Pipeline

Data Scientist Nanodegree Project

# Table of Contents

1. [Installation](#Installation)
2. [File descriptions](#File-descriptions)
3. [Instruction to run code](#Instruction-to-run-code)
3. [Project description](#Project-description)
4. [Acknowledgment](#Acknowledgment)




## Installation
- Machine Learning Libraries: Numpy, Pandas, Sklearn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraries: SQLalchemy
- Web App and Data Visualization: Flask, Plotly

## File descriptions
- App: is where the web app files is stored and it has the main file to run the page which is "run.py" file
- Data: is where the data wrangling cleaning and preparing before applying the ML algorithms and it has the process_data.py file which is to prepare and create the.db file
- Models: is where the training and preparing the ML model and it has train_Classifier.py file which is to train and build the .pkl file is

## Instruction to run code

 Run the following commands to set up the  database and model.
- "python data/process_data.py" for ETL pipeline
- "python models/train_classifier.py" for ML pipeline.

Finally you can run the web app by the following command: "python app/run.py"


# Project description 

This project aims to analyze message data for disaster response. The data was taken from Figure Eight to help them classify disaster messages and run a web app where you can input a new message and get its classification results.

# Acknowledgment 


A special thanks to Figure Eight for providing this important data. Also, I would like to thank Udacity for this great project which taught me too many skills required for data scientists
