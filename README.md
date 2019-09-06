# Analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages
This repository includes the code to enable generating meaningful insights from messages during natural disasters and to classify them. Includes 3 aspects
## 1. Analysis of disaster message data
The objective is to Analyze disaster message data from Figure Eight to build a model for an API that classifies disaster messages.
Analyze data set containing real messages that were sent during disaster events.
Here real disaster messages are analyzed for data quality and an ETL pipeline is created to clean data, remove duplicates and other extraneous data and the resultant clean data is loaded onto a database for further retrieval

Libraries required: pandas, sqlalchemy, re
## 2. Create a machine learning pipeline to categorize events
Create a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.
Here the messages are analyzed through Natural Language processing toolkits. The messages are stripped of punctuation, normalized, and then common English stopwords are removed. The modified messages are then tokenized, and lemmatized.
A MultiOutputClassifier is then trained to classify the messages into various disaster-related categories (Each message could correspond to multiple categories). Metrics such as precision, recall, f1 are calculated to determine the efficacy of the classifier.
The best hyperparameters are then determined to identify the best model which is then pickled.

Libraries required: nltk, sklearn, joblib
## 3. Web app for emergency workers
Web app for emergency workers to input a new message and get classification results in several categories. 
The web app displays visualizations of the data



