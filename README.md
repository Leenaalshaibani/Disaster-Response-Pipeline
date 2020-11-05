# Disaster-Response-Pipeline
# Introduction
The repo is a project in the Udacity Data Scientist Nanodegree Program.

The dataset (in the data folder) comes from appen which contains about 26k text messages from news, social media, and some other sources when some disasters happened.

Our goal here is to build a machine learning model to identify if these messages are related to disaster or not, and further label the nature of these messages. This would be of great help for some disaster relief agencies. We have 36 labels for these messages in total. Note, however, these labels are not mutually exclusive. Hence it is a multi-label classification problem.

The most obvious feature of those data messages is they are highly imbalanced. Several categories getting very few labels. To improve the accuracy, we implement a up-sample scheme before training.

After building and training such a model, we can next launch a web service which can label new messages from users' input.

# The Project contains three components mainly,
# 1. ETL Pipeline
Python script, process_data.py,
Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database
# 2. ML Pipeline
Python script, train_classifier.py,
Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file
# 3. Flask Web App
Outputs the classification according to the input
