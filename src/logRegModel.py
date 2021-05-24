#!/usr/bin/env python
"""
Specify file path of the csv file of Game of Thrones lines and name of the output cross validation graphs. You can also specify whether to balance the data, the size of the test data in percentage points and vectorization method. The output will be a classification matrix saved as png, a classification report saved as csv (this will also be printed in the terminal) and a cross validation graph saved as png. These will all be saved in a folder called output in the parent directory of the working directory.
Parameters:
    filepath: str <filepath-of-csv-file>, default = '../data/Game_of_Thrones_Script.csv'
    output_filename: str <name-of-png-file>, default = 'logReg_cross_validation.png'
    balance: str <balance-data>, default = 'True'
    test_size: float <size-of-test-data>, default = 0.25
    vectorizer: str <vectorization-method>, default = 'tfidf'
Usage:
    logRegModel.py -f <filepath-of-csv-file> -o <name-of-png-file> -b <balance-data> -t <size-of-test-data> -v <vectorizer>
Example:
    $ python3 logRegModel.py -f ../data/Game_of_Thrones_Script.csv -o LogReg_cross_validation.png -b True -t 0.25, -v tfidf
    
## Task
- Train a multiple logistic regression model to classify lines from GoT characters into their respective seasons (1-8). 
- Output is in the form of two png files (classification matrix and cross validation graphs) and a csv file of the classification report which will also be printed in the terminal. All outputs can be found in the path "../output".
"""

# libraries
# system tools
import os
import sys
sys.path.append(os.path.join(".."))

# pandas, numpy, gensim
import pandas as pd
import numpy as np
#import gensim.downloader

# import my classifier utility functions - see the Github repo!
import utils.classifier_utils as clf

# Machine learning stuff
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
#from sklearn.preprocessing import LabelBinarizer

# matplotlib
import matplotlib.pyplot as plt

# argparse
import argparse

# argparse 
ap = argparse.ArgumentParser()
# adding argument
# path to csv
ap.add_argument("-f", "--filepath", 
                default = "../data/Game_of_Thrones_Script.csv", 
                help= "Path to the Game of Thrones csv-file")
# output filename
ap.add_argument("-o", "--output_filename", 
                default = "logReg_cross_validation.png", 
                help = "Name of output file")
# balance the data
ap.add_argument("-b", "--balance", 
                default = True, 
                help = "Balance the data? Choose between 'True' or 'False'")
# test size
ap.add_argument("-t", "--test_size", 
                default = 0.25, 
                help = "Size of test data")
# vectorization method
ap.add_argument("-v", "--vectorizer",
                default = "tfidf",
                help = "Vectorization method - Choose between: 'count' and 'tfidf'")

# parsing arguments
args = vars(ap.parse_args())


def main(args):
    # get variables from argparse
    filepath = args["filepath"]
    out_name = args["output_filename"]
    balance = args["balance"]
    test_size = args["test_size"]
    vectorize = args["vectorizer"]
    
    # Create out directory if it doesn't exist in the data folder
    dir_name = os.path.join("..", "output")
    create_dir(dirName = dir_name)
    
    # load data and balance class distribution if specified as True
    got = load_and_prepare(filepath = filepath, 
                           balance = balance)
    
    # Start message to user
    print("\n[INFO] Initializing the construction of a logistic regression model...")
    # build model and 
    classifier, X_test_feats, y_test, sentences, seasons = logRegModel(data = got,
                                                                       test_size = test_size, 
                                                                       vectorize = vectorize)
    
    
    print("\n[INFO] Evaluating the logistic regression model...")
    # build model and 
    evaluate_model(classifier = classifier, 
                   X_test = X_test_feats,
                   y_test = y_test)
     
    
    print("\n[INFO] Running cross validation on the logistic regression model...")
    cross_validation(sentences = sentences, 
                     seasons = seasons, 
                     out_name = out_name, 
                     vectorize = vectorize)
    
    
    # end message
    print("\nYou have now successfully trained and evaluated a logistic regression model on the Game of Thrones data. Have a nice day!")

def load_and_prepare(filepath, balance):
        # load the data with pandas
    got = pd.read_csv(filepath)
    
    # if the user has chosen to balance data
    if balance == True:
        # create empty data frame
        column_names = ["Release Date", "Season", "Episode", "Episode Title", "Name", "Sentence"]
        new_df = pd.DataFrame(columns = column_names)
        
        # for each unique season
        for season in got["Season"].unique():
            # subset data on season and save as temporary dataframe
            seasons = got['Season'] == season
            temp_df = got[seasons]
            # random sample of 1466 rows (as this is the min number of lines in a season)
            subset_df = temp_df.sample(n = 1466)
            # append to new dataframe
            new_df = new_df.append(subset_df)
        # overwrite old dataframe with new dataframe        
        got = new_df
    return got


def logRegModel(data, test_size, vectorize):
    '''
    Make train test split.
    Vectorize the data.
    Train the logistic regression model.
    
    '''
    
    # get the values in each cell; returns a list
    sentences = data['Sentence'].values
    seasons = data['Season'].values
    
    # train and test split using sklearn
    X_train, X_test, y_train, y_test = train_test_split(sentences, 
                                                        seasons, 
                                                        test_size=test_size, # define test size
                                                        random_state=42, # random state for reproducibility
                                                        stratify = seasons)
    # if vectorizer is tfidf
    if vectorize == "tfidf":
        # initialize tfidf vectorizer
        vectorizer = TfidfVectorizer()
    else:
        # initialize count vectorizer
        vectorizer = CountVectorizer()

    # First we fit the vectorization on our training data...
    X_train_feats = vectorizer.fit_transform(X_train)
    #... then we do it for our test data
    X_test_feats = vectorizer.transform(X_test)
    # We can also create a list of the feature names. 
    #feature_names = vectorizer.get_feature_names()
    
    
    print("\n[INFO] Training the model...")
    # fitting the data to the model
    classifier = LogisticRegression(random_state=42, max_iter = 1000).fit(X_train_feats, y_train)
    
    return classifier, X_test_feats, y_test, sentences, seasons


def evaluate_model(classifier, X_test, y_test):
    
    # predicting on the test data
    y_pred = classifier.predict(X_test)
    
    # define path for classification report
    classification_path = os.path.join("..", "output", "classification_report_logRegModel.csv")
    
    # making the classification report
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    # print to terminal
    print(classifier_metrics)
    
    # making the classification report as dict
    classifier_metrics_dict = metrics.classification_report(y_test, y_pred, output_dict=True)
    # transpose and make into a dataframe
    classification_metrics_df = pd.DataFrame(classifier_metrics_dict).transpose()
    # saving as csv
    classification_metrics_df.to_csv(classification_path)
    # print that the csv file has been saved
    print(f"\n[INFO] Classification metrics are saved as {classification_path}")
    
    
    # specify path
    matrix_path = os.path.join("..", "output", "classification_matrix_logRegModel.png")
    # make classification matrix
    clf.plot_cm(y_test, y_pred, normalized=True)
    # save as png
    plt.savefig(matrix_path)
    # print that the png file has been saved
    print(f"\n[INFO] Classification matrix are saved as {matrix_path}")
    


    
def cross_validation(sentences, seasons, out_name, vectorize):
    '''
    Perform cross validation with 100 splits.
    Plotting the learning curve.
    Save plot in the output folder.
    '''
    # if vectorizer is tfidf
    if vectorize == "tfidf":
        # initialize tfidf vectorizer
        vectorizer = TfidfVectorizer()
    else:
        # initialize count vectorizer
        vectorizer = CountVectorizer()
    
    
    # Vectorize full dataset
    X_vect = vectorizer.fit_transform(sentences)

    # initialise cross-validation method
    title = "Learning Curves (Logistic Regression)"
    cv = ShuffleSplit(n_splits=100, test_size=0.25, random_state=0)
    
    # define output path
    cross_val_path = os.path.join("..", "output", out_name)
    # run on data
    model = LogisticRegression(random_state=42, max_iter = 1000)
    clf.plot_learning_curve(model, title, X_vect, seasons, cv=cv, n_jobs=4)
    # save as png
    plt.savefig(cross_val_path)
    
    # print that the png file has been saved
    print(f"\n[INFO] Cross validation plots are saved as {cross_val_path}")

    
def create_dir(dirName):
    '''
    Helper function for creating directory if it doesn't exist
    '''
    # if the path does not exist
    if not os.path.exists(dirName):
        # make directory
        os.mkdir(dirName)
        print("\n[INFO] Directory " , dirName ,  " Created ")
    else:   
        # print that it already exists
        print("\n[INFO] Directory " , dirName ,  " already exists")
    
    
if __name__ == "__main__":
    main(args)

