#!/usr/bin/env python
"""
Specify file path of the csv file of Game of Thrones lines and name of the output model history plot. You can also specify size of the test data in percentage points. The default will be 0.25. Furthermore, the user can specify pooling method, optimizer and number of epochs to train over. The output will be a summary of the model architecture saved as both txt and png, a model history plot saved as png and a classification report saved as csv and printed to the terminal. These will all be saved in a folder called output in the parent directory of the working directory.

Parameters:
    filepath: str <filepath-of-csv-file>, default = '../data/Game_of_Thrones_Script.csv'
    output_filename: str <name-of-png-file>, default = 'CNN_model_history.png'
    balance: str <balance-data>, default = 'True'
    test_size: float <size-of-test-data>, default = 0.25
    optimizer: str <optimization-method>, default = 'adam'
    pooling: str <pooling-method>, default = 'MaxPooling'
    n_epochs: int <number-of-epochs>, default = 20
Usage:
    deepLearnModel.py -f <filepath-of-csv-file> -o <name-of-png-file> -b <balance-data> -t <size-of-test-data> -opt <optimization-method> -p <pooling-method> -e <number-of-epochs>
Example:
    $ python3 deepLearnModel.py -f ../data/Game_of_Thrones_Script.csv -o deepLearn_model_history.png -b True -t 0.25 -opt adam -p MaxPooling -e 20
    
## Task
- Train a convolutional neural network to classify lines from GoT characters into their respective seasons (1-8). 
- Output is in the form of two png files (model architecture and plot of model history), a txt file of the model architecture and a csv file of the classification metrics. These outputs can be found in the path "../output". Model accuracy for training and testing data will also be printed in the terminal.
"""

# libraries
# system tools
import os
import sys
sys.path.append(os.path.join(".."))

# pandas, numpy, gensim, contextlib
import pandas as pd
import numpy as np
import gensim.downloader
from contextlib import redirect_stdout

# import my classifier utility functions - see the Github repo!
import utils.classifier_utils as clf

# Machine learning stuff
from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, 
                                     Flatten, GlobalMaxPool1D, Conv1D, GlobalAveragePooling1D)
from tensorflow.keras.optimizers import SGD, Adam
#from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import L2

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
                default = "CNN_model_history.png", 
                help = "Name of output file")
# balance the data
ap.add_argument("-b", "--balance", 
                default = True, 
                help = "Balance the data? Choose between 'True' or 'False'")
# test size
ap.add_argument("-t", "--test_size", 
                default = 0.25, 
                help = "Size of test data")
# optimizer
ap.add_argument("-opt", "--optimizer", 
                default = 'adam', 
                help = "Optimizer: Choose between 'SGD' or 'adam'.")
# type of pooling
ap.add_argument("-p", "--pooling", 
                default = "MaxPooling", 
                help = "Pooling: Choose between 'MaxPooling' and 'AveragePooling'")
# number of epochs
ap.add_argument("-e", "--n_epochs", 
                default = 20, 
                help = "Number of epochs")

# parsing arguments
args = vars(ap.parse_args())




def main(args):
    # get variables from argparse
    filepath = args["filepath"]
    out_name = args["output_filename"]
    balance = args["balance"]
    test_size = args["test_size"]
    opt = args["optimizer"]
    pool = args["pooling"]
    epochs = int(args["n_epochs"])
    
    # Create out directory if it doesn't exist in the data folder
    dir_name = os.path.join("..", "output")
    create_dir(dirName = dir_name)
    
    # load and prepare data (check distribution and balance data)
    got = load_and_prepare(filepath = filepath,
                           balance = balance)
        
    # make train and test data ready
    vocab_size, maxlen, X_train_pad, X_test_pad, trainY, testY, tokenizer = train_test_preprocessing(data = got,
                                                                                                     test_size = test_size)
    
    # train the model and return the model summary and the model history
    model, history = training_model(opt = opt,
                                    pool = pool,
                                    epochs = epochs,
                                    tokenizer = tokenizer,
                                    vocab_size = vocab_size, 
                                    maxlen = maxlen,
                                    X_train = X_train_pad, 
                                    X_test = X_test_pad, 
                                    trainY = trainY, 
                                    testY = testY)
    
    # evaluate the model
    evaluate_model(model = model, 
                   history = history,
                   epochs = epochs,
                   out_name = out_name,
                   X_train = X_train_pad, 
                   X_test = X_test_pad, 
                   trainY = trainY, 
                   testY = testY)
    
    print("\nYou have now successfully trained and evaluated a deep learning model on the Game of Thrones data. Have a nice day!")
        
    
    
def load_and_prepare(filepath, balance):
    '''
    Load the data and check distribution.
    Balance the data if it is specified by the user.
    Return data.
    '''
    # load the data with pandas
    got = pd.read_csv(filepath)
    
    # calculate length of lines

    len_of_sentences = []   # create empty list
    
    for sentence in got["Sentence"]:           # for each sentence in column
        len_sentence = len(sentence)           # calculate length of sentence
        len_of_sentences.append(len_sentence)  # append to list
       
    got["len_sentence"] = len_of_sentences     # add list as column in data frame
    
    # check data distribution
    for season in got["Season"].unique():                # for each season
        seasons = got['Season'] == season                # subset for that season
        temp_df = got[seasons]
        avg_sentence = temp_df['len_sentence'].mean()    # calculate average sentence length
        max_length = temp_df['len_sentence'].max()       # calculate max sentence length
        min_length = temp_df['len_sentence'].min()       # calculate min sentence length
        
        # print information in terminal
        print(f"\n[INFO] {season} has an average sentence length of {avg_sentence}, a max of {max_length} and a min of {min_length}.")
    
    # if the user has chosen to balance data
    if balance == True:
        print("\n[INFO] Balancing the data...")
        
        column_names = ["Release Date", "Season", "Episode", "Episode Title", "Name", "Sentence"] # define column names
        new_df = pd.DataFrame(columns = column_names)                                             # create empty data frame
        
        
        for season in got["Season"].unique():      # for each unique season
            seasons = got['Season'] == season      # subset data on season 
            temp_df = got[seasons]                 # save as temporary dataframe
            subset_df = temp_df.sample(n = 1466)   # random sample of 1466 rows (as this is the min number of lines in a season)
            new_df = new_df.append(subset_df)      # append to new dataframe        
        got = new_df                               # overwrite old dataframe with new dataframe
    
    return got


def train_test_preprocessing(data, test_size):
    '''
    Making the training and testing data from the dataframe.    
    Finding max length of sentences.
    Tokenizing the data and padding to match max length.
    Transforming labels to binarized vectors.
    Returning vocabulary size, max length of sentences, padded training and test data and binarized y labels for training and testing.
    '''

    
    # get the values in each cell; returns a list
    sentences = data['Sentence'].values
    seasons = data['Season'].values
    
    # train and test split using sklearn
    X_train, X_test, y_train, y_test = train_test_split(sentences, # X 
                                                        seasons, # labels
                                                        test_size=test_size, 
                                                        random_state=42, # random state for reproducibility
                                                        stratify = seasons) # stratify to ensure same ratio of class data in train and test 
    
    # finding max length of lines to use for padding
    maxlen = int(max(data["len_sentence"]))
    
    # initialize tokenizer
    tokenizer = Tokenizer(num_words=4000) # maximum number of words to keep
    tokenizer.fit_on_texts(X_train)       # fit to training data

    # tokenized training and test data
    X_train_toks = tokenizer.texts_to_sequences(X_train)
    X_test_toks = tokenizer.texts_to_sequences(X_test)

    # overall vocabulary size
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
    
    
    # pad training data to maxlen
    X_train_pad = pad_sequences(X_train_toks,   # tokenized X train data
                                padding='post', # padding after sentence length
                                maxlen=maxlen)  # length of the longest sentence 
    # pad testing data to maxlen
    X_test_pad = pad_sequences(X_test_toks,     # tokenized X test data
                               padding='post',  
                               maxlen=maxlen)
    
    # transform labels to binarized vectors
    lb = LabelBinarizer()
    trainY = lb.fit_transform(y_train)
    testY = lb.fit_transform(y_test)
    
    return vocab_size, maxlen, X_train_pad, X_test_pad, trainY, testY, tokenizer





def training_model(opt, pool, epochs, tokenizer, vocab_size, maxlen, X_train, X_test, trainY, testY):
    '''
    Training the model. 
    Define model architecture and save as txt and png.
    Fitting the data to the model and returning the model and the history.
    '''

    l2 = L2(0.0001)        # define regularizer
    embedding_dim = 50     # define embedding size we want to work with
    
    # create embedding matrix to use as weights by loading the glove embeddings
    embedding_matrix = create_embedding_matrix('../data/glove/glove.6B.50d.txt', # filepath to GloVe word embedding
                                               tokenizer.word_index,             # indices from keras Tokenizer
                                               embedding_dim)                    # embedding size

    
    
    # if user specifies pooling method as MaxPooling
    if pool == "MaxPooling":
        # initialize Sequential model
        model = Sequential()
        
        # add Embedding layer with embedding matrix
        model.add(Embedding(input_dim=vocab_size,        # vocab size from Tokenizer()
                            output_dim=embedding_dim,    # user defined embedding size
                            weights=[embedding_matrix],  # we've added our pretrained GloVe weights
                            input_length=maxlen,         # maxlen of padded docs
                            trainable=False))            # embeddings are static - not trainable
        
        
        model.add(Conv1D(256,                            # 256 nodes
                         5,                              # kernel size = 5
                        activation='relu',               # ReLU activation
                        kernel_regularizer=l2))          # kernel regularizer
        
        # Max pooling layer
        model.add(GlobalMaxPool1D())
        
        # Add Dense layer; 
        model.add(Dense(128,                             # 128 nodes
                        activation='relu'))              # ReLU activation
        
        # Add output layer 
        model.add(Dense(8,                               # 8 predicion nodes
                        activation='softmax'))           # softmax activation

        
        # compile model
        model.compile(loss='categorical_crossentropy',   # categorical loss function (as the classification is more than binary)
                      optimizer=opt,                     # user specified optimization
                      metrics=['accuracy'])

        # print summary
        model.summary()
    
    # else if pooling method is average pooling
    elif pool == "AveragePooling":
        # initialize Sequential model
        model = Sequential()
        
        # add Embedding layer with embedding matrix
        model.add(Embedding(input_dim=vocab_size,        # vocab size from Tokenizer()
                            output_dim=embedding_dim,    # user defined embedding size
                            weights=[embedding_matrix],  # we've added our pretrained GloVe weights
                            input_length=maxlen,         # maxlen of padded docs
                            trainable=False))            # embeddings are static - not trainable
        
        
        model.add(Conv1D(256,                            # 256 nodes
                         5,                              # kernel size = 5
                        activation='relu',               # ReLU activation
                        kernel_regularizer=l2))          # kernel regularizer
        
        # Average pooling layer
        model.add(GlobalAveragePooling1D())
        
        # Add Dense layer; 
        model.add(Dense(128,                             # 128 nodes
                        activation='relu'))              # ReLU activation
        
        # Add output layer 
        model.add(Dense(8,                               # 8 predicion nodes
                        activation='softmax'))           # softmax activation

        # compile model
        model.compile(loss='categorical_crossentropy',   # categorical loss function (as the classification is more than binary)
                      optimizer=opt,                     # user specified optimization
                      metrics=['accuracy'])

        # print summary
        model.summary()
    
    # else print not a valid method
    else:
        print("Not a valid pooling method. Choose between 'MaxPooling' or 'AveragePooling'.")
    
    # name for saving model summary
    model_path = os.path.join("..", "output", "CNN_model_summary.txt")
    # Save model summary
    with open(model_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
            
    # name for saving plot
    plot_path = os.path.join("..", "output", "CNN_model_architecture.png")
    # Visualization of model
    plot_DL_model = plot_model(model,
                               to_file = plot_path,
                               show_shapes=True,
                               show_layer_names=True)
    
    print(f"\n[INFO] Model architecture is saved as txt in '{model_path}' and as png in '{plot_path}'.")
    
    
    
    # training the model and saving model history
    print(f"\n[INFO] Training the model with opt = {opt} and pooling = {pool}")
    history = model.fit(X_train, trainY,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(X_test, testY))
    
    return model, history


def evaluate_model(model, history, epochs, X_train, X_test, trainY, testY, out_name):
    '''
    Evaluate the model. 
    Print the model accuracy for the training and testing data. 
    Save the model history as png.
    '''
    # Predictions
    predY = model.predict(X_test)
    
    # Classification report
    classification = classification_report(testY.argmax(axis=1), 
                                           predY.argmax(axis=1))
            
    # Print classification report
    print(classification)
    
    # name for saving report
    report_path = os.path.join("..", "output", "classification_report_CNN_model.txt")
    
    # Save classification report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(classification_report(testY.argmax(axis=1),
                                           predY.argmax(axis=1)))
    
    print(f"\n[INFO] Classification report is saved as '{report_path}'.")
    
    # Evaluate the model
    print(f"\n[INFO] Evaluating the model...")
    loss, accuracy = model.evaluate(X_train, trainY, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, testY, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    
    # plot history
    print(f"Plotting the model history across {epochs} epochs")
    plot_history(history, epochs=epochs, out_name = out_name)

    
    
def create_embedding_matrix(filepath, word_index, embedding_dim):
    """ 
    A helper function to read in saved GloVe embeddings and create an embedding matrix
    
    filepath: path to GloVe embedding
    word_index: indices from keras Tokenizer
    embedding_dim: dimensions of keras embedding layer
    """
    vocab_size = len(word_index) + 1                          # adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))  # making embedding matrix with all zeros
    
    # add vectors for each word to matrix
    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

    
def plot_history(H, epochs, out_name):
    """
    Utility function for plotting model history using matplotlib
    
    H: model history 
    epochs: number of epochs for which the model was trained
    out_name: filename for the output png
    """
    # name for saving output
    figure_path = os.path.join("..", "output", out_name)
    # Visualize performance
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path)
    
    
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