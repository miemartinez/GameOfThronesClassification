# Text Classification with Game Of Thrones 
### Text classification using Deep Learning
**This project was developed as part of the spring 2021 elective course Cultural Data Science - Language Analytics at Aarhus University.** <br>

__Task:__ The task for this project is to do text classification on the lines from Game of Thrones and see whether the model can predict which season a line comes from. 
For this, a baseline nodel is created using a 'classical' machine learning solution such as count vectorization and logistic regression and this is used as a means of comparison.
Then we should come up with a solution which uses a deep learning model, such as the convolutional neural network and evaluate how well it performs on the data.

The data used for this project can be found on Kaggle (https://www.kaggle.com/albenft/game-of-thrones-script-all-seasons). 
The data is in the form of a csv file which contain the complete set of Game of Thrones script for all seasons alongside the label of which season and character the line is from.
The csv file can also be found in the data folder. <br>

In the data folder there is also a zipped folder with the glove embeddings. For the script to work, this folder should first be unzipped and the structure from the data folder should be as follows:
./glove/glove.6B.50d.txt <br>

The output of the topic model is provided in the output folder. This contains classification reports for the logistic regression model (with unbalanced data) and the deep learning model (with balanced data).  
Furthermore, the folder holds the model architecture for the deep learning model as both png and txt file and the model history plot (with balanced data).
For the logistic regression, the folder also contains the classification matrix (unbalanced data) and the cross validation plot (for both the balanced and unbalanced data). <br>

The scripts logRegModel.py and deepLearnModel.py are in the src and they can run without any input but several parameters can be user defined. <br>

### Method: <br>
After loading in the data, the first problem occurs with the nature of the data. The seasons are very unbalanced in number of lines and also in length of lines. 
For instance, Season 2 has more than 3900 lines whereas Season 8 only has 1466 lines. Similarly, the average line is around 60 to 65 words in most seasons but in Season 8 it is only 47 words. 
Lastly, maximum number of words differ immensely across seasons where the minimum value is 560 words in Season 8 and the maximum is almost 1700 words in Season 2. 
To accommodate these problems, I balance the data by taking a random sample of 1466 lines from each season and append these to the same data frame. 
I then train the logistic regression classifier using TF-IDF vectorization and run the model on both the balanced and unbalanced data. 
Subsequently, I train a deep learning model with a convolutional neural network. 
I choose a relatively simple CNN model architecture with one convolutional layer and two fully connected layers to reduce chances of overfitting. 
I also allow the user to specify the test size, how to pool the data (average and max pooling), which optimizer to use (Adam and SGD) and how many epochs to run. 
First, I train the convolutional neural network with max pooling and Adam optimizer for 20 epochs. This model is also trained on both the balanced and unbalanced data. 
To further improve the model, I implement a pretrained GloVe word embedding layer (which incidentally also makes the model run faster as it contains many non-trainable parameters). 
I also run the model with the SGD optimizer to see if it can increase the model performance.



__Dependencies:__ <br>
To ensure dependencies are in accordance with the ones used for the script, you can create the virtual environment ‘GoT_venv"’ from the command line by executing the bash script ‘create_GoT_venv.sh’. 

```
    $ bash ./create_GoT_venv.sh
```
This will install an interactive command-line terminal for Python and Jupyter as well as all packages specified in the ‘requirements.txt’ in a virtual environment. 
After creating the environment, it will have to be activated before running the classification scripts.
```    
    $ source GoT_venv/bin/activate
```
After running these two lines of code, the user can commence running the scripts. <br>

### How to run logRegModel.py <br>

The script logRegModel.py can run from command line without additional input. 
However, the user can specify the file path of the csv file of the Game of Thrones lines and name of the output model history. 
Similarly, the user can also specify whether the data should be balanced or unbalanced, the test size and the vectorization method. 
The default is balanced data with a test size of 0.25 and TF-IDF vectorization. 
The outputs will be a model summary, a classification report, and cross validation plots. <br>

__Parameters:__ <br>
```
    filepath: str <filepath-of-csv-file>, default = '../data/Game_of_Thrones_Script.csv'
    output_filename: str <name-of-png-file>, default = 'logReg_model_history.png'
    balance: str <balance-data>, default = 'True'
    test_size: float <size-of-test-data>, default = 0.25
    vectorizer: str <vectorization-method>, default = 'tfidf'
```
    
__Usage:__ <br>
```
    logRegModel.py -f <filepath-of-csv-file> -o <name-of-png-file> -b <balance-data> -t <size-of-test-data> -v <vectorizer>
```
    
__Example:__ <br>
```
    $ cd src
    $ python3 logRegModel.py -f ../data/Game_of_Thrones_Script.csv -o LogReg_cross_validation.png -b True -t 0.25, -v tfidf

```


### How to run deepLearnModel.py <br>

The script deepLearnModel.py can also run from command line without additional input. 
However, the user can specify the file path of the csv file of the Game of Thrones lines and name of the output model history. 
Similarly, the user can also specify whether the data should be balanced or unbalanced, the test size, the optimization method, the pooling method and number of epochs for running the model. 
The default is balanced data with a test size of 0.25, Adam optimization, max pooling and 20 epochs.  
The outputs will be a model summary, a classification report and a model history plot. <br>

__Parameters:__ <br>
```
    filepath: str <filepath-of-csv-file>, default = '../data/Game_of_Thrones_Script.csv'
    output_filename: str <name-of-png-file>, default = 'CNN_model_history.png'
    balance: str <balance-data>, default = 'True'
    test_size: float <size-of-test-data>, default = 0.25
    optimizer: str <optimization-method>, default = 'adam'
    pooling: str <pooling-method>, default = 'MaxPooling'
    n_epochs: int <number-of-epochs>, default = 20
```
    
__Usage:__ <br>
```
    deepLearnModel.py -f <filepath-of-csv-file> -o <name-of-png-file> -b <balance-data> -t <size-of-test-data> -opt <optimization-method> -p <pooling-method> -e <number-of-epochs>
```
    
__Example:__ <br>
```
    $ cd src
    $ python3 deepLearnModel.py -f ../data/Game_of_Thrones_Script.csv -o deepLearn_model_history.png -b True -t 0.25 -opt adam -p MaxPooling -e 20

```
The code has been developed in Jupyter Notebook and tested in the terminal on Jupyter Hub on worker02. I therefore recommend cloning the Github repository to worker02 and running the scripts from there. 

### Results:
Running the logistic regression classifier with an unbalanced dataset and TF-IDF vectorization, a weighted average accuracy score of 26% was obtained. 
However, running the data with a balanced dataset yielded an accuracy score of 23%. So, using a balanced dataset surprisingly reduces the model performance in this case.
For the convolutional neural network, I ran the model with max pooling and Adam optimizer over 20 epochs. 
For the unbalanced data, the model got an accuracy of 93% on the testing data but only 24% on the testing data. 
Running the model again with balanced data made the model overfit even more on the training data with an accuracy of 95% on training data and 22% on the testing data. 
I then ran the model with stochastic gradient descent (SGD) as the optimizer and here the model is less overfitted. 
However, the weighted average accuracy was at 17% on the balanced data and 14% on the unbalanced data.
In conclusion, the best model was the simple logistic regression model with TF-IDF vectorization and unbalanced data. 
This was not the expected outcome. However, the convolutional neural network seems to be overfitting a lot on the training data with the best performing model that includes the Adam optimization. 
So, another approach that might be worth exploring would be to remove the convolutional layer from this model and maybe add a dropout layer.

The plot below shows the model history for the CNN run on the balanced data for 20 epochs with the SGD optimizer:
![alt text](https://github.com/miemartinez/GameOfThronesClassification/blob/main/output/CNN_model_history.png?raw=true)
Here it is evident that the model is starting to overfit on the data after around 13 epochs.
