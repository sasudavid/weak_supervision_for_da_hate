

'''
This script is the going to call all of the other scripts implemented in this project
'''

import sys
import os
import pandas as pd
import numpy as np
import ktrain

from ktrain import text
from snorkel.labeling.analysis import LFAnalysis

sys.path.insert(0, '/internship_project/hate-speech-and-offensive-language/data')

from split_data import generate_split_data
from data_processing import remove_twitter_handles
from data_processing import derive_hate_words
from data_processing import add_hate_words
from data_processing import pre_process
from labelling_functions import checking_hate_bow
from labelling_functions import check_bigrams
from labelling_functions import check_trigrams
from labelling_functions import check_emojis
from labelling_functions import check_sentiment
from labelling_functions import check_toxicity
from applying_lfs import apply_lfs
from analyzing_lfs_performance import analyze_labelling_performance
from snorkel.labeling.model import LabelModel
from sklearn.model_selection import train_test_split





#USEFUL FILE PATHS
data_path = '/internship_project/hate-speech-and-offensive-language/data'
split_data_directory = data_path + '/hate_offensive_neutral_data'
split_data_directory_no_handles = data_path + '/hate_offensive_neutral_no_handles_data'

#English original training data categorized into hate, offensive and neutral speech text files
labeled_data = data_path + '/hate_offensive_neutral_data/labeled_data.csv'
hate_data = data_path + '/hate_offensive_neutral_data/hate.txt'
offensive_data = data_path + '/hate_offensive_neutral_data/offensive.txt'
neutral_data = data_path + '/hate_offensive_neutral_data/neutral.txt'


#English preprocessed training data containing no twitter handles categorized into hate, offensive and neutral speech text files
hate_data_no_handles = data_path + '/hate_offensive_neutral_no_handles_data/hate_no_handles.txt'
offensive_data_no_handles = data_path + '/hate_offensive_neutral_no_handles_data/offensive_no_handles.txt'
neutral_data_no_handles = data_path + '/hate_offensive_neutral_no_handles_data/neutral_no_handles.txt'


#Danish original training data and test data
danish_hate_training_data  = data_path + '/labelling_functions_hate_data/offenseval-da-training-v1.tsv'
danish_hate_test_data = data_path + '/labelling_functions_hate_data/offenseval-da-test-v1.tsv'

#Cleaned danish training data
cleaned_danish_hate_training_data = data_path + '/cleaned_danish_hatespeech.txt'

#Tag file containing the tags in the hatespeech corpus
tag_file = data_path + '/danish_hatespeech_tags.txt'

#name of the supervised model to be used in the weak supervision pipeline
model_name = 'distilbert-base-uncased'






'''
The populate_pandas_dataframe is meant to create a pandas dataframe and populate it with
data samples that can be found in a given text file.

Params:
    text_file: This is a file that contains the data samples.
    dataframe_columns: This is a list containing the names of the columns of the pandas dataframe to be created.

Returns:
    pd: This is the created and populated pandas dataframe.
'''
def populate_pandas_dataframe(text_file, dataframe_columns):
    hatespeech_sentences = open(text_file,'r').readlines()
    pandas_dataframe = pd.DataFrame(hatespeech_sentences, columns=dataframe_columns)
    return pandas_dataframe



def train_model(training_data):
    #initialize the Label Model
    label_model = LabelModel()
    #fit the labelling model to the training data 
    label_model.fit(L_train=training_data)
    return label_model


'''
The generate_danish_hatespeech_testdata function produces a dataframe containing gold-labeled
danish hatespeech sentences that can be used to test the accuracy of the model developed
using the snorkel labels.

Params:
    inputfile (str): This is a string representing the name of the text file where the gold labeled 
                    data can be found.

Returns:
    danish_hatespeech_testdata_dataframe (obj): This is a pandas table containing the hatespeech sentences
                                                and their corresponding gold labels.
'''
def generate_danish_hatespeech_testdata(goldlabeled_testfile):
    #open the inputfile
    inputfile = open(goldlabeled_testfile, 'r')

    sentence_label_pairs = []
    #place the sentences in the text files in a format which can be used to create a pandas dataframe
    test_sentences = inputfile.readlines()
    for test_sentence in test_sentences:
        #strip the sentence of newline characters
        test_sentence_without_newline_chars = test_sentence.strip('\n')

        #split the test sentence
        split_test_sentence = test_sentence_without_newline_chars.split()

        #join the aspects of the split test sentence that are just the hatespeech sentence
        hatespeech_sentence = ' '.join(split_test_sentence[:len(split_test_sentence)-1])
        hatespeech_label = int(split_test_sentence[len(split_test_sentence)-1])
        sentence_label_pairs.append([hatespeech_sentence, hatespeech_label])

    #return a pandas dataframe containing the hatespeech sentences and their corresponding labels
    return pd.DataFrame(sentence_label_pairs, columns=['Sentence', 'Label'])



def generate_predictions(training_model, data):
    return training_model.predict(L=data)



def generate_unigram_hatefile():
    #hate speech data file to be used as source file for generating hate speech word tokens
    hate_speech_word_token_file =  data_path + '/labelling_functions_hate_data/offenseval-da-training-v1.tsv'
    derive_hate_words(hate_speech_word_token_file, 'hate_bag_of_words.txt')
    #add new lists of hate words to the created bag of hate bag of words
    #add_hate_words('intermediate_hate_words.txt', 'test_file.txt')
    return



def generate_english_hate_speechdata():
    # if the datasets have not already been generated then the split datasets 
    if not os.path.exists(split_data_directory):
        generate_split_data(labeled_data, hate_data, offensive_data, neutral_data)
    
    
    if not os.path.exists(split_data_directory_no_handles):
        os.mkdir(split_data_directory_no_handles)

    
        data_files_to_process = [hate_data, offensive_data, neutral_data]
        processed_data_files = [hate_data_no_handles, offensive_data_no_handles, neutral_data_no_handles]

        for filename_index in range(len(data_files_to_process)):
            remove_twitter_handles(data_files_to_process[filename_index], processed_data_files[filename_index])
    

    #get all of the sentences belonging to hate_no_handles.txt that have their handles removed and set those
    #sentences as the training data for this project since this project is about hatespeech.

    #place all of the sentences into a pandas dataframe.

    hatespeech_dataframe = populate_pandas_dataframe(hate_data_no_handles, ['hate sentences'])

    return hatespeech_dataframe


def generate_danish_hate_speechdata_gold_labels():
    #create a pandas dataframe to contain the gold labels of the hatespeech sentences in the danish_hatespeech_dataframe
    danish_hatespeech_gold_labels_dataframe = populate_pandas_dataframe(tag_file, ['gold labels'])

    return danish_hatespeech_gold_labels_dataframe



def generate_danish_hate_speechdata():
    #create pandas dataframe to contain the hatespeech sentences
    danish_hatespeech_dataframe = populate_pandas_dataframe(cleaned_danish_hate_training_data, ['hate sentences'])

    return danish_hatespeech_dataframe


'''
The generate_sentence_snorkel_label_pairs is a function that creates a text file that contains on each line
a hatespeech sentence and its corresponding predicted snorkel label.

Params:
    sentences (list(str)): This is a list of strings, where each string represents a sentence.
    labels (list(int)): This is a list of ints, where each int represents a predicted snorkel label.
    dataframe_name: This is the name of the pandas dataframe that is going to contain the sentence - snorkel pairs.

Returns:
    A pandas dataframe containing sentence and its corresponding predicted snorkel label.
'''
def generate_sentence_snorkel_label_pairs(sentences, labels, dataframe_name):
    sentence_label_pairs = []

    #iterate through the sentence list and append it and its corresponding label to the output file
    for index in range(len(sentences)):
        sentence = sentences[index][0].strip('\n')
        label = labels[index]
        sentence_label_pairs.append([sentence , label])
        
    
    return pd.DataFrame(sentence_label_pairs, columns=['Sentence', 'Snorkel Label'])


'''
The check_labelling_functions_efficacy function checks the predictive efficiency of the formulated labelling functions.

Params:
    labelled_matrix (n x m matrix): This is an n x m matrix of the predictions of the labelling functions for each sentence.
    labelling_functions (list(func)): This is a list of the labelling functions
    gold_labels (list(int)): This is a list of the gold labels of each sentence

Returns:
    None
'''
def check_labelling_functions_efficacy(labelled_matrix, labelling_functions, gold_labels):
    lf_efficacy = LFAnalysis(labelled_matrix, labelling_functions).lf_summary(gold_labels)
    print(lf_efficacy)
    return



def apply_supervised_classifier_using_testdata(train_hatespeech_dataframe, test_hatespeech_dataframe, supervised_classifier_name):

    #single out the text values and the labels into their respective lists for the training data
    training_text_values = train_hatespeech_dataframe['Sentence'].values.tolist()
    training_label_values = train_hatespeech_dataframe['Snorkel Label'].values.tolist()

    #single out the text values and the labels into their respective lists for the test data
    test_text_values = test_hatespeech_dataframe['Sentence'].values.tolist()
    test_label_values = test_hatespeech_dataframe['Label'].values.tolist()

    
    
    #split the data in the hatespeech dataframe into train and test splits
    X_train = training_text_values
    X_test = test_text_values
    y_train = training_label_values
    y_test = test_label_values

    
    #initialize the supervised model being used
    t = text.Transformer(supervised_classifier_name, class_names = ['1','0'], maxlen=500)

    #process the training data
    train = t.preprocess_train(X_train, y_train)

    #process the testing data
    validation = t.preprocess_test(X_test, y_test) 

    #activate the model and declare the hyperparameters to be used
    model = t.get_classifier()
    learner = ktrain.get_learner(model, train_data=train, val_data=validation, batch_size=4)
    learner.fit_onecycle(3e-5,3)

    #validate the model
    learner.validate(class_names=t.get_classes())



    return


'''
The apply_supervised_classifier function is going to apply a transformer for supervised text classification.

Params:
    hatespeech_dataframe: This is a pandas dataframe object containing hatespeech sentences and their corresponding 
                            snorkel labels.
    supervised_classifier_name: This is the name of the supervised classifier that is going to be used to classify the text.


Returns:
    None
'''
def apply_supervised_classifier(hatespeech_dataframe, supervised_classifier_name):

    #single out the text values and the labels into their respective lists
    training_text_values = hatespeech_dataframe['Sentence'].values.tolist()

    training_label_values = hatespeech_dataframe['Snorkel Label'].values.tolist()


    #split the data in the hatespeech dataframe into train and test splits
    X_train, X_test, y_train, y_test = train_test_split(training_text_values, training_label_values, test_size=0.30, random_state=98052)

    #initialize the supervised model being used
    t = text.Transformer(supervised_classifier_name, class_names = ['1','0'], maxlen=500)

    #process the training data
    train = t.preprocess_train(X_train, y_train)

    #process the testing data
    validation = t.preprocess_test(X_test, y_test) 

    #activate the model and declare the hyperparameters to be used
    model = t.get_classifier()
    learner = ktrain.get_learner(model, train_data=train, val_data=validation, batch_size=4)
    learner.fit_onecycle(3e-5,3)

    #validate the model
    learner.validate(class_names=t.get_classes())

    

    return




def main():
    #generate hatespeech data
    hatespeech_data = generate_danish_hate_speechdata()

    #generate testdata
    test_data = generate_danish_hatespeech_testdata('tagged_hatespeech_test_data.txt')


    
    #split the dataframe into the training data and the validation data set
    training_data = hatespeech_data[:200]
    validation_data = hatespeech_data[2000:]

    #obtain the gold labels for the validation set and the training set 
    gold_labels_dataframe = generate_danish_hate_speechdata_gold_labels()
    gold_labels_values = gold_labels_dataframe['gold labels'].values
    training_set_gold_labels = np.array([int(x.strip('\n')) for x in gold_labels_values[:2000]])
    validation_set_gold_labels = np.array([int(x.strip('\n')) for x in gold_labels_values[2000:]])


    #list of the labelling functions to be used 
    formulated_labelling_functions = [checking_hate_bow, check_bigrams, check_trigrams, check_emojis, check_sentiment, check_toxicity]

    #apply the labelling functions in the list
    L_train = apply_lfs(training_data, formulated_labelling_functions)
    L_valid = apply_lfs(validation_data, formulated_labelling_functions)


    #train the labelling model on the outputs generated by the labelling functions
    trained_model = train_model(L_train)


    #use the trained model to provide predictions for the validation set and the training set
    valid_predictions = generate_predictions(trained_model, L_valid)
    training_predictions = generate_predictions(trained_model, L_train)

    #generate sentence-snorkel label pairs
    sentence_snorkel_label_dataframe = generate_sentence_snorkel_label_pairs(training_data.values, training_predictions, 'training_sentence_label_dataframe')

    #preprocess the produced predictions by removing the hatespeech sentences that have a prediction value of -1.
    #This is to ensure that we only have data points that have a definitive prediction
    sentence_snorkel_label_dataframe = sentence_snorkel_label_dataframe[sentence_snorkel_label_dataframe['Snorkel Label'] >= 0]

  

    #apply the supervised  model
    #apply_supervised_classifier(sentence_snorkel_label_dataframe, model_name)
    apply_supervised_classifier_using_testdata(sentence_snorkel_label_dataframe, test_data, model_name)

   

    

    
   


main()

