'''
The general purpose of this script is to house functions that would preprocess all of the 
data files that would be used for this project
'''


import string


'''
The remove_twitter_handles function removes the twitter usernames from the tweets in the inputfile

Params:
    original_tweet_file: This is a file object containing the unprocessed tweets
    tweet_file_without_handles: This is a file object containing the processed tweets

Returns:
    None
'''
def remove_twitter_handles(original_tweet_file, tweet_file_without_handles):
    #open the files
    unprocessed_tweet_file = open(original_tweet_file, 'r')
    processed_tweet_file = open(tweet_file_without_handles, 'a')

    #iterate through the unprocessed tweet file and for each tweet remove all the user
    #handles by removeing all strings with the '@' sign
    for sentence in unprocessed_tweet_file:
        split_sentence = sentence.split()
        for word in split_sentence:
            if '@' in word:
                split_sentence.remove(word)

        joined_sentence = ' '.join(split_sentence)
        processed_tweet_file.write(joined_sentence+'\n')
    
    unprocessed_tweet_file.close()
    processed_tweet_file.close()

    return
'''
The pre_process function is a helper function that preprocesses
the sentences in the oecd dataset by removing punctuations, OFF tags,
NOT tags and numbers.

Params:
    sentence (str): This is the sentence that is supposed to be preprocessed

Return:
    preprocessed_sentence (str): This is the preprocessed sentence

'''
def pre_process(sentence):
    sentence = sentence.strip('\n')

    #if there are capital letters in the string, reduce them into small leters
    lowered_sentence = sentence.lower()

    #split the sentence into a list
    split_lowered_sentence = lowered_sentence.split()

    #slice the list to select from only the second to the last but one index
    sliced_split_lowered_sentence = split_lowered_sentence[1:len(split_lowered_sentence)-1]

    #join the sliced list back together again
    joined_sliced_lowered_sentence = ' '.join(sliced_split_lowered_sentence)

    #parse the joined sliced list to get rid of punctuations
    parsed_string = joined_sliced_lowered_sentence.translate(str.maketrans('','',string.punctuation))




    return parsed_string



'''
The remove_irrelevant_words function is a helper function that takes in a hate speech comment and 
removes the irrelevant words from this sentence.

Params:
    sentence (str): This is the hatespeech sentence in which the irrelevant words are to be removed.

Return:
    processed_sentence (str): This is the hatespeech sentence without certain words that have no
                                bearing to whether the sentence is hateful or not.
'''
def remove_irrelevant_words(sentence):

    danish_pronouns = ['Jeg','Du', 'Han', 'hun', 'den', 'det', 'Vi', 'I', 'De', 
                        'mig', 'dig','ham', 'hende', 'den', 'det', 'os', 'jer', 'dem',
                        'min','mit', 'mine', 'din', 'dit','dine', 'hans', 'hendes', 'dens', 'dets',
                        'vores', 'jeres', 'deres', 'sin', 'sit', 'sine', 'mig', 'dig', 'sig',
                        'os', 'jer', 'som', 'der', 'hvem', 'hvilke', 'hvilken', 'hvilket', 'hvad',
                        'hvis']
    danish_indefinite_articles = ['en','et']
    danish_definite_articles = ['den','det','de']

    danish_numerals = ['nul', 'en','et', 'to', 'tre', 'fire', 'fem', 'seks', 'syv', 'otte', 'ni', 'ti',
                        'elleve', 'tolv', 'tretten','fjorten', 'femten', 'seksten', 'sytten', 'atten', 'nitten',
                        'tyve', 'enogtyve', 'toogtyve', 'tredive', 'fyrre', 'halvtreds', 'tres', 'halvfjerds', 'firs',
                        'halvfems', 'hundred', 'to hundred', 'tusind', 'to tusind', 'en million', 'to millioner',
                        'en milliard', 'to millianrder']
    
    #preprocess the sentence
    preprocessed_sentence = pre_process(sentence)


    #split the sentence
    split_sentence = preprocessed_sentence.split()

    #iterate through the words in the split sentence in remove the irrelevant words within it
    for word in split_sentence:
        if word in danish_pronouns or word in danish_definite_articles or word in danish_indefinite_articles or word in danish_numerals:
            split_sentence.remove(word)
        else:
            continue
    
    reconstructed_sentence = ' '.join(split_sentence)

    return reconstructed_sentence

'''
The derive_oecd_hate_comments function is a helper function that takes the hate comments in the oecd dataset and 
removes the pronouns, numerals and articles from the these comments so that what remains 
is the actual hateful content in the sentences.

Params:
    oecd_filename: This is the name of the oecd file that contains the hate comments

Return:
    None
'''
def derive_hate_words(hate_filename, hate_bag_of_words):
    #open the file
    datafile = open(hate_filename,'r')
    hatefile = open(hate_bag_of_words, 'a')

    #read from the file one line at a time and preprocess it
    for line in datafile:

        #preprocess the sentence line
        clean_sentence = remove_irrelevant_words(line)

        #break the sentence into a list of words
        split_sentence = clean_sentence.split()

        #for each word in the split list, if the word is not in 
        #either the pronoun,  article or numeral lists, then write
        #the word into the hate file
        for word in split_sentence:
            hatefile.write(word+'\n')
    
    datafile.close()
    hatefile.close()

    return


 

'''
The add_hate_words function is a helper function that adds other identified hate words to the hate bag of words
file to aid in the classification of a comment as hate speech or not.

Params:
    hatefile (str): This is the name of the file that contains the hate words to be added to the hate bag of words
    hate_bow_file (str): This is the filename of the hate bag of words

Returns:
    None
'''
def add_hate_words(hatefile, hate_bow_file):
    inputfile = open(hatefile,'r')
    outputfile = open(hate_bow_file, 'a')
    file_lines = inputfile.readlines()
    if len(file_lines) > 1:
        for word in file_lines:
            word_without_quotes = word.strip('"/')
            word_without_trailing_spaces = word_without_quotes.strip(' ')
            outputfile.write(word_without_trailing_spaces + '\n')
    
    elif len(file_lines) == 1:
        split_line = file_lines[0].split(',')
        for word in split_line:
            word_without_quotes = word.strip('”/')
            word_without_trailing_spaces = word_without_quotes.strip(' ')
            outputfile.write(word_without_trailing_spaces + '\n')
    
    else:
        raise Exception('There are no words in the provided file!')
    
    inputfile.close()
    outputfile.close()
    return




'''
The generate_bigram_hate_words takes each of the sentences in a given text file, splits it into 
bigrams and places each bigram on a newline in another specified text file.

Params:
    source_file : This is a text file that contains the hatespeech sentences to be split into bigrams
    bigram_file: This is a text file that contains bigrams of hatespeech

Returns:
    None
'''
def generate_bigram_hate_words(source_file, bigram_file):

    #open the text files
    source = open(source_file, 'r').readlines()
    target = open(bigram_file, 'a')

    #for each sentence in the source file, split it into bigrams and
    #write each bigram into the bigram_file
    for sentence in source:

        #remove irrelevant words from the sentence
        processed_sentence = remove_irrelevant_words(sentence)


        #split the sentence into word tokens
        split_sentence = processed_sentence.split()

        #split the sentence into bigrams
        bigrams = []
        bigram_index = 0
        for i in range(0,len(split_sentence),2):
            if i+1 != len(split_sentence):
                bigram_1 = split_sentence[bigram_index]
                bigram_2 = split_sentence[bigram_index + 1]

                bigrams.append(bigram_1 + ' ' + bigram_2)
                bigram_index = i
        
        #join all of the bigrams left that have not yet been iterated on
        left_bigrams = ' '.join(split_sentence[bigram_index:])

        #place the left bigrams into the bigrams list
        bigrams.append(left_bigrams)

        for bigram in bigrams:
            target.write(bigram+'\n')


    #close the text file
    target.close()

    return



'''
The generate_trigram_hate-words takes each of the sentences in a given text file, splits it into 
trigrams and places each trigram on a newline in another specified text file. 

Params:
    source_file : This is a text file that contains the hatespeech sentences to be split into trigrams
    trigram_file: This is a text file that contains trigrams of hatespeech

Returns:
    None
'''
def generate_trigram_hate_words(source_file, trigram_file):
    #open the text files
    source = open(source_file, 'r').readlines()
    target = open(trigram_file, 'a')

    #for each sentence in the source file, split it into trigrams and 
    #write each trigram into the trigram_file

    for sentence in source:
        #remove irrelevant words from the sentence
        processed_sentence = remove_irrelevant_words(sentence)

        
        #split the sentence into word tokens
        split_sentence = processed_sentence.split() 

        #split the sentence into bigrams
        trigrams = []
        trigram_index = 0
        for i in range(0,len(split_sentence),3):
            if not(i+2 >= len(split_sentence)):
                trigram_1 = split_sentence[trigram_index]
                trigram_2 = split_sentence[trigram_index + 1]
                trigram_3 = split_sentence[trigram_index + 2]

                trigrams.append(trigram_1 + ' ' + trigram_2 + ' ' + trigram_3)
                trigram_index = i
        

        #join all of the bigrams left that have not yet been iterated on
        left_trigrams = ' '.join(split_sentence[trigram_index:])

        #place the left bigrams into the bigrams list
        trigrams.append(left_trigrams)

        for trigram in trigrams:
            target.write(trigram+'\n')
    
    #close the text file
    target.close()

    return

def clean_dataset():
    source_file_path = '/internship_project/hate-speech-and-offensive-language/data/labelling_functions_hate_data/offenseval-da-training-v1.tsv'
    target_filename = "cleaned_danish_hatespeech.txt"

    source = open(source_file_path, 'r')
    target = open(target_filename, 'a')

    for line in source:
        preprocessed_line = pre_process(line)
        target.write(preprocessed_line+'\n')
    
    source.close()
    target.close()

    return

def clean_dataset_with_tags():
    source_filepath_1 = '/internship_project/hate-speech-and-offensive-language/data/labelling_functions_hate_data/offenseval-da-training-v1.tsv'
    source_filepath_2 = '/internship_project/hate-speech-and-offensive-language/data/cleaned_danish_hatespeech.txt'
    target_file = 'cleaned_danish_hatespeech_tagged.txt'
    tag_file = 'danish_hatespeech_tags.txt'

    source_file_list_1 = open(source_filepath_1, 'r').readlines()
    source_file_list_2 = open(source_filepath_2, 'r').readlines()
    target = open(target_file, 'a')
    tag_file = open(tag_file, 'a')

    for index in range(len(source_file_list_1)):
        file_line_1 = source_file_list_1[index].strip('\n')
        file_line_2 = source_file_list_2[index].strip('\n')

        split_file_line_1 = file_line_1.split()

        if split_file_line_1[len(split_file_line_1)-1] == 'NOT':
            target.write(file_line_2+' '+'0'+'\n')
            tag_file.write('0'+'\n')
        
        if split_file_line_1[len(split_file_line_1)-1] == 'OFF':
            target.write(file_line_2+' '+'1'+'\n')
            tag_file.write('1'+'\n')
    

    target.close()
    tag_file.close()
            

    return

'''
The aggregate_testing_data function is a helper function that creates a labelled testing dataset using 
the hatespeech sentences in the 'offenseval-da-test-v1.tsv' dataset. This dataset is to be used to test 
the efficiency of the model trained using the data labelled with snorkel labels provided by the labelling 
functions.

Params:
    dataset_path (str): This is going to be the path to the testing dataset file that is going to be used.

Return:
    None
'''
def aggregate_testing_data(dataset_path, outputfile):
    #open the file in the provided path
    inputfile = open(dataset_path, 'r')
    outputfile = open(outputfile, 'a')

    #get all of the lines in the file
    file_lines = inputfile.readlines()[1:]

    for file_line in file_lines:
        #strip the newline character at the end of the line
        line_without_newline_char = file_line.strip('\n')

        #split the line into a list
        split_line = line_without_newline_char.split()

        #check to see if the last string in the split is 'OFF' or 'NOT'
        if split_line[len(split_line)-1] == 'OFF':
            new_line = ' '.join(split_line[1:len(split_line)-1]) + ' ' + '1'
            outputfile.write(new_line + '\n')
        else:
            new_line = ' '.join(split_line[1:len(split_line)-1]) + ' ' + '0'
            outputfile.write(new_line + '\n')
    
    #close the text files
    inputfile.close()
    outputfile.close()

    return







