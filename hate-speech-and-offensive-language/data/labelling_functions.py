import emoji
from snorkel import labeling
from data_processing import *
from snorkel.labeling import labeling_function
from sentida import Sentida
from googleapiclient import discovery
from googletrans import Translator
import json




#These are the tags for the sentences that are provided to the labelling functions
HATE = 1
NOT_HATE = 0
ABSTAIN = -1


#These are the paths to the textfiles containing the necessary list of words to be used
unigram_hatefile = '/internship_project/hate_bag_of_words.txt'
bigram_hatefile = '/internship_project/bigram_hate_words.txt'
trigram_hatefile = '/internship_project/trigram_hate_words.txt'
toxic_emojies = '/internship_project/toxic_emojies.txt'





'''
The checking_hate_bow is a labelling function that checks to see whether the words that constitute a given sentence
can be found in a constructed hate words bag.

Params:
    record (str): This is the sentence that is to be judged as to whether it is hateful or not
    hatefile (str): This is the filename that contains the hatewords that a given sentence is to be judged against

Return:
    result (int): 1 if the sentence given as a parameter has been judged to be a sample of hate speech, 0 if the sentence
                has been judged to not be hatespeech and -1 if the function is not sure


'''
@labeling_function()
def checking_hate_bow(record, hatefile = unigram_hatefile):
    #read the words from the hatefile and place them in a list
    hate_bag_of_words = []

    hate_words = open(hatefile)

    for word in hate_words:
        hate_bag_of_words.append(word)
    
    
    #call the preprocess function from data_processing.py to preprocess the input sentence
    preprocessed_sentence = pre_process(record['hate sentences'])

    #split the preprocessed sentence into words and check to see if more than one of its words are in the hate bag of words
    split_preprocessed_sentence = preprocessed_sentence.split()
    hate_count = 0

    for word in split_preprocessed_sentence:
        if word in hate_bag_of_words:
            hate_count += 1
            if hate_count >= 2:
                return HATE

    if hate_count == 1:
        return ABSTAIN
    
    else:
        return NOT_HATE
    


'''
The check_bigrams function checks to see whether a sentence is hatespeech or not
by removing the non-essential words in the sentence, splitting the sentence into
bigrams and running it through the bigrams hatespeech textfile

Params:
    record (str): This is the sentence that is to be judged as to whether it is hateful or not
    bigram_hatefile (str): This is the filename that contains the bigram hatewords that a given sentence 
                            is to be judged against

Returns:
    result (int): 1 if the sentence given as a parameter has been judged to be a sample of hate speech, 0 if the sentence
            has been judged to not be hatespeech and -1 if the function is not sure
'''
@labeling_function()
def check_bigrams(record, hatefile = bigram_hatefile):
    #read the bigrams from the textfile and place them into a list
    hatefile = open(hatefile, 'r')
    bigram_hatefile = hatefile.readlines()
    
    #strip new line characters from the bigrams
    bigram_hatefile = [x.strip('\n') for x in bigram_hatefile]

    #get the sentence from the provided record and preprocess it
    split_preprocessed_sentence = pre_process(record['hate sentences']).split()

    #split the sentence into bigrams
    bigrams = []
    bigram_index = 0
    for i in range(0,len(split_preprocessed_sentence),2):
        if i+1 != len(split_preprocessed_sentence):
            bigram_1 = split_preprocessed_sentence[bigram_index]
            bigram_2 = split_preprocessed_sentence[bigram_index + 1]

            bigrams.append(bigram_1 + ' ' + bigram_2)
            bigram_index = i
    
    #join all of the bigrams left that have not yet been iterated on
    left_bigrams = ' '.join(split_preprocessed_sentence[bigram_index:])

    #place the left bigrams into the bigrams list
    bigrams.append(left_bigrams)

 
    
    
    #check to see if a bigram is present in the list of hateful_bigrams
    hate_bigram_count = 0
    for bigram_pair in bigrams:
        if bigram_pair in bigram_hatefile:
            hate_bigram_count += 1
         
    if hate_bigram_count >= 2:
        return HATE
    
    elif hate_bigram_count == 1:
        return ABSTAIN
    
    else:
        return NOT_HATE


'''
The check_trigrams function checks to see whether a sentence is hatespeech or not
by removing the non-essential words in the sentence, splitting the sentence into
trigrams and running it through the bigrams hatespeech textfile

Params:
    record (str): This is the sentence that is to be judged as to whether it is hateful or not
    trigram_hatefile (str): This is the filename that contains the trigram hatewords that a given sentence 
                            is to be judged against

Returns:
    result (int): 1 if the sentence given as a parameter has been judged to be a sample of hate speech, 0 if the sentence
            has been judged to not be hatespeech and -1 if the function is not sure

'''
@labeling_function()
def check_trigrams(record, hatefile = trigram_hatefile):
    #read the bigrams from the textfile and place them into a list
    hatefile = open(hatefile, 'r')
    trigram_hatefile = hatefile.readlines()
    
    #strip new line characters from the bigrams
    trigram_hatefile = [x.strip('\n') for x in trigram_hatefile]

    #get the sentence from the provided record and preprocess it
    split_preprocessed_sentence = pre_process(record['hate sentences']).split()

    #split the sentence into bigrams
    trigrams = []
    trigram_index = 0
    for i in range(0,len(split_preprocessed_sentence),3):
        if not(i+2 >= len(split_preprocessed_sentence)):
            trigram_1 = split_preprocessed_sentence[trigram_index]
            trigram_2 = split_preprocessed_sentence[trigram_index + 1]
            trigram_3 = split_preprocessed_sentence[trigram_index + 2]

            trigrams.append(trigram_1 + ' ' + trigram_2 + ' ' + trigram_3)
            trigram_index = i
    

    #join all of the bigrams left that have not yet been iterated on
    left_trigrams = ' '.join(split_preprocessed_sentence[trigram_index:])

    #place the left bigrams into the bigrams list
    trigrams.append(left_trigrams)


    #check to see if a bigram is present in the list of hateful_bigrams
    hate_trigram_count = 0
    for trigram_pair in trigrams:
        if trigram_pair in trigram_hatefile:
            hate_trigram_count += 1
         
    if hate_trigram_count >= 2:
        return HATE
    
    elif hate_trigram_count == 1:
        return ABSTAIN
    
    else:
        return NOT_HATE



'''
The check_emojis function extracts all of the emojis in a sentence and then runs each extracted
emoji through a list of toxic emojis to ascertain if the comment is a hateful comment or not.

Params:
    record (str): This is the sentence that is to be judged as to whether it is hatespeech or not.
    toxic_emojies (str): This is the filename containing all of the toxic emojis

Returns:
    result (int): 1 if the sentence is judged to be hatespeech since it contains an instance of a toxic
                    emoji, 0 if the sentence is judged not be hatespeech since it does not contain an 
                    instance of a toxic emoji or -1 if the function is not sure. 
'''
@labeling_function()
def check_emojis(record, hatefile=toxic_emojies):
    try:
        #split the sentence into tokens
        split_sentence = record['hate sentences'].split()
    
        #extract the emojies in the sentence into a list
        sentence_emoji_list = [c for c in split_sentence if c in emoji.UNICODE_EMOJI['en']]
        processed_sentence_emojies = [emoji.encode('unicode-escape').decode('ASCII') for emoji in sentence_emoji_list]

        #read the emojis from the toxic_emojies.txt
        toxic_emojies_file = open(hatefile, 'r').readlines()
        toxic_emojies_file_without_newline = [emoji.strip('\n') for emoji in toxic_emojies_file]
        processed_toxic_emojies = [emoji.encode('unicode-escape').decode('ASCII') for emoji in toxic_emojies_file_without_newline]

        #find out if the emojies in the input sentence can be found in the list of toxic emojies
        #from the toxic emojies text file.
        bad_emoji_count = 0
        for emoji_value in processed_sentence_emojies:
            if emoji_value in processed_toxic_emojies:
                bad_emoji_count += 1
        
        if bad_emoji_count >= 1:
            return HATE
        
        else:
            return NOT_HATE
    except:
        return ABSTAIN


'''
The check_sentiment function checks the overall sentiment of the sentence and then declares a judgement on it
as to whether it can be classified as hatespeech or not.

Params:
    record (str): This is the sentence that is to be judged as to whether it is hatespeech or not.

Returns:
    result (int): 1 if the sentence is judged to be hatespeech since it carries a negative sentiment, 
                0 if the sentence is judged not be hatespeech since it does not carry a negative sentiment or 
                -1 if the function is not sure. 
'''
@labeling_function()
def check_sentiment(record):
    try:
        SV = Sentida()
        sentiment_result = SV.sentida(text=record['hate sentences'], output='mean', normal=False)
        if sentiment_result == 0:
            return ABSTAIN
        elif sentiment_result > 0:
            return NOT_HATE
        else:
            return HATE
    except:
        return ABSTAIN


'''
The check_toxicity function checks the overall level of toxicity of the given sentence using the Google
Perspective API. 

Params:
    record (str): This is the sentence that is judged as to whether it is hatespeech or not.

Return:
    result (int): 1 if the sentence is judged to be hatespeech since it carries a negative sentiment, 
                0 if the sentence is judged not be hatespeech since it does not carry a negative sentiment or 
                -1 if the function is not sure.
'''
@labeling_function()
def check_toxicity(record):
    try:
        #translate the danish sentence into english
        translator = Translator()
        translated_text_object = translator.translate(record['hate sentences'], dest='en')
        translated_text = translated_text_object.text

        #using the perspective api
        API_KEY = 'AIzaSyAooukwfBHXpY_dfvWgkXiNb_DjNkA65ws'
        client = discovery.build("commentanalyzer","v1alpha1", developerKey=API_KEY, discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1", static_discovery=False)
        analyze_request = {'comment': { 'text': translated_text}, 'requestedAttributes': {'TOXICITY': {}}}
        response = client.comments().analyze(body=analyze_request).execute()
        score = json.dumps(response, indent=2)
        formatted_score = json.loads(score)['attributeScores']['TOXICITY']['spanScores'][0]['score']['value']

        if formatted_score > 0.5:
            return HATE
        elif formatted_score == 0.5:
            return ABSTAIN
        else:
            return NOT_HATE
    except:
        return ABSTAIN








