import pandas as pd
import numpy as np
import contractions
import spacy
import string
import re
import os
import xml.etree.ElementTree as ET
import nltk
import emoji


def get_pan15_twitter_data(filename):
    ''' Fetches PAN15 data from folder, both textual posts and labels, and converts it into a Pandas Dataframe
    
        Args:
            filename(string): Name of the folder containing the PAN15 data
    
        Returns:
            Dataframe containing PAN15 combined data
    '''
    # Get path for PAN15 dataset
    current_dir = os.getcwd()
    pan15_data = []
    path = os.path.join(current_dir, 'data', 'datasets', 'pan15', filename)

    # Get all filenames for HTML files, each corresponding to one user
    user_files = os.listdir(path)

    # Read file with the labels (personality scores)
    labels = pd.read_csv(
            'data/datasets/pan15/' + filename + '/truth.txt',
            sep = ':::',
            header = None,
            names = ['USER', 'GENDER', 'AGE', 'EXT', 'NEU', 'AGR', 'CON', 'OPN'],
            engine='python'
            )    
        
    # Dataset contains emotional stability score which must be converted to neuroticism score by inverting
    labels['NEU'] = -labels['NEU']


    for file in user_files:
        # Check if extension is XML to exclude truth.txt file
        if file.endswith('.xml'):
            # Get user ID from HTML file
            tree = ET.parse(os.path.join(path, file))
            root = tree.getroot()
            user_id = root.attrib['id']

            # Extract personality scores
            ext = labels.loc[labels['USER'] == user_id, 'EXT'].values[0]
            neu = labels.loc[labels['USER'] == user_id, 'NEU'].values[0]
            agr = labels.loc[labels['USER'] == user_id, 'AGR'].values[0]
            con = labels.loc[labels['USER'] == user_id, 'CON'].values[0]
            opn = labels.loc[labels['USER'] == user_id, 'OPN'].values[0]

            # For each post, append dictionary with the user ID, text and personality scores
            user_posts = root.findall('document')
            for post in user_posts:
                post = post.text
                            
                pan15_data.append({'USER':user_id , 'POST':post , 'EXT':ext, 'NEU':neu, 'AGR':agr, 'CON':con, 'OPN':opn})
        
    return pd.DataFrame(pan15_data)

def replace_mentions(text):
    ''' Replaces Twitter mentions (@username) with custom token "[USER]"
    
        Args:
            text(string): text to convert
    
        Returns:
            text(string): converted text
    '''
    text = re.sub(r'@\w+', '[USER]', text)
    
    return text

def replace_hashtags(text):
    ''' Replaces Twitter hashtags (#example) with custom token "[HASHTAG]"
    
        Args:
            text(string): text to convert
    
        Returns:
            text(string): converted text
    '''
    text = re.sub(r'#\w+', '[HASHTAG]', text)
    
    return text

def remove_emojis(text):
    ''' Removes emojis from text
    
        Args:
            text(string): text to convert
    
        Returns:
            text(string): converted text
    '''
    text = emoji.replace_emoji(text, '')
    
    return text

def remove_special_characters(text):
    ''' Replaces "\t" and "\n" characters with spaces
    
        Args:
            text(string): text to convert
    
        Returns:
            text(string): converted text
    '''
    text = re.sub('[\n, \t]', ' ', text)
    
    return text


def remove_punctuations(text):
    '''Replace punctuation characters with spaces
    
        Args:
            text(string): text to convert
    
        Returns:
            text(string): converted text
    '''
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ')
    return text


def remove_extra_spaces(text):
    ''' Removes extra spaces from text, converting multiple spaces to a single one
    
        Args:
            text(string): text to convert
    
        Returns:
            text(string): converted text
    '''
    text = " ".join(text.split())
    
    return text

def remove_urls(text):
    ''' Removes URL's from text
    
        Args:
            text(string): text to convert
    
        Returns:
            text(string): converted text
    '''
    text = re.sub('http[s]?://\S+', '', text)
    
    return text

def remove_digits(text):
    ''' Removes digits from text
    
        Args:
            text(string): text to convert
    
        Returns:
            text(string): converted text
    '''
    text = re.sub('[0-9]+', '', text)
    
    return text

def standardize_text(text, lowercase, punctuation_rem, url_rem, digit_rem, emoji_rem, special_char_rem, mention_repl, hashtag_repl, expand_contractions):
    ''' Perform several preprocessing techniques in order to standardize text
    
        Args:
            text(string): text to convert
    
        Returns:
            text(string): converted text
    '''
    if lowercase:
        # Lowercase text
        text = str.lower(text)
    
    if expand_contractions:
        text = contractions.fix(text)
                
    if url_rem:
        text = remove_urls(text)
                        
    if punctuation_rem:
        text = remove_punctuations(text)
                
    if digit_rem:
        text = remove_digits(text)
                
    if emoji_rem:
        text = remove_emojis(text)
                
    if special_char_rem:
        text = remove_special_characters(text)
                
    if mention_repl:
        text = replace_mentions(text)
        
    if hashtag_repl:
        text = replace_hashtags(text)
            
    text = remove_extra_spaces(text)
    
    return text

def remove_undersampled_users(data, min_posts, min_tokens):
    ''' Remove users with few posts
    
        Args:
            data(Dataframe): Object containing all user data
            min_posts(int): minimum of number of posts for user to be included
            min_tokens(int): minimum of number of total tokens for user to be included
            
        Returns:
            data(Dataframe): Data after user removal
    '''
    # Get all unique user id's
    user_ids = data['USER'].unique()
    
    for user_id in user_ids:
        # Get user's posts
        user_posts = data.loc[data['USER'] == user_id, 'POST']
        n_tokens = 0
        n_posts = 0
        
        for post in user_posts:
            # Count number of tokens in the post
            words = nltk.word_tokenize(post)
            # Add it to the total number of tokens
            n_tokens += len(words)
            # Increment number of posts
            n_posts += 1
    
        # Remove users with less than the number of tokens or number of posts limits
        if n_posts < min_posts or n_tokens < min_tokens:
            data = data[data['USER'] != user_id]  
    
    return data

def tokenize_and_lemmatize(text, nlp):
    # Tokenize using spacy tokenizer
    sentence = nlp(text)
    
    # Get the word lemmas and create list
    token_list = [token.lemma_ for token in sentence]    
    
    return token_list

def detokenize(tokens):
    '''Convert from tokenized text into raw text
    
        Args:
            text(string): text to convert
    
        Returns:
            text(string): converted text
    '''
    text = tokens[0]
    
    for i in range(1, len(tokens)):
        text = text + ' ' + tokens[i]
    
    return text

def scale_personality_score(score, scale_min, scale_max):
    ''' Converts score to desired  scale [scale_min, scale_max]
    
        Args:
            score: Score to convert
            scale_min: Lower boundary for desired scale
            scale_max: Upper boundary for desired scale
    
        Returns:
            new score: Converted score
    '''
    new_score = ((score - scale_min)/(scale_max - scale_min))
    
    return new_score

def preprocessing(lowercase, punctuation_rem, url_rem, digit_rem, emoji_rem, special_char_rem, mention_repl, hashtag_repl, expand_contractions, lemmatize):
    ''' Perform preprocessing on text according to specified arguments.
    
        Args:
            lowercase(int): 1 to lowercase text, 0 otherwise
            punctuation_rem(int): 1 to remove punctuation, 0 otherwise
            url_rem(int): 1 to remove URL's, 0 otherwise
            digit_rem(int): 1 to remove digits, 0 otherwise
            emoji_rem(int): 1 to remove emojis, 0 otherwise
            hashtag_repl(int): 1 to replace mentions, 0 otherwise
            expand_contractions(int): 1 to expand contractions, 0 otherwise
            lemmatize(int):  1 to lemmatize, 0 otherwise
    
    '''
    
    # Fetch PAN15 data and convert into CSV files
    pan15_train_data = get_pan15_twitter_data(filename='pan15-author-profiling-training-dataset-english-2015-04-23')
    pan15_test_data = get_pan15_twitter_data(filename='pan15-author-profiling-test-dataset2-english-2015-04-23')
    pan15_train_data.to_csv(path_or_buf='data/datasets/pan15.csv', index=False)
    pan15_test_data.to_csv(path_or_buf='data/datasets/pan15_test.csv', index=False)
    
    # Read myPersonality data
    mypersonality_data = pd.read_csv('data/datasets/mypersonality_final.csv', encoding='mac-roman')
    pan15_train_data = pd.read_csv(filepath_or_buffer='data/datasets/pan15.csv')
    pan15_test_data = pd.read_csv(filepath_or_buffer='data/datasets/pan15_test.csv')

    # Rename columns to standardize column names between datasets
    mypersonality_data.rename(columns = {'#AUTHID':'USER', 'STATUS':'POST', 'sEXT':'EXT' , 'sNEU':'NEU', 'sAGR':'AGR', 'sCON':'CON','sOPN':'OPN'}, inplace=True)
    
    
    for dim in ['EXT', 'NEU', 'AGR', 'CON', 'OPN']:
        # Standardize personality trait scores to 0-1 scale
        mypersonality_data[dim] = mypersonality_data[dim].apply(scale_personality_score, args=(1, 5))
        pan15_train_data[dim] = pan15_train_data[dim].apply(scale_personality_score, args=(-0.5, 0.5))
        pan15_test_data[dim] = pan15_test_data[dim].apply(scale_personality_score, args=(-0.5, 0.5))
        

    # Standardize the textual posts
    # in facebook data mentions are not removed since they are non-existant
    mypersonality_data['POST'] = mypersonality_data['POST'].apply(standardize_text, args=(lowercase, punctuation_rem, url_rem,
                                                                                digit_rem, emoji_rem, special_char_rem, 0,
                                                                                hashtag_repl, expand_contractions,))
    pan15_train_data['POST'] = pan15_train_data['POST'].apply(standardize_text, args=(lowercase, punctuation_rem, url_rem,
                                                                                          digit_rem, emoji_rem, special_char_rem,
                                                                                          mention_repl, hashtag_repl, expand_contractions,))
    pan15_test_data['POST'] = pan15_test_data['POST'].apply(standardize_text, args=(lowercase, punctuation_rem, url_rem,
                                                                                          digit_rem, emoji_rem, special_char_rem,
                                                                                          mention_repl, hashtag_repl, expand_contractions,))
    
    # Drop lines with null text
    mypersonality_data = mypersonality_data[mypersonality_data['POST'].astype(bool)]
    pan15_train_data = pan15_train_data[pan15_train_data['POST'].astype(bool)]
    pan15_test_data = pan15_test_data[pan15_test_data['POST'].astype(bool)]
    
    # Remove users with insufficient information
    mypersonality_data = remove_undersampled_users(mypersonality_data, min_posts=10, min_tokens=100)
    pan15_train_data = remove_undersampled_users(pan15_train_data, min_posts=10, min_tokens=100)
    pan15_test_data = remove_undersampled_users(pan15_test_data, min_posts=10, min_tokens=100)
    
    # Load vocabulary for lemmatization
    nlp = spacy.load('en_core_web_sm')
    
    if lemmatize:
        # Perform lemmatization and create new column with the tokenized posts
        mypersonality_data['TOKENS'] = mypersonality_data['POST'].apply(tokenize_and_lemmatize, args=(nlp,))
        pan15_train_data['TOKENS'] = pan15_train_data['POST'].apply(tokenize_and_lemmatize, args=(nlp,))
        pan15_test_data['TOKENS'] = pan15_test_data['POST'].apply(tokenize_and_lemmatize, args=(nlp,))
    
        # Update posts column with lemmatized words
        mypersonality_data['POST'] = mypersonality_data['TOKENS'].apply(detokenize)
        pan15_train_data['POST'] = pan15_train_data['TOKENS'].apply(detokenize)
        pan15_test_data['POST'] = pan15_test_data['TOKENS'].apply(detokenize)
    
    # Save preprocessed data into CSV files
    mypersonality_data.to_csv(path_or_buf='data/preprocessed_data/mypersonality.csv', index=False)
    pan15_train_data.to_csv(path_or_buf='data/preprocessed_data/pan15.csv', index=False)
    pan15_test_data.to_csv(path_or_buf='data/preprocessed_data/pan15_test.csv', index=False)
    
# Preprocess data with desired operations (if argument set to 0, operation is not performed, otherwise it is)
# Scenario 1
#preprocessing(lowercase=1, punctuation_rem=0, url_rem=1, digit_rem=0, emoji_rem=1, special_char_rem=1, mention_repl=1, hashtag_repl=1, expand_contractions=0, lemmatize=0)
# Scenario 2
preprocessing(lowercase=1, punctuation_rem=1, url_rem=1, digit_rem=0, emoji_rem=1, special_char_rem=1, mention_repl=1, hashtag_repl=1, expand_contractions=1, lemmatize=1)