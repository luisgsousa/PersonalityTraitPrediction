import pandas as pd
import numpy as np
import torch
import tensorflow as tf
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer,DistilBertModel, DistilBertTokenizer
from transformers import XLNetModel, XLNetTokenizerFast
from sentence_transformers import SentenceTransformer
from keras.utils import pad_sequences
import openai
import os

def get_bert_word_embeddings(data, model, directory):
    """Extracts BERT word embeddings for each user's posts and saves the resulting tensor as a .pt file. 
    Does this for all users
       
    
    Args:
        data(Dataframe): Dataframe containing all user data (user Id, posts and personality scores)
        model(string): Model to use to extract embeddings ('bert', 'roberta' or 'distilbert')
        directory(string): Desired path to save embeddings
    
    """

    # Load chosen pretrained model
    # Embeddings will be derived from the output of this model
    # Get tokenizer used in the specific model to ensure consistency
    if model == 'bert':
        model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif model == 'roberta':
        model = RobertaModel.from_pretrained('roberta-base-uncased', output_hidden_states = True)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base-uncased')
    elif model == 'distilbert':
        model = DistilBertModel.from_pretrained('distilbert-base-uncased', output_hidden_states = True)
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    else:
        print('Error: Invalid Model')
        exit(0)
    
    
    count = 0
    # get user ID's
    users = data['USER'].unique()
    for user in users:
        # List where embeddings will be stored
        embeddings = []
        # Print and update progress
        print('step ' + str(count) + ' of ' + str(len(users)))
        count += 1
        # Check if the embeddings for the user were already extracted
        if not os.path.exists(directory + user + '.pt'):
            # Get Series with the user's posts
            user_posts = data[data['USER'] == user]['POST']
            
            for post in user_posts:
                # Prepare text and tokenize it using the bert tokenizer
                tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text=post, tokenizer=tokenizer, max_length=50)
                # Extract embeddings and append to list
                embeddings.append(extract_bert_embeddings(tokens_tensor, segments_tensors, model))
                
            # Convert list of embeddings to tensor
            embeddings_tensor = torch.Tensor(embeddings)
            # Save tensor in specified directory with the user ID as filename 
            torch.save(embeddings_tensor, directory + user + '.pt')
            
def get_sbert_embeddings(data, directory):
    """Extracts S-BERT sentence embeddings from a Dataframe containing collections of posts from
    multiple users.
       
    
    Args:
        data(Dataframe): Dataframe containing all user data (user Id, posts and personality scores)
        directory(string): Desired path to save embeddings
    
    """

    # Get model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Change the maximum sequence length to 200
    model.max_seq_length = 200
    
    # List to store final tensor sentence embeddings
    embeddings = []
    # List to store number of posts per user
    n_posts_per_user = []
    
    # Get user ID's
    users = data['USER'].unique()
    
    count=0
    for user in users:
        user_embeddings = []
        
        # Print and update progress
        print('step ' + str(count) + ' of ' + str(len(users)))
        count += 1

        # Get collection of user posts
        user_posts = data[data['USER'] == user]['POST']
        # Append number of user posts
        n_posts_per_user.append(len(user_posts))
        
        # Append sentence embedding for each post by the user
        for post in user_posts:
            # Sentences are encoded by calling model.encode()
            user_embeddings.append(model.encode(post))
        # Append collection of user sentence embeddings to list
        embeddings.append(user_embeddings)
    
    # Pad input so all users have the same number of embeddings
    pad_length = max(n_posts_per_user)
    embeddings = pad_sequences(embeddings, maxlen=pad_length, padding='post', dtype='float32')
    
    # Convert to tensor and save in desired path
    embeddings_tensor = torch.Tensor(embeddings)
    torch.save(embeddings_tensor, directory)
    
    
            
def bert_text_preparation(text, tokenizer, max_length):
    """Preparing the input for BERT (from https://towardsdatascience.com/3-types-of-contextualized-word-embeddings-from-bert-using-transfer-learning-81fcefe3fe6d)
    
    Takes a string argument and performs
    pre-processing like adding special tokens,
    tokenization, tokens to ids, and tokens to
    segment ids. All tokens are mapped to seg-
    ment id = 1.
    
    Args:
        text (str): Text to be converted
        tokenizer (obj): Tokenizer object
            to convert text into BERT-re-
            adable tokens and ids
        
    Returns:
        list: List of BERT-readable tokens
        obj: Torch tensor with token ids
        obj: Torch tensor segment ids
    
    
    """
    
    # Add special tokens required by BERT
    marked_text = "[CLS] " + text + " [SEP]"
    # Get encoded tokens as well as padding/truncating posts to max_length
    encoded_text = tokenizer(marked_text, padding='max_length', truncation=True, max_length=max_length)
    # Decode to obtain tokenized tokens
    tokenized_text = tokenizer.convert_ids_to_tokens(encoded_text['input_ids'])
    
    indexed_tokens = encoded_text['input_ids']
    # Comment this later
    segments_ids = [1]*len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    
    return tokenized_text, tokens_tensor, segments_tensors

def extract_bert_embeddings(tokens_tensor, segments_tensors, model):
    """Get embeddings from an embedding model (from https://towardsdatascience.com/3-types-of-contextualized-word-embeddings-from-bert-using-transfer-learning-81fcefe3fe6d)
    
    Args:
        tokens_tensor (obj): Torch tensor size [n_tokens]
            with token ids for each token in text
        segments_tensors (obj): Torch tensor size [n_tokens]
            with segment ids for each token in text
        model (obj): Embedding model to generate embeddings
            from token and segment ids
    
    Returns:
        list: List of lists of floats, each of size
            [n_tokens, n_embedding_dimensions]
            containing embeddings for each token
    
    """
    
    # Gradient calculation id disabled
    # Model is in inference mode
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # Removing the first hidden state
        # The first state is the input state
        hidden_states = outputs.hidden_states[-4:]
    
    # Getting embeddings by averaging BERT's final 4 layers
    hidden_states = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.mean(hidden_states, dim=0)
    # Collapsing the tensor into 1-dimension
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    # Converting torchtensors to lists
    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]
    
    return list_token_embeddings

def extract_features():
    
    # Get facebook and pan15 datsets from their respective files
    facebook_data = pd.read_csv(filepath_or_buffer='data/preprocessed_data/mypersonality.csv')
    pan15_twitter_data = pd.read_csv(filepath_or_buffer='data/preprocessed_data/pan15.csv')
    pan15_twitter_data_test = pd.read_csv(filepath_or_buffer='data/preprocessed_data/pan15_test.csv')

    
    # Extract and save DistilBERT embeddings
    '''model = 'distilbert'
    directory = 'data/embeddings/distilbert/Scenario 2/facebook/'
    get_bert_word_embeddings(facebook_data, model, directory)
    directory = 'data/embeddings/distilbert/Scenario 2/pan15/'
    get_bert_word_embeddings(pan15_twitter_data, model, directory)
    directory = 'data/embeddings/distilbert/Scenario 2/pan15_test/'
    get_bert_word_embeddings(pan15_twitter_data_test, model, directory)'''

    # Extract and save S-BERT embeddings
    get_sbert_embeddings(data=facebook_data, directory='data/embeddings/sbert/Scenario 2/facebook_embeddings.pt')
    get_sbert_embeddings(data=pan15_twitter_data, directory='data/embeddings/sbert/Scenario 2/pan15_embeddings.pt')
    get_sbert_embeddings(data=pan15_twitter_data_test, directory='data/embeddings/sbert/Scenario 2/pan15_test_embeddings.pt')

extract_features()

#######################################################################
# Unused functions for early tests with other models (XLNet, GPT-3)
#######################################################################

def get_openai_embeddings(texts):
    
    # Set OpenAI key obtained from OpenAI API page
    openai.api_key = openAI_key

    # Get OpenAI embedding
    embeddings = openai.Embedding.create(input=texts, model="text-embedding-ada-002")["data"][0]["embedding"]

    return embeddings

def concatenate_user_posts(data):
    user_ids = data['USER'].unique()
    posts_by_user = []
    # Using OpenAI's tokenizer for consistency
    encoding = tiktoken.get_encoding("cl100k_base")
    
    for user_id in user_ids:
        user_posts = ''
        for post in data.loc[data['USER'] == user_id, 'POST']:
            user_posts = user_posts + ' || ' + post
        
        n_tokens = len(encoding.encode(user_posts))
        
        if n_tokens > 8191:            
            tokens = encoding.encode(user_posts)
            tokens = tokens[:8191]
            
            user_posts = encoding.decode(tokens)
            
            
            
        
        posts_by_user.append([user_id, user_posts])

        
    posts_df = pd.DataFrame(posts_by_user, columns=['USER', 'POST'])
            
    return posts_df

def get_xlnet_embeddings(data, max_length, directory):

    users = data['USER'].unique()

    tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-base-cased')
    model = XLNetModel.from_pretrained('xlnet-base-cased')

    count = 0
    for user in users:
        embeddings = []
        
        print('step ' + str(count) + ' of ' + str(len(users)))
        count += 1

        if not os.path.exists(directory + user + '.pt'):
            user_posts = data[data['USER'] == user]['POST']
            
            for post in user_posts:
            
                input = tokenizer(post, return_tensors="pt",  padding='max_length', truncation=True, max_length=max_length)
                output = model(**input)
                
                embeddings.append(output.last_hidden_state.tolist())
                    
            embeddings_tensor = torch.squeeze(torch.Tensor(embeddings))
            torch.save(embeddings_tensor, directory + user + '.pt')
