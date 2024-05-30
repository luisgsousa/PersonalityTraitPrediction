import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

from collections import Counter
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang

import torch
import os
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import Sequential
from keras.regularizers import l2, l1
from keras.layers import LSTM, Dense, Dropout, GRU, Flatten, Masking, LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import pad_sequences


def get_labels(data, trait):
    """Gets and returns perssonality score labels for selected trait, as a Numpy array
       
    
    Args:
        data(Dataframe): Dataframe containing all user data (user Id, posts and personality scores)
        trait(string): Trait for which personality scores will be returned
        
    Returns:
        Numpy array containing personality scores of the selected trait
    
    """
    
    user_ids = data['USER'].unique()
    
    labels = []
    for user_id in user_ids:
        labels.append(data.loc[data['USER'] == user_id, trait].iloc[0])
            
    return np.array(labels)

def get_bert_sentence_embeddings(data, embeddings_dir):
    """Extracts BERT word embeddings from the files the in disk, and extracts the sentence embeddings from them. 
    This is done by extracting the first word embedding corresponding to the "[CLS]" token
       
    
    Args:
        data(Dataframe): Dataframe containing all user data (user Id, posts and personality scores)
        embeddings_dir(string): Path where the embeddigns files are located
    
    """
    user_ids = data['USER'].unique()
    cls_embeddings = []
    
    for user_id in user_ids:

        # If embeddings file exists
        if(os.path.exists(embeddings_dir + user_id + '.pt')):
            # Load the embeddings as a tensor
            embeddings = torch.load(embeddings_dir + user_id + '.pt').numpy()
            # Extract first embedding, corresponding to "[CLS]" token and convert to list
            cls_embedding = embeddings[:,0,:].tolist()
            # Append to list
            cls_embeddings.append(cls_embedding)
            
    return cls_embeddings
         
def get_padding_mask(data):
    """Gets a padding mask, which can be used to scale the data while ignoring padding inputs.
       
    
    Args:
        data(ArrayLike): Data to extract padding mask from 
        
    Returns:
        data(Numpy Array): Padding mask 
    
    """
    
    # Get data shape
    shape = np.shape(data)
    
    # If an embedding contains only zeros, it is a padding input
    mask_1d = np.invert(np.all(data == 0, axis=(2))).astype(int)
    padding_mask = np.broadcast_to(mask_1d[:, :, np.newaxis], shape)
    
    return padding_mask
            
def scale_data(data):
    """Scales data according to MinMax method
       
    
    Args:
        data(ArrayLike): Data to be scaled 
        
    Returns:
        data(Numpy Array): Scaled data 
    
    """
    scaler = MinMaxScaler()

    # Posts are grouped per user, so must be degrouped by turning 3d array into 2d
    data_shape = np.shape(data)
    data = np.reshape(data, newshape=(-1, data_shape[2]))
    
    # Scale data
    data = scaler.fit_transform(data)
    # Return to original shape
    data = np.reshape(data, newshape=(data_shape[0], data_shape[1], data_shape[2]))
    
    return data

def create_regression_model(input_shape, learning_rate, lds_weights):
    """Creates regression model, which corresponds to a Keras neural network.
    Parameters can be changed by hand
       
    
    Args:
        input_shape(2D ArrayLike): shape of each input, corresponding to a collection of sentence embeddings of the user's posts
        learning_rate: Learning rate to use for training
        lds_weights: LDS weights to calculate adjusted training loss
        
    
    """
    # Create model
    model = Sequential()

    # Add masking layer to ignore padded inputs (dummy posts)
    model.add(Masking(mask_value=0.0, input_shape=(input_shape[1], input_shape[2])))

    # GRU layer tested in some configurations
    #model.add(GRU(8, return_sequences=True, kernel_regularizer=l2(0.0001)))
    #model.add(LeakyReLU())
    #model.add(Dropout(0.3))
    
    # Add dense layers
    model.add(Dense(256, kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    
    model.add(Dense(128, kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Dense(64, kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    
    # Flatten output before feeding into output layer
    model.add(Flatten())
    model.add(Dense(1 , activation='linear'))
        
    # Compile model with adjusted loss function
    model.compile(loss=weighted_mse_loss(weights=lds_weights), metrics=['mean_absolute_error', 'mean_squared_error'], optimizer=Adam(learning_rate=learning_rate, clipvalue=0.01))
    # Compile model with normal loss function
    #model.compile(loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'], optimizer=Adam(learning_rate=learning_rate, clipvalue=0.01))
        
    return model

def get_lds_kernel_window(kernel, ks, sigma):
    """Get LDS kernel window, with desired kernel function. Extracted from: https://github.com/YyzHarry/imbalanced-regression
    """
    
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window

def get_lds_weights(labels, bin_size, kernel, ks, sigma):
    """Get LDS weights from input. Extracted from: https://github.com/YyzHarry/imbalanced-regression
    """
    # Source: https://github.com/YyzHarry/imbalanced-regression
    
    # assign each label to its corresponding bin (start from 0)
    # with your defined get_bin_idx(), return bin_index_per_label: [Ns,] 
    bin_index_per_label = [int(label/bin_size) for label in labels]

    # calculate empirical (original) label distribution: [Nb,]
    # "Nb" is the number of bins
    Nb = max(bin_index_per_label) + 1
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]


    # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
    lds_kernel_window = get_lds_kernel_window(kernel=kernel, ks=ks, sigma=sigma)

    # calculate effective label distribution: [Nb,]
    eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')

    #plt.plot(range(Nb), emp_label_dist, color='b')
    #plt.plot(range(Nb), eff_label_dist, color='r')
    #plt.show()
    #quit()
    
    # Use re-weighting based on effective label distribution, sample-wise weights: [Ns,]
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]

def weighted_mse_loss(weights=None):
    """Calculate adjusted MSE loss. Extracted from https://github.com/YyzHarry/imbalanced-regression with minor changes

        Args:
            weights(ArrayLike): Weights to use to calculate adjusted loss
    
        Returns:
            loss: Adjusted MSE loss
    """
    def weighted_mse_loss_fn(inputs, targets):
        
        # Calculate squared error
        loss = (inputs - targets) ** 2
        # Multiply by lds_weights
        if weights is not None:
            loss *= weights.expand_as(loss)
        
        # Calculate mean
        loss = tf.reduce_mean(loss)
        return loss

    return weighted_mse_loss_fn

def weighted_mae_loss(weights=None):
    """Calculate adjusted MAE loss. Extracted from https://github.com/YyzHarry/imbalanced-regression with minor changes
    
        Returns:
            loss: Adjusted MAE loss
    """
    def weighted_mae_loss_fn(inputs, targets):
        
        # Calculate absolute error
        loss = tf.math.abs(inputs - targets)
        # Multiply by lds_weights
        if weights is not None:
            loss *= weights.expand_as(loss)
            
        # Calculate mean
        loss = tf.reduce_mean(loss)
        return loss

    return weighted_mae_loss_fn

def weighted_rmse_loss(weights=None):
    """Calculate adjusted RMSE loss. Extracted from https://github.com/YyzHarry/imbalanced-regression with minor changes
    
        Returns:
            loss: Adjusted RMSE loss
    """
    def weighted_rmse_loss_fn(inputs, targets):
        
        # Calculate root mean squared error
        loss = K.sqrt(K.mean(K.square(targets - inputs))) 

        # Multiply by lds weights
        if weights is not None:
            loss *= weights.expand_as(loss)
            
        # Calculate mean
        loss = tf.reduce_mean(loss)
        return loss

    return weighted_rmse_loss_fn

def adjusted_score_scales(Y_test, Y_pred):
    """Converts objective and predicted scores form [0, 1] scale to [1, 4] scale
    
        Args:
            Y_test: Objective score
            Y_pred: Predicted score
            
        Returns:
            Y_test: Scaled objective score
            Y_pred: Scaled predicted score
    """
    
    Y_test = (Y_test * 4) + 1
    Y_pred = (Y_pred * 4) + 1
    
    return Y_test, Y_pred


############################################
# Classification - K-Fold CV - Single dataset
############################################

############################################
# Parameters than can be manually changed:
# Dataset to use to train the model
train_dataset = 'twitter'
batch_size = 32
n_epochs = 80
learning_rate=0.0005
############################################

# Load datasets
facebook_data = pd.read_csv(filepath_or_buffer='data/preprocessed_data/mypersonality.csv')
pan15_data = pd.read_csv(filepath_or_buffer='data/preprocessed_data/pan15.csv')
pan15_test_data = pd.read_csv(filepath_or_buffer='data/preprocessed_data/pan15_test.csv')

# Get post embeddings and user labels
#facebook_embeddings = get_bert_sentence_embeddings(data=facebook_data,
#                                                           embeddings_dir='data/embeddings/distilbert/Scenario 1/facebook/')
#pan15_embeddings_1 = get_bert_sentence_embeddings(data=pan15_data,
#                                                         embeddings_dir='data/embeddings/distilbert/Scenario 1/pan15/')
#pan15_embeddings_2 = get_bert_sentence_embeddings(data= pan15_test_data,
#                                                         embeddings_dir='data/embeddings/distilbert/Scenario 1/pan15_test/')
          
# Load S-BERT embeddings                  
facebook_embeddings = torch.load('data/embeddings/sbert/Scenario 1/facebook_embeddings.pt').numpy()
pan15_embeddings_1 = torch.load('data/embeddings/sbert/Scenario 1/pan15_embeddings.pt').numpy()
pan15_embeddings_2 = torch.load('data/embeddings/sbert/Scenario 1/pan15_test_embeddings.pt').numpy()

# Get analytics of users to access their number of posts
facebook_analytics = pd.read_csv(filepath_or_buffer='data/analytics/facebook_analytics.csv', index_col=0)
pan15_analytics_1 = pd.read_csv(filepath_or_buffer='data/analytics/pan15_analytics.csv', index_col=0)
pan15_analytics_2 = pd.read_csv(filepath_or_buffer='data/analytics/pan15_test_analytics.csv', index_col=0)

# Pad all inputs to max length (max number of posts)
pad_length = max(facebook_analytics['NPOSTS'].max(), pan15_analytics_1['NPOSTS'].max(), pan15_analytics_2['NPOSTS'].max())
facebook_padded_embeddings = pad_sequences(facebook_embeddings, maxlen=pad_length, padding='post', dtype='float32')
pan15_padded_embeddings_1 = pad_sequences(pan15_embeddings_1, maxlen=pad_length, padding='post', dtype='float32')
pan15_padded_embeddings_2 = pad_sequences(pan15_embeddings_2, maxlen=pad_length, padding='post', dtype='float32')

# Combine PAN15 train and test sets
pan15_padded_embeddings = np.concatenate((pan15_padded_embeddings_1, pan15_padded_embeddings_2), axis=0)

results = []

for trait in ['OPN', 'CON', 'EXT', 'AGR', 'NEU']:
        
    # Get labels
    facebook_labels = get_labels(data=facebook_data, trait=trait)
    pan15_labels_1 = get_labels(data=pan15_data, trait=trait)
    pan15_labels_2 = get_labels(data=pan15_test_data, trait=trait)
    
    # Combine PAN15 train and test labels
    pan15_labels = np.concatenate((pan15_labels_1, pan15_labels_2), axis=0)

    # Create 5 folds for both datasets separately (to perform 5-fold CV)
    n_folds = 5
    if train_dataset == 'facebook':
        facebook_folds = list(KFold(n_folds, shuffle=True, random_state=42).split(facebook_padded_embeddings, facebook_labels))
    else:
        pan15_folds = list(KFold(n_folds, shuffle=True, random_state=42).split(pan15_padded_embeddings, pan15_labels))

    # For each fold
    for fold in range(n_folds):
        
        print('#########################')
        print(trait + ': Fold ' + str(fold))

        # If train set is Facebook, set train and validation data as Facebook data, and test data as Twitter data
        # Else do the opposite
        if train_dataset == 'facebook':
            # Get train and validation ids
            facebook_train_ids, facebook_val_ids = facebook_folds[fold]
            
            # Extract train and validation sets from Facebook data
            X_train = np.take(facebook_padded_embeddings, facebook_train_ids, 0)
            Y_train = np.take(facebook_labels, facebook_train_ids, 0)
            X_val = np.take(facebook_padded_embeddings, facebook_val_ids, 0)
            Y_val = np.take(facebook_labels, facebook_val_ids, 0)
            
            # Extract test sets from Twitter dataset 
            X_test = pan15_padded_embeddings
            Y_test = pan15_labels     
        else:
            # Get train and validation ids
            pan15_train_ids, pan15_val_ids = pan15_folds[fold]
            
            # Extract train and validation sets from Facebook data
            X_train = np.take(pan15_padded_embeddings, pan15_train_ids, 0)
            Y_train = np.take(pan15_labels, pan15_train_ids, 0)
            X_val = np.take(pan15_padded_embeddings, pan15_val_ids, 0)
            Y_val = np.take(pan15_labels, pan15_val_ids, 0)
            
            # Extract test sets from Twitter dataset 
            X_test = facebook_padded_embeddings
            Y_test = facebook_labels
        
        # Shuffle training data
        X_train, Y_train = shuffle(X_train, Y_train, random_state=42)

        ###################################
        # Scaling (not used in final model)
        ###################################

        # Scale input data and multiply by padding mask 
        # (scaling turns zeros into non zero values so they must be converted back to zeros)
        #padding_mask = get_padding_mask(X_train)
        #X_train = scale_data(X_train) * padding_mask

        #padding_mask = get_padding_mask(X_test)
        #X_test = scale_data(X_test) * padding_mask

        #scaler = MinMaxScaler()
        #X_train = scaler.fit_transform(X_train)
        #X_test = scaler.fit_transform(X_test)
        
        ###################################

        ###################################

        # Get LDS weights to apply to loss function
        lds_weights = get_lds_weights(labels=Y_train, bin_size=0.05, kernel='gaussian', ks=5, sigma=2)
        model = create_regression_model(input_shape=np.shape(X_train), learning_rate=learning_rate, lds_weights=lds_weights)
        
        # Create early stopping callback
        earlystopping = EarlyStopping(monitor="val_mean_absolute_error", patience = 10, 
                                    mode="min", restore_best_weights=True, start_from_epoch=20)
        callbacks = (earlystopping)

        # fit model to training data, with early stopping based on validation data
        history = model.fit(X_train, Y_train, batch_size=batch_size, epochs = n_epochs, validation_data=(X_val, Y_val), callbacks=callbacks)
        
        # Predict validation scores
        y_pred_val = model.predict(X_val)
        y_pred_val = np.array(y_pred_val).ravel()
        
        # Calculate MAE and RMSE for validation set
        val_mae = mean_absolute_error(Y_val, y_pred_val)
        val_rmse = mean_squared_error(Y_val, y_pred_val, squared=False)
        # Scale scores to [1, 5] scale and calculate MSE for validation set
        scaled_Y_test, scaled_y_pred = adjusted_score_scales(Y_val, y_pred_val)
        val_mse = mean_squared_error(scaled_Y_test, scaled_y_pred, squared=True)

        # Predict test scores
        y_pred_test = model.predict(X_test)
        y_pred_test = np.array(y_pred_test).ravel()
        
        # Calculate MAE and RMSE for test set
        test_mae = mean_absolute_error(Y_test, y_pred_test)
        test_rmse = mean_squared_error(Y_test, y_pred_test, squared=False)
        # Scale scores to [1, 5] scale and calculate MSE for test set
        scaled_Y_test, scaled_y_pred = adjusted_score_scales(Y_test, y_pred_test)
        test_mse = mean_squared_error(scaled_Y_test, scaled_y_pred, squared=True)
        
        # Print val and test MAE's
        print('Val MAE ' + str(val_mae))
        print('Test MAE ' + str(test_mae))
        
        # Plot and save objective validation scores distribution
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.hist(Y_val, bins=np.arange(0, 1, 0.05))
        plt.title('Y_val ' + str(trait))

        # Plot and save predicted validation scores distribution
        plt.subplot(1, 2, 2)
        plt.hist(y_pred_val, bins=np.arange(0, 1, 0.05))
        plt.title('Y_pred ' + str(trait))
        plt.savefig('results/Single Dataset/prediction_distribution_' + str(trait) + '_fold_' + str(fold))
        plt.close()
        
        # Get training and validation MAE, from the training process
        val_mae_log = history.history['val_mean_absolute_error']
        train_mae_log = history.history['mean_absolute_error']
        
        # Plot and save train and validation MAE
        plt.figure()
        plt.plot(range(len(train_mae_log)), train_mae_log, color='b', label='Train MAE')
        plt.plot(range(len(val_mae_log)), val_mae_log, color='r', label='Test MAE')
        plt.legend()
        plt.title('Train and test MAE ' + str(trait))
        plt.savefig('results/Single Dataset/train_val_mae_ ' + str(trait) + '_fold_' + str(fold))
        plt.close()

        # Append results (MAE, RMSE and MSE) from this fold
        if train_dataset == 'facebook':
            results.append({'trait':trait, 'fold':fold, 'facebook_mae':val_mae,'pan15_mae':test_mae,
                            'facebook_rmse':val_rmse, 'pan15_rmse':test_rmse,
                            'facebook_mse':val_mse, 'pan15_mse':test_mse})
        else:
            results.append({'trait':trait, 'fold':fold, 'facebook_mae':test_mae,'pan15_mae':val_mae,
                            'facebook_rmse':test_rmse, 'pan15_rmse':val_rmse,
                            'facebook_mse':test_mse, 'pan15_mse':val_mse})
    
# Save results to CSV file
results = pd.DataFrame(results)
results.to_csv(path_or_buf='results/Single Dataset/results.csv', index=False)

# Calculate average of the metrics, across 5 folds, for all 5 traits
avg_results = []
for trait in ['OPN', 'CON', 'EXT', 'AGR', 'NEU']:
    facebook_avg_mae = results.loc[results['trait'] == trait, 'facebook_mae'].mean()
    pan15_avg_mae = results.loc[results['trait'] == trait, 'pan15_mae'].mean()
    facebook_avg_rmse = results.loc[results['trait'] == trait, 'facebook_rmse'].mean()
    pan15_avg_rmse = results.loc[results['trait'] == trait, 'pan15_rmse'].mean()
    facebook_avg_mse = results.loc[results['trait'] == trait, 'facebook_mse'].mean()
    pan15_avg_mse = results.loc[results['trait'] == trait, 'pan15_mse'].mean()
    
    avg_results.append({'trait':trait, 'facebook_avg_mae':facebook_avg_mae, 'pan15_avg_mae':pan15_avg_mae,
                        'facebook_avg_rmse':facebook_avg_rmse, 'pan15_avg_rmse':pan15_avg_rmse, 
                        'facebook_avg_mse':facebook_avg_mse, 'pan15_avg_mse':pan15_avg_mse})
    
# Save average results to CSV file
avg_results = pd.DataFrame(avg_results)
avg_results.to_csv(path_or_buf='results/Single Dataset/avg_results.csv', index=False)