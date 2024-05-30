import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

def get_labels(data, trait):
    user_ids = data['USER'].unique()
    labels = []
    
    for user_id in user_ids:
        labels.append(data.loc[data['USER'] == user_id, trait].iloc[0])
            
    return labels

def adjusted_score_scales(Y_test, Y_pred):
    
    Y_pred = (Y_pred * 4) + 1
    Y_test = (Y_test * 4) + 1
    
    return Y_pred, Y_test

#########################################
#Single Dataset
#########################################

facebook_data = pd.read_csv(filepath_or_buffer='data/preprocessed_data/mypersonality.csv')
pan15_data = pd.read_csv(filepath_or_buffer='data/preprocessed_data/pan15.csv')
pan15_test_data = pd.read_csv(filepath_or_buffer='data/preprocessed_data/pan15_test.csv')

for train_data in ['facebook', 'twitter']:
    
    results = []

    for trait in ['OPN', 'CON', 'EXT', 'AGR', 'NEU']:
        # Get user labels
        facebook_labels = get_labels(facebook_data, trait)
        pan15_labels_1 = get_labels(pan15_data, trait)
        pan15_labels_2 = get_labels(pan15_test_data, trait)

        pan15_labels = np.concatenate((pan15_labels_1, pan15_labels_2), axis=0)
        
        n_folds = 5
        facebook_folds = list(KFold(n_folds, shuffle=True, random_state=42).split(facebook_labels))
        pan15_folds = list(KFold(n_folds, shuffle=True, random_state=42).split(pan15_labels))

        facebook_mae = []
        facebook_mse = []
        facebook_rmse = []
        pan15_mae = []
        pan15_mse = []
        pan15_rmse = []

        for fold in range(n_folds):
            facebook_train_ids, facebook_test_ids = facebook_folds[fold]
            pan15_train_ids, pan15_test_ids = pan15_folds[fold]
            
            facebook_y_train = np.take(facebook_labels, facebook_train_ids, 0)
            facebook_y_test = np.take(facebook_labels, facebook_test_ids, 0)
            pan15_y_train = np.take(pan15_labels, pan15_train_ids, 0)
            pan15_y_test = np.take(pan15_labels, pan15_test_ids, 0)

            if train_data == 'facebook':
                Y_train = facebook_y_train
            else:
                Y_train = pan15_y_train

            mean = np.mean(Y_train)

            facebook_preds = np.repeat(mean , len(facebook_y_test))
            pan15_preds = np.repeat(mean, len(pan15_y_test))
            
            facebook_mae.append(mean_absolute_error(facebook_y_test, facebook_preds))
            facebook_rmse.append(mean_squared_error(facebook_y_test, facebook_preds, squared=False))
            scaled_y_pred, scaled_Y_test = adjusted_score_scales(facebook_y_test, facebook_preds)
            facebook_mse.append(mean_squared_error(scaled_Y_test, scaled_y_pred))
            
            pan15_mae.append(mean_absolute_error(pan15_y_test,pan15_preds))
            pan15_rmse.append(mean_squared_error(pan15_y_test, pan15_preds, squared=False))
            scaled_y_pred, scaled_Y_test = adjusted_score_scales(pan15_y_test, pan15_preds)
            pan15_mse.append(mean_squared_error(scaled_Y_test, scaled_y_pred))

        results.append({'dim':trait,
                        'facebook_mae':np.mean(facebook_mae),'pan15_mae':np.mean(pan15_mae),
                        'facebook_rmse':np.mean(facebook_rmse), 'pan15_rmse':np.mean(pan15_rmse),
                        'facebook_mse':np.mean(facebook_mse),'pan15_mse':np.mean(pan15_mse)})
        
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/single dataset/baseline/baseline_results_' + train_data + '.csv')

#########################################
#Cross-media
#########################################

'''results = []

facebook_data = pd.read_csv(filepath_or_buffer='data/preprocessed_data/mypersonality.csv')
pan15_data = pd.read_csv(filepath_or_buffer='data/preprocessed_data/pan15.csv')
pan15_test_data = pd.read_csv(filepath_or_buffer='data/preprocessed_data/pan15_test.csv')

for trait in ['OPN', 'CON', 'EXT', 'AGR', 'NEU']:
    # Get user labels
    facebook_labels = get_labels(facebook_data, trait)
    pan15_labels_1 = get_labels(pan15_data, trait)
    pan15_labels_2 = get_labels(pan15_test_data, trait)

    pan15_labels = np.concatenate((pan15_labels_1, pan15_labels_2), axis=0)
    
    n_folds = 5
    facebook_folds = list(KFold(n_folds, shuffle=True, random_state=42).split(facebook_labels))
    pan15_folds = list(KFold(n_folds, shuffle=True, random_state=42).split(pan15_labels))

    facebook_mae = []
    facebook_mse = []
    facebook_rmse = []
    pan15_mae = []
    pan15_mse = []
    pan15_rmse = []
    full_mae = []
    full_mse = []
    full_rmse = []

    for fold in range(n_folds):
        facebook_train_ids, facebook_test_ids = facebook_folds[fold]
        pan15_train_ids, pan15_test_ids = pan15_folds[fold]
        
        facebook_y_train = np.take(facebook_labels, facebook_train_ids, 0)
        facebook_y_test = np.take(facebook_labels, facebook_test_ids, 0)
        pan15_y_train = np.take(pan15_labels, pan15_train_ids, 0)
        pan15_y_test = np.take(pan15_labels, pan15_test_ids, 0)

        Y_train = np.concatenate((facebook_y_train, pan15_y_train), axis=0)
        Y_test = np.concatenate((facebook_y_test, pan15_y_test), axis=0)

        mean = np.mean(Y_train)

        facebook_preds = np.repeat(mean , len(facebook_y_test))
        pan15_preds = np.repeat(mean, len(pan15_y_test))
        Y_pred = np.repeat(mean, len(Y_test))
        
        facebook_mae.append(mean_absolute_error(facebook_y_test, facebook_preds))
        facebook_rmse.append(mean_squared_error(facebook_y_test, facebook_preds, squared=False))
        facebook_mse.append(mean_squared_error(facebook_y_test, facebook_preds))
        
        pan15_mae.append(mean_absolute_error(pan15_y_test,pan15_preds))
        pan15_rmse.append(mean_squared_error(pan15_y_test, pan15_preds, squared=False))
        pan15_mse.append(mean_squared_error(pan15_y_test, pan15_preds))
        
        full_mae.append(mean_absolute_error(Y_test, Y_pred))
        full_rmse.append(mean_squared_error(Y_test, Y_pred, squared=False))
        full_mse.append(mean_squared_error(Y_test, Y_pred))

    results.append({'dim':trait,
                    'facebook_mae':np.mean(facebook_mae),'pan15_mae':np.mean(pan15_mae), 'full_mae':np.mean(full_mae),
                    'facebook_rmse':np.mean(facebook_rmse), 'pan15_rmse':np.mean(pan15_rmse), 'full_rmse':np.mean(full_rmse),
                    'facebook_mse':np.mean(facebook_mse),'pan15_mse':np.mean(pan15_mse),'full_mse':np.mean(full_mse)})
results_df = pd.DataFrame(results)
results_df.to_csv('results/cross-media learning/baseline/baseline_results_cross_media.csv')'''