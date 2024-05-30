import pandas as pd
from matplotlib import pyplot as plt
import nltk
from math import floor
import numpy as np

def data_analytics(data):
    """Calculate analytics for specified data.
    
    Calculates the toal number of posts and tokens per user. 
    Returns this data combined with the user ID and the personality scores.
    
        Args:
            data(ArrayLike): Data to compute analytics
            
        Returns:
            user_analytics(Dataframe): Contains the analytics for the user, as well a sthe user's ID and personality scores
    
    """
    user_ids = data['USER'].unique()
    user_counts = []

    count = 1
    for user_id in user_ids:
        user_posts = data.loc[data['USER'] == user_id, 'POST']
        n_tokens_total = 0
        n_posts = 0
        
        # Get user personality scores
        ext = data.loc[data['USER'] == user_id, 'EXT'].iloc[0]
        neu = data.loc[data['USER'] == user_id, 'NEU'].iloc[0]
        agr = data.loc[data['USER'] == user_id, 'AGR'].iloc[0]
        con = data.loc[data['USER'] == user_id, 'CON'].iloc[0]
        opn = data.loc[data['USER'] == user_id, 'OPN'].iloc[0]
        
        # For each post, add word count of the post and increase post count
        for post in user_posts:
            words = nltk.word_tokenize(post)
            n_tokens_total += len(words)
            n_posts += 1
            
        # Append user analytics
        user_counts.append([user_id, n_posts, n_tokens_total, ext, neu, agr, con, opn])
        count += 1

    user_analytics = pd.DataFrame(user_counts, columns=['USERID', 'NPOSTS', 'NTOKENS', 'EXT', 'NEU', 'AGR', 'CON', 'OPN'])
    
    return user_analytics
    
def print_statistics(analytics):
    """Print some statistics regarding the analytics calculated in data_analytics function.
    
        Args:
            analytics(Dataframe): Analytics from data_analytics function
    
    """
    
    # Print minimum, average and max number of tokens in data
    min_words = analytics['NTOKENS'].min()
    max_words = analytics['NTOKENS'].max()
    mean_words = analytics['NTOKENS'].mean()
    
    print('Number of tokens ranges from ' + str(min_words) + ' to ' + str(max_words) + ', with an average of ' + str(mean_words))
    
    # Print minimum, average and max number of posts in data
    min_posts = analytics['NPOSTS'].min()
    max_posts = analytics['NPOSTS'].max()
    mean_posts = analytics['NPOSTS'].mean()
    
    print('Number of posts ranges from ' + str(min_posts) + ' to ' + str(max_posts) + ', with an average of ' + str(mean_posts))
    
    # Print total number of posts in data
    n_total_posts = analytics['NPOSTS'].sum()
    print('Total number of posts: ' + str(n_total_posts))


def plot_histograms(data, bins, title, xlabel, ylabel, fig_name):
    """Plots histogram and saves it as a figure.
    
        Args:
            data(ArrayLike): data to plot in histogram
            bins: Edge values for the histogram bins
            title: Title for figure
            xlabel: Label for x axis
            ylabel: LAbel for y axix«s
            fig_name: Name of the saved figure
    
    """
    plt.figure()
    plt.hist(data, bins=bins)
    plt.grid(color='white', lw = 0.5, axis='x')
    plt.xticks(bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig('data/analytics/histograms/' + fig_name)
    
def user_distribution(data):
    """Calculate user score distribution for the 5 traits
    
        Args:
            data(ArrayLike): data to calculate distribution from
            
    """
    print('Calculating score distributions...')
    opn = user_score_dist(data, 'OPN')
    con = user_score_dist(data, 'CON')
    ext = user_score_dist(data, 'EXT')
    agr = user_score_dist(data, 'AGR')
    neu = user_score_dist(data, 'NEU')


    scores_dist = {'OPN':opn, 'CON':con, 'EXT':ext, 'AGR':agr, 'NEU':neu}
        
    return pd.DataFrame(scores_dist)

def user_score_dist(data, trait):
    """Calculate user score distribution for the specified trait, from 1 to 0, in steps of 0.2
    
        Args:
            data(ArrayLike): data to calculate distribution from
            
    """
    user_ids = []

    for i in np.arange(0, 1, 0.2):
        user_ids.append(data[(i < data[trait]) & (data[trait] <= i + 1)]['USER'].unique().tolist())
            
    return user_ids

# Get original and preprocessed datasets for comparison
original_facebook_data = pd.read_csv('data/datasets/mypersonality_final.csv',  encoding='mac-roman')
preprocessed_facebook_data = pd.read_csv('data/preprocessed_data/mypersonality.csv')

original_pan15_data_1 = pd.read_csv('data/datasets/pan15.csv')
original_pan15_data_2 = pd.read_csv('data/datasets/pan15_test.csv')
original_pan15_data = pd.concat([original_pan15_data_1, original_pan15_data_2])

preprocessed_pan15_data_1 = pd.read_csv('data/preprocessed_data/pan15.csv')
preprocessed_pan15_data_2 = pd.read_csv('data/preprocessed_data/pan15_test.csv')
preprocessed_pan15_data = pd.concat([preprocessed_pan15_data_1, preprocessed_pan15_data_2])

# Rename columns for standardization purposes
original_facebook_data.rename(columns = {'#AUTHID':'USER', 'STATUS':'POST', 'sEXT':'EXT', 'sNEU':'NEU', 'sAGR':'AGR', 'sCON':'CON', 'sOPN':'OPN'}, inplace=True)

# Save preprocessed data analytics to use in prediction
user_analytics = data_analytics(preprocessed_pan15_data_1)
user_analytics.to_csv(path_or_buf='data/analytics/pan15_analytics.csv')

user_analytics = data_analytics(preprocessed_pan15_data_2)
user_analytics.to_csv(path_or_buf='data/analytics/pan15_test_analytics.csv')

user_analytics = data_analytics(preprocessed_facebook_data)
user_analytics.to_csv(path_or_buf='data/analytics/facebook_analytics.csv')

# Bins for histogram plots
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Plot histograms for preprocessed Facebook dataset 5 traits
plot_histograms(data=user_analytics['EXT'],
                bins=bins,
                title='Facebook: Extraversion scores distribution',
                xlabel='Extraversion score',
                ylabel='Number of users',
                fig_name='facebook_ext')
plot_histograms(data=user_analytics['NEU'],
                bins=bins,
                title='Facebook: Neuroticism scores distribution',
                xlabel='Neuroticism score',
                ylabel='Number of users',
                fig_name='facebook_neu')
plot_histograms(data=user_analytics['AGR'],
                bins=bins,
                title='Facebook: Agreeableness scores distribution',
                xlabel='Agreeableness score',
                ylabel='Number of users',
                fig_name='facebook_agr')
plot_histograms(data=user_analytics['CON'],
                bins=bins,
                title='Facebook: Conscientiousness scores distribution',
                xlabel='Conscientiousness score',
                ylabel='Number of users',
                fig_name='facebook_con')
plot_histograms(data=user_analytics['OPN'],
                bins=bins,
                title='Facebook: Openness scores distribution',
                xlabel='Openness score',
                ylabel='Number of users',
                fig_name='facebook_opn')

user_analytics = data_analytics(preprocessed_pan15_data)

# Plot histograms for preprocessed PAN15 dataset 5 traits
plot_histograms(data=user_analytics['EXT'],
                bins=bins,
                title='Twitter(PAN15): Extraversion scores distribution',
                xlabel='Extraversion score',
                ylabel='Number of users',
                fig_name='pan15_ext')
plot_histograms(data=user_analytics['NEU'],
                bins=bins,
                title='Twitter(PAN15): Neuroticism scores distribution',
                xlabel='Neuroticism score',
                ylabel='Number of users',
                fig_name='pan15_neu')
plot_histograms(data=user_analytics['AGR'],
                bins=bins,
                title='Twitter(PAN15): Agreeableness scores distribution',
                xlabel='Agreeableness score',
                ylabel='Number of users',
                fig_name='pan15_agr')
plot_histograms(data=user_analytics['CON'],
                bins=bins,
                title='Twitter(PAN15): Conscientiousness scores distribution',
                xlabel='Conscientiousness score',
                ylabel='Number of users',
                fig_name='pan15_con')
plot_histograms(data=user_analytics['OPN'],
                bins=bins,
                title='Twitter(PAN15): Openness scores distribution',
                xlabel='Openness score',
                ylabel='Number of users',
                fig_name='pan15_opn')

# Print analytics and total post number for original facebook dataset
print('Original facebook data:')
user_analytics = data_analytics(original_facebook_data)
print_statistics(user_analytics)
count = 0
for n_posts in user_analytics['NPOSTS']:
    count += n_posts
print(count)

# Print analytics and total post number for preprocessed Facebook dataset
print('Preprocessed facebook data:')
user_analytics = data_analytics(preprocessed_facebook_data)
print_statistics(user_analytics)
count = 0
for n_posts in user_analytics['NPOSTS']:
    count += n_posts
print(count)

# Print analytics and total post number for original PAN15 dataset
print('Original PAN15 data:')
user_analytics = data_analytics(original_pan15_data)
print_statistics(user_analytics)
count = 0
for n_posts in user_analytics['NPOSTS']:
    count += n_posts
print(count)

# Print analytics and total post number for preprocessed PAN15 dataset
print('Preprocessed PAN15 data:')
user_analytics = data_analytics(preprocessed_pan15_data)
print_statistics(user_analytics)
count = 0
for n_posts in user_analytics['NPOSTS']:
    count += n_posts
print(count)

# Calculate and save user score distributions for both preprocessing datasets to use in prediction script
facebook_distribution = user_distribution(preprocessed_facebook_data)
pan15_distribution  = user_distribution(preprocessed_pan15_data)

facebook_distribution.to_csv(path_or_buf='data/analytics/facebook_score_counts.csv')
pan15_distribution.to_csv(path_or_buf='data/analytics/pan15_score_counts.csv')