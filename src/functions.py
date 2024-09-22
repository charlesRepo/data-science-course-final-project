import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from sentence_transformers import SentenceTransformer # pip install -U sentence-transformers
import nltk
from nltk.corpus import stopwords
import numpy as np
import os
import matplotlib.pyplot as plt
import streamlit as st
import requests
import sys

root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(root_dir)

from config import GITHUB_TOKEN




def add_growth_score_based_on_main_features(df, train=True):
    """
    Adds a 'growth_score' column to the dataframe based on the scaled values of
    specified feature columns ('num_stars', 'num_forks', 'num_pull_requests').

    The growth score is calculated as the sum of the scaled values of these features.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the relevant features.

    
    Returns:
    pd.DataFrame: The dataframe with the new 'growth_score' column added.
    """

    feature_columns = ['num_stars', 'num_forks', 'num_watchers', 'num_pull_requests', 'num_releases', 'num_open_issues']
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(current_dir, '../models/growth_score_training_scaler.pkl')
    scaler_path = os.path.abspath(data_file_path)

    # Extract the relevant features for scaling
    features = df[feature_columns]
    
    if train == True:
        # Fit the scaler if none is provided
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)
        # Save the fitted scaler after training
        joblib.dump(scaler, scaler_path)

    else:
        growth_score_training_scaler = joblib.load(scaler_path)

        # Use the provided scaler
        scaled_features = growth_score_training_scaler.transform(features)

    # Calculate the growth_score as the sum of the scaled features
    df['growth_score'] = scaled_features.sum(axis=1)
    return df


def add_repo_age_days_col(df):
    df['creation_date'] = pd.to_datetime(df['creation_date'])
    now = pd.Timestamp.now(tz='UTC')
    df['repo_age_days'] = (now - df['creation_date']).dt.days
    df.drop(columns=['creation_date'], inplace=True)
    return df


def add_days_since_last_release_col(df):
    df['release_date'] = pd.to_datetime(df['release_date'])
    now = pd.Timestamp.now(tz='UTC')
    df['days_since_last_release'] = (now - df['release_date']).dt.days
    return df


def remove_stop_words_from_topics(df):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    def remove_stopwords(text):
        return ' '.join([word for word in str(text).split() if word.lower() not in stop_words])
    df['topics'] = [remove_stopwords(topics) for topics in df.topics]
    
    return df

    

def convert_topics_to_embeddings(df):
    # Apply stop words removal function
    df = remove_stop_words_from_topics(df)
    
    # Convert Topics to embeddings
    sentence_embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
    df['topic_embedding'] = df['topics'].apply(lambda topic: sentence_embeddings_model.encode(topic))

    # Spread embeddings across multiple columns
    topic_embedding_df = pd.DataFrame(df['topic_embedding'].tolist(), index=df.index)
    df = pd.concat([df, topic_embedding_df], axis=1)
    df.drop(columns=['topic_embedding', 'topics'], inplace=True)
    
    return df

def indexify_release_dates(df):
    df = df.set_index('release_date')
    df = df.sort_index()
    return df

def get_num_of_releases(df):
    return df['num_releases'].astype(int).unique()[0]


def add_lag_features_based_on_target(df, num = 4):
    for i in range(1, num+1):
        df[f'growth_score_lag_{i}'] = df['growth_score'].shift(i)
    df = df.dropna()
    return df


def remove_unwanted_features(df):
    return df.drop(columns=['growth_score', 'org_name', 'repo_name', 'description', 'repo_url', 'release_tag', 'update_date'])


def scale_final_data(X_train=None, X_test=None, X=None, scaler_path='../models/final_input_scaler.pkl'):
    """
    Scales the data using MinMaxScaler and handles both train/test split and manual testing scenarios.

    If X_train and X_test are provided, the function scales them both and fits the scaler on X_train.
    If X is provided, it only scales X using a pre-fitted scaler.

    Parameters:
    X_train (np.array or pd.DataFrame, optional): The training data.
    X_test (np.array or pd.DataFrame, optional): The test data.
    X (np.array or pd.DataFrame, optional): The data to scale during manual testing.
    scaler_path (str, optional): Path to save or load the scaler. Defaults to '../models/final_scaler.pkl'.

    Returns:
    tuple: Scaled data (X_train_scaled, X_test_scaled) if using train/test split, or scaled X if in manual test mode.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(current_dir, scaler_path)
    scaler_path = os.path.abspath(data_file_path)
    
    if X_train is not None and X_test is not None:
        # Training phase with train/test split
        scaler = MinMaxScaler()
        
        # Fit the scaler on the training data and transform both train and test sets
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save the fitted scaler for future use (e.g., manual testing)
        joblib.dump(scaler, scaler_path)
        
        return X_train_scaled, X_test_scaled
    
    elif X is not None:
        # Manual testing phase (no train/test split)
        # Load the pre-fitted scaler
        scaler = joblib.load(scaler_path)
        
        # Transform the input data using the loaded scaler
        X_scaled = scaler.transform(X)
        
        return X_scaled
    
    else:
        raise ValueError("Either provide X_train and X_test for training or X for manual testing.")
    


def reshape_for_lstm(X, y=None, n_timesteps=5):
    """
    Reshapes the input data (X) for LSTM training or testing.
    
    Parameters:
    X (np.array): The input feature array to be reshaped.
    y (np.array, optional): The target array to be reshaped (if provided).
    n_timesteps (int, optional): The number of timesteps to use in the LSTM. Default is 5.
    
    Returns:
    tuple: A tuple containing the reshaped X and (optionally) y arrays for LSTM.
    If y is not provided, only X is returned.
    """
    
    # Determine the number of features
    n_features = X.shape[1]
    
    # Ensure X is divisible by n_timesteps
    n_samples = len(X) // n_timesteps

    X = X[:n_samples * n_timesteps]
    
    # Reshape X to 3D (samples, timesteps, features) for LSTM
    X_lstm = np.reshape(X, (n_samples, n_timesteps, n_features))
    
    if y is not None:
        # Ensure y matches the reshaped X (only if y is provided)
        y = y[:n_samples * n_timesteps:n_timesteps]
        return X_lstm, y
    
    return X_lstm


def detect_outliers_z_score(df, column):
    threshold = 2.5
    mean = np.mean(df[column])
    std = np.std(df[column])
    z_scores = (df[column] - mean) / std
    return df[np.abs(z_scores) > threshold]


def remove_outliers(df):
    outliers = detect_outliers_z_score(df, 'growth_score')
    return df[~df.index.isin(outliers.index)]


def reduce_dimentionality_pca(df, pca_path = '../models/pca_model.pkl'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(current_dir, pca_path)
    pca_path = os.path.abspath(data_file_path)
    pca = joblib.load(pca_path)
    return pca.transform(df)


def forecast_growth(model, initial_data, n_steps, scaler, timesteps=5, n_features=133):
    """
    Forecast future growth score based on the trained LSTM model.

    Parameters:
    - model: The trained LSTM model.
    - initial_data (np.ndarray): The initial sequence of data to start forecasting.
    - n_steps (int): The number of steps (e.g., months) to forecast.
    - scaler: Scaler used for scaling the data (for inverse transforming).
    - timesteps (int): The number of time steps used in the LSTM.

    Returns:
    - forecasted_values (np.ndarray): The forecasted growth scores.
    """

    forecasted_values = []
    input_seq = initial_data[-timesteps:]  # Start with the last available data
    
    for _ in range(n_steps):
        # Prepare the input by reshaping (3D: 1 × timesteps × features)
        input_seq_reshaped = input_seq.reshape(1, timesteps, n_features)

        # Predict the next step (3D output)
        next_pred_scaled = model.predict(input_seq_reshaped)

        # Inverse transform the prediction to original scale (2D output)
        next_pred = scaler.inverse_transform(next_pred_scaled)

        # Append the forecasted value
        forecasted_values.append(next_pred[0, 0])  # Append the forecasted value

        # Create a new input sequence with the predicted value as the last feature
        next_input = np.zeros((1, n_features))  # Initialize an array with the same number of features as input_seq
        
        # Assign the predicted value to the relevant feature
        next_input[0, -1] = next_pred[0, 0]  # Assuming growth score is the last feature
        
        # Update the input sequence by shifting and appending the new prediction
        input_seq = np.vstack([input_seq[1:], next_input])  # Shift and append
    
    return np.array(forecasted_values)
    

def plot_historical_and_forecasted_growth(test_df, forecasted_values, n_forecast_steps, freq='M'):
    """
    Plots historical and forecasted growth scores.

    Parameters:
    - test_df (pd.DataFrame): DataFrame containing historical data with 'growth_score' and datetime index.
    - forecasted_values (array-like): Forecasted growth scores to be plotted.
    - n_forecast_steps (int): Number of forecasted steps to be plotted.
    - freq (str): Frequency of the forecasted dates (e.g., 'M' for monthly). Default is 'M'.

    Returns:
    - None
    """
    # Ensure the index is in datetime format (if not already)
    test_df.index = pd.to_datetime(test_df.index)

    # Historical data: Plot the actual release dates with the growth score
    historical_growth = test_df['growth_score']

    # Generate future dates for the forecasted values based on the release_date index
    # If you are forecasting 'n_steps' into the future (assuming monthly intervals)
    forecasted_dates = pd.date_range(start=test_df.index[-1], periods=n_forecast_steps + 1, freq='M')[1:]

    st.title("Historical and Forecasted Growth Score")

    # Create a figure and axis with Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot historical data
    ax.plot(test_df.index, historical_growth, label='Historical Growth', color='blue')

    # Plot forecasted data
    ax.plot(forecasted_dates, forecasted_values, label='Forecasted Growth', color='orange', linestyle='--')

    # Add labels and title
    ax.set_xlabel('Release Date')
    ax.set_ylabel('Growth Score')
    ax.set_title('Historical and Forecasted Growth Score')

    # Add legend
    ax.legend()

    # Rotate x-axis labels
    plt.xticks(rotation=45)

    # Use Streamlit to display the Matplotlib figure
    st.pyplot(fig)


def get_single_repo_data(url):
    org_name, repo_name = get_org_and_repo_from_url(url)
    token = GITHUB_TOKEN
    headers = {}
    headers['Accept'] = 'application/vnd.github.v3.star+json'
    if token:
        headers['Authorization'] = f'token {token}'
        
    repo_api = f'https://api.github.com/repos/{org_name}/{repo_name}'
    response = requests.get(repo_api, headers=headers)
    
    if response.status_code == 200:
        repo_object = response.json()
        filtered_data = fetch_repo_data(org_name, repo_object, headers)
        return pd.DataFrame(filtered_data)
        
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None


def get_org_and_repo_from_url(url):
    """
    Extracts the organization name and repository name from a GitHub URL.
    Parameters:
    - url (str): The GitHub URL.
    Returns:
    - tuple: (org_name, repo_name) extracted from the URL.
    """
    # Split the URL and get the organization and repository names
    parts = url.rstrip('/').split('/')  # Remove trailing slash if present and split
    if len(parts) >= 5 and parts[2] == 'github.com':
        org_name = parts[3]
        repo_name = parts[4]
        return org_name, repo_name
    else:
        raise ValueError("Invalid GitHub URL format")



def convert_to_naive_or_aware(df, column, make_naive=True):
    """
    Converts a datetime column to be either tz-naive or tz-aware.

    Parameters:
    df (pd.DataFrame): The dataframe containing the datetime column.
    column (str): The name of the column to convert.
    make_naive (bool): If True, converts to tz-naive. If False, converts to tz-aware (UTC).

    Returns:
    pd.DataFrame: Dataframe with the updated datetime column.
    """
    if make_naive:
        df[column] = df[column].apply(lambda x: x.tz_localize(None) if pd.notna(x) and x.tzinfo is not None else x)
    else:
        df[column] = df[column].apply(lambda x: x.tz_localize('UTC') if pd.notna(x) and x.tzinfo is None else x)
    return df



def fetch_repo_data(org_name, repo_object, headers):
    repo_name = repo_object['name']
    pr_query = f'type:pr repo:{org_name}/{repo_name}'
    pr_params = {'q': pr_query, 'per_page': 1}

    issues_url = 'https://api.github.com/search/issues'
    pr_response = requests.get(issues_url, headers=headers, params=pr_params)
    
    if pr_response.status_code == 200:
        num_pulls = pr_response.json().get('total_count', 0)
    else:
        print(f"Error fetching pull requests: {pr_response.status_code} - {pr_response.text}")
        num_pulls = None

    release_date, release_tag = get_release_dates(headers, org_name, repo_name)
    topics = get_topics(headers, org_name, repo_name)

    
    num_releases = len(release_date)
    num_stars = repo_object['stargazers_count']
    num_forks = repo_object['forks_count']
    num_watchers = repo_object['watchers_count']


    repo_data = {
        'org_name': org_name,
        'repo_name': repo_name,
        'description': repo_object['description'],
        'repo_url': repo_object['html_url'],
        'topics': topics,
        'creation_date': repo_object['created_at'],
        'update_date': repo_object['updated_at'],
        'release_tag': release_tag,
        'release_date': release_date,
        'num_releases': num_releases,
        'num_open_issues': repo_object['open_issues_count'],
        'num_pull_requests': num_pulls,
        'num_stars': num_stars,
        'num_forks': num_forks,
        'num_watchers': num_watchers
    }
    return repo_data
    

import time
def fetch_github_data(headers, query, pages=1, per_page=50):
    data = []
    repos_url = 'https://api.github.com/search/repositories'
    params = {'q': query, 'sort': 'stars', 'order': 'desc', 'per_page': per_page, 'page': pages}

    response = requests.get(repos_url, headers=headers, params=params)

    if response.status_code == 200:
        repos = response.json().get('items', [])
        
        for i, repo in enumerate(repos):
            org_name = repo['full_name'].split('/')[0]
            repo_data = fetch_repo_data(org_name, repo, headers)
            data.append(repo_data)
            print(f"Fetched {i + 1} repositories")
            
            # Add a small delay to prevent hitting the API rate limit
            time.sleep(1)
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    return data



def get_release_dates(headers, owner, repo):
    """Fetch release tags and dates for a specific repository."""
    release_url = 'https://api.github.com/repos/{owner}/{repo}/releases'
    url = release_url.format(owner=owner, repo=repo)
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        releases = response.json()
        release_date = [release['published_at'] for release in releases]
        release_tag = [release['tag_name'] for release in releases]
        return release_date, release_tag
    else:
        print(f"Error fetching releases for {owner}/{repo}: {response.status_code} - {response.text}")
        return []


    
def get_topics(headers, owner, repo):
    """Fetch topics for a specific repository."""
    topics_url = 'https://api.github.com/repos/{owner}/{repo}/topics'
    url = topics_url.format(owner=owner, repo=repo)
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        topics_data = response.json()
        topics = [topic for topic in topics_data['names']]
        topics_string = ', '.join(topics)
        return topics_string
    else:
        print(f"Error fetching topics for {owner}/{repo}: {response.status_code} - {response.text}")
        return []