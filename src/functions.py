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
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import streamlit as st # pip install streamlit
import time




def distribute_values_across_releases(release_dates, total_value):
    """
    Distribute the total value across a series of release dates incrementally.
    
    Parameters:
    - release_dates (list): A list of release dates for a repository.
    - total_value (int): The total value (e.g., stars, forks) to be distributed across the releases.

    Returns:
    - values (list): A list of integers representing the distributed values across the release dates.
    """

    # Determine the number of releases
    n_releases = len(release_dates)

    # Calculate the step value to incrementally distribute the total value across releases.
    # If there's only one release, the step is equal to the total value.
    step = total_value / (n_releases - 1) if n_releases > 1 else total_value

    # Create the values starting from 0 and incrementing by the calculated step
    values = [round(step * i) for i in range(n_releases)]

    # Ensure the last value matches the total count
    values[-1] = total_value  

    return values

def distribute_features_across_releases(df, features):
    """
    Distributes specified feature values incrementally across release dates for each repository.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing repository data. 
                        It should include columns for 'org_name', 'repo_name', and 'release_date'.
    - features (list): A list of feature column names (e.g., 'num_stars', 'num_forks') to be distributed across release dates.

    Returns:
    - df (pd.DataFrame): The updated DataFrame with distributed feature values.
    """

    # Group the DataFrame by organization name and repository name
    for repo_index, repo_group in df.groupby(['org_name', 'repo_name']):
        # Get release dates for the current repository
        release_dates = repo_group['release_date']
        
        # Distribute each feature column across the release dates
        for feature in features:
            # Get the total value of the feature for the current repository
            total_value = repo_group[feature].iloc[0] 
            # Distribute the values
            distributed_values = distribute_values_across_releases(release_dates, total_value)
            # Update the original DataFrame with the distributed values for the current feature  
            df.loc[release_dates.index, feature] = distributed_values 
    
    return df



def add_time_based_noise(series, factor=0.02, seed=None):
    """
    Adds time-based noise to a pandas Series to simulate temporal variations in the data.

    Parameters:
    - series (pd.Series): The input Series to which noise will be added.
    - factor (float): The scaling factor for the random noise component (default is 0.02).
    - seed (int): A random seed for reproducibility (default is None).

    Returns:
    - pd.Series: The original Series with added time-based noise.
    """
    # Set the random seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)
    # Generate a time array representing each point in the series
    time = np.arange(len(series))

    # Create a temporal variation using a sine wave pattern
    # Adjusting the divisor in (time / 5) changes the frequency of the sine wave
    temporal_variation = np.sin(time / 5)

    # Generate random noise using a normal distribution centered at 0
    # 'scale' determines the standard deviation of the noise
    random_noise = np.random.normal(loc=0, scale=factor, size=len(series))
    
    # Combine the sine wave (temporal variation) with random noise
    time_noise = temporal_variation + random_noise

    # Add the generated noise to the original series
    # The noise is scaled by the mean of the series to maintain relative magnitude
    return series + series.mean() * time_noise

def apply_time_based_noise(df, features):
    """
    Apply time-based noise to specified feature columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the features.
    features (list): List of feature column names to which noise should be applied.

    Returns:
    pd.DataFrame: The DataFrame with noise added to the specified features.
    """
    for feature in features:
        df[feature] = add_time_based_noise(df[feature])
    return df


def add_proportional_noise(series, factor=0.05, seed=None, min_value=1):
    """
    Adds proportional noise to a pandas Series. The noise is proportional to the
    original values in the Series, ensuring that higher values get larger variations.

    Parameters:
    - series (pd.Series): The input Series to which noise will be added.
    - factor (float): The scaling factor for the proportional noise (default is 0.05).
    - seed (int): A random seed for reproducibility (default is None).
    - min_value (int): The minimum value to ensure that all elements are at least this value (default is 1).

    Returns:
    - pd.Series: The Series with added proportional noise, rounded to integers and bounded by min_value.
    """
    # Set the random seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)

    # Generate random noise using a normal distribution centered at 0    
    noise = np.random.normal(loc=0, scale=1, size=len(series))

    # Add proportional noise to the series
    # Each element in the series is adjusted by a factor of its own value
    noisy_series = series * (1 + factor * noise)
    
    # Round the values to the nearest integer and ensure they are at least min_value
    noisy_series = np.maximum(noisy_series.round().astype(int), min_value)

    return noisy_series


def apply_proportional_noise(df, features):
    """
    Apply proportional noise to specified feature columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the features.
    features (list): List of feature column names to which noise should be applied.

    Returns:
    pd.DataFrame: The DataFrame with proportional noise added to the specified features.
    """
    for feature in features:
        df[feature] = add_proportional_noise(df[feature])
    return df


def remove_first_augmented_rows(df):
    """
    Remove rows from the DataFrame where the 'num_releases' column has an initial value of zero.
    This is typically used to remove augmented or placeholder rows in the data.

    Parameters:
    - df (pd.DataFrame): The input DataFrame from which to remove the rows.

    Returns:
    - pd.DataFrame: The filtered DataFrame with rows removed and index reset.
    """

    # Filter out rows where 'num_releases' is zero, assuming these rows are placeholders or augmented
    df = df[df['num_releases'] != 0]
    
    # Reset the index after filtering, drop the old index to avoid retaining it as a new column
    df.reset_index(drop=True, inplace=True)
    return df



def add_growth_score_based_on_main_features(df, train=True):
    """
    Adds a 'growth_score' column to the DataFrame based on a combination of key features.
    The function can be used in training mode to fit and save a scaler or in inference mode to load and use an existing scaler.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing feature columns.
    - train (bool): If True, the function will fit and save a new scaler based on the input data. 
                    If False, it will load and use a pre-existing scaler.

    Returns:
    - pd.DataFrame: The DataFrame with an added 'growth_score' column.
    """
    # Define the feature columns to be used for calculating the growth score
    feature_columns = ['num_stars', 'num_forks', 'num_watchers', 'num_pull_requests', 'num_releases', 'num_open_issues']
    
    # Construct the path to the scaler file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(current_dir, '../models/growth_score_training_scaler.pkl')
    scaler_path = os.path.abspath(data_file_path)

    # Extract the relevant features for scaling from the DataFrame
    features = df[feature_columns]
    
    if train == True:
        # If in training mode, fit a new MinMaxScaler on the features
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)
        # Save the fitted scaler after training
        joblib.dump(scaler, scaler_path)

    else:
        # If not in training mode, load the existing scaler from the file
        growth_score_training_scaler = joblib.load(scaler_path)

        # Use the provided scaler
        scaled_features = growth_score_training_scaler.transform(features)

    # Calculate the growth_score as the sum of the scaled features
    df['growth_score'] = scaled_features.sum(axis=1)
    
    return df


def add_repo_age_days_col(df):
    """
    Adds a 'repo_age_days' column to the DataFrame, representing the age of the repository in days.
    The 'repo_age_days' is calculated based on the difference between the current date and the 'creation_date' of the repository.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing a 'creation_date' column.

    Returns:
    - pd.DataFrame: The modified DataFrame with a new 'repo_age_days' column and the 'creation_date' column removed.
    """
    # Convert the 'creation_date' column to datetime format, ensuring correct handling of dates
    df['creation_date'] = pd.to_datetime(df['creation_date'])
    # Get the current date and time in UTC
    now = pd.Timestamp.now(tz='UTC')
    # Calculate the repository age in days by finding the difference between the current date and the 'creation_date'
    df['repo_age_days'] = (now - df['creation_date']).dt.days
    # Remove the 'creation_date' column as it's no longer needed
    df.drop(columns=['creation_date'], inplace=True)
    return df


def add_days_since_last_release_col(df):
    """
    Adds a 'days_since_last_release' column to the DataFrame, representing the number of days since the last release date.
    The 'days_since_last_release' is calculated based on the difference between the current date and the 'release_date'.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing a 'release_date' column.

    Returns:
    - pd.DataFrame: The modified DataFrame with a new 'days_since_last_release' column.
    """
    # Convert the 'release_date' column to datetime format to ensure correct date operations
    df['release_date'] = pd.to_datetime(df['release_date'])
    # Get the current date and time in UTC
    now = pd.Timestamp.now(tz='UTC')
    # Calculate the number of days since the last release date by finding the difference between the current date and the 'release_date'
    df['days_since_last_release'] = (now - df['release_date']).dt.days
    return df


def remove_stop_words_from_topics(df):
    """
    Removes stopwords from the 'topics' column in the DataFrame using the NLTK English stopword list.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing a 'topics' column with text data.

    Returns:
    - pd.DataFrame: The modified DataFrame with stopwords removed from the 'topics' column.
    """
    # Download the NLTK stopwords list if not already available
    nltk.download('stopwords')

    # Create a set of English stopwords from the NLTK library
    stop_words = set(stopwords.words('english'))
    def remove_stopwords(text):
        # Split the text into words, remove stopwords, and rejoin the remaining words into a single string
        return ' '.join([word for word in str(text).split() if word.lower() not in stop_words])
    # Apply the remove_stopwords function to each entry in the 'topics' column
    df['topics'] = [remove_stopwords(topics) for topics in df.topics]
    
    return df


def convert_topics_to_embeddings(df):
    """
    Converts the 'topics' column of a DataFrame into sentence embeddings using a pre-trained SentenceTransformer model.
    The embeddings are then spread across multiple columns in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing a 'topics' column with text data.

    Returns:
    - pd.DataFrame: The modified DataFrame with the 'topics' column replaced by multiple columns representing the embeddings.
    """
    # Apply stop words removal function
    df = remove_stop_words_from_topics(df)
    
    # Load the SentenceTransformer model for generating sentence embeddings
    sentence_embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Apply the SentenceTransformer model to convert each topic to an embedding
    # The resulting 'topic_embedding' column contains a list of embeddings for each topic
    df['topic_embedding'] = df['topics'].apply(lambda topic: sentence_embeddings_model.encode(topic))

    # Spread embeddings across multiple columns
    topic_embedding_df = pd.DataFrame(df['topic_embedding'].tolist(), index=df.index)
    # Concatenate the original DataFrame with the new embedding columns
    df = pd.concat([df, topic_embedding_df], axis=1)
    # Drop the 'topic_embedding' and original 'topics' columns as they are no longer needed
    df.drop(columns=['topic_embedding', 'topics'], inplace=True)
    
    return df


def indexify_release_dates(df):
    df = df.set_index('release_date')
    df = df.sort_index()
    return df


def get_num_of_releases(df):
    return df['num_releases'].astype(int).unique()[0]


def add_lag_features_based_on_target(df, num = 4):
    """
    Adds lag features based on the target variable 'growth_score' to the DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the 'growth_score' column.
    - num (int): The number of lag features to add (default is 4).
    
    Returns:
    - pd.DataFrame: The DataFrame with added lag features, and rows with NaN values removed.
    """
    # Iterate through the range from 1 to 'num' (inclusive) to create multiple lag features
    for i in range(1, num+1):
        # Create a lagged version of the 'growth_score' column with a lag of 'i' and name it accordingly
        df[f'growth_score_lag_{i}'] = df['growth_score'].shift(i)

    # Apply log transformation to the 'growth_score' column to reduce skewness, by reducing outliers and improving residuals.
    # df['growth_score'] = np.log1p(df['growth_score'])

    # Remove rows with NaN values, which are introduced by the shift operation
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
    """
    Detects outliers in a given column of a DataFrame using the Z-score method.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column (str): The column name for which to detect outliers.

    Returns:
    - pd.DataFrame: A DataFrame containing only the rows where outliers are detected in the specified column.
    """

    # Define the Z-score threshold for outlier detection
    threshold = 3
    mean = np.mean(df[column])
    std = np.std(df[column])
    # Calculate the Z-scores for each value in the column
    z_scores = (df[column] - mean) / std
    return df[np.abs(z_scores) > threshold]


def remove_outliers(df):
    outliers = detect_outliers_z_score(df, 'growth_score')
    return df[~df.index.isin(outliers.index)]


def reduce_dimentionality_pca(df, pca_path = '../models/pca_model.pkl'):
    """
    Applies a pre-trained PCA model to reduce the dimensionality of the input DataFrame.

    Parameters:
    - df (pd.DataFrame or np.ndarray): The input data to be transformed using PCA.
    - pca_path (str): The path to the pre-trained PCA model file. Defaults to '../models/pca_model.pkl'.

    Returns:
    - np.ndarray: The transformed data with reduced dimensionality.
    """
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

        # For LSTM model with return_sequences=True
        # next_pred_scaled_2d = next_pred_scaled[:, -1, :] # transform to 2D array
        # next_pred = scaler.inverse_transform(next_pred_scaled_2d)

        # For GRU and LSTM model with return_sequences=False
        next_pred = scaler.inverse_transform(next_pred_scaled)

        # Append the forecasted value
        forecasted_values.append(next_pred[0, 0])  # Assuming growth score is the first feature
        
        # # Create a new input sequence with the predicted value
        next_input = input_seq[-1].copy()  # Copy last row of input sequence
        next_input[-1] = next_pred[0, 0] 

        # # Update the input sequence by shifting and appending the new prediction
        input_seq = np.concatenate([input_seq[1:], next_input.reshape(1, -1)], axis=0)  # Shift and append
    
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

    st.header("Historical and Forecasted Growth Score")

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


def get_single_repo_data(url, token):
    """
    Fetches and returns data for a single GitHub repository.

    Parameters:
    - url (str): The GitHub repository URL in the format 'https://github.com/org_name/repo_name'.
    - token (str): A GitHub API access token to authenticate the request. If None, no token is used.

    Returns:
    - pd.DataFrame: A DataFrame containing the repository data, or None if an error occurs.
    """

    org_name, repo_name = get_org_and_repo_from_url(url)
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
    """
    Fetches additional data for a given GitHub repository, including pull requests, releases, topics, and other metrics.

    Parameters:
    - org_name (str): The name of the organization or user owning the repository.
    - repo_object (dict): The repository data returned by the GitHub API.
    - headers (dict): Headers for the GitHub API request, including authorization and other parameters.

    Returns:
    - dict: A dictionary containing detailed information about the repository.
    """
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
    

def fetch_github_data(headers, query, pages=1, per_page=50):
    """
    Fetches data for GitHub repositories based on a search query.

    Parameters:
    - headers (dict): Headers for the GitHub API request, including authorization.
    - query (str): The search query for repositories.
    - pages (int): The number of pages to retrieve from the search results. Defaults to 1.
    - per_page (int): The number of repositories to retrieve per page. Defaults to 50.

    Returns:
    - list: A list of dictionaries containing detailed information about each repository.
    """
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
    """
    Fetch release tags and dates for a specific GitHub repository.

    Parameters:
    - headers (dict): Headers for the GitHub API request, including authorization.
    - owner (str): The owner of the GitHub repository.
    - repo (str): The name of the GitHub repository.

    Returns:
    - tuple: Two lists containing release dates and release tags, respectively.
    """
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
    """
    Fetch the topics associated with a specific GitHub repository.

    Parameters:
    - headers (dict): Headers for the GitHub API request, including authorization.
    - owner (str): The owner of the GitHub repository.
    - repo (str): The name of the GitHub repository.

    Returns:
    - str: A comma-separated string of repository topics.
    """
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
    



# Main App Functions 

def get_repo_data(url):

    st.write(f"Processing the URL: {url}")
    #'https://github.com/Significant-Gravitas/AutoGPT'
    with st.spinner(text='In progress'):
        time.sleep(3)
        message_placeholder = st.empty()
        message_placeholder.success("Repo's data fetched!")
        time.sleep(3)
        message_placeholder.empty()

    test_df = get_single_repo_data(url, st.secrets.GITHUB_TOKEN)
    test_df['release_date'] = pd.to_datetime(test_df['release_date'])
    latest_release_year = test_df['release_date'].max().year

    if (test_df['num_releases'] <= 15).any() or test_df.empty:
        st.warning('Please provide a link to a repository that has more than 15 releases.')
        return None, None, None, None 
    elif latest_release_year < 2024 :
        st.warning('Please provide a link to a repository with up-to-date releases later thatn 2023.')
        return None, None, None, None 
    else:
        n_lag_features = 5
        n_timesteps = 5
        n_forecast_steps = 12 

        test_df = test_df.sort_values(by='release_date', ascending=True).reset_index(drop=True)
        test_df = distribute_features_across_releases(test_df, ['num_stars', 'num_forks', 'num_watchers', 'num_pull_requests', 'num_open_issues', 'num_releases'])
        test_df = remove_first_augmented_rows(test_df)
        test_df = add_growth_score_based_on_main_features(test_df, train=False)        
        test_df = apply_time_based_noise(test_df, ['num_stars', 'num_forks', 'num_watchers', 'num_pull_requests', 'num_open_issues'])
        test_df = apply_proportional_noise(test_df, ['num_stars', 'num_forks', 'num_watchers', 'num_pull_requests', 'num_open_issues'])
        test_df = add_repo_age_days_col(test_df)
        test_df = add_days_since_last_release_col(test_df)
        test_df = convert_topics_to_embeddings(test_df)
        test_df = remove_outliers(test_df)
        test_df = indexify_release_dates(test_df)
        test_df = add_lag_features_based_on_target(test_df, num=n_lag_features)
        X = remove_unwanted_features(test_df)
        test_df_scaled = scale_final_data(X=X.values)
        test_df_scaled_pca = reduce_dimentionality_pca(test_df_scaled)

        # Define the number of features
        n_features = test_df_scaled_pca.shape[1] 
        

        test_generator = TimeseriesGenerator(test_df_scaled_pca, np.zeros(len(test_df_scaled_pca)), length=n_timesteps, batch_size=1)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_file_path = os.path.join(current_dir, '../models/best_rnn_model.keras')
        model_path = os.path.abspath(data_file_path)
        model = load_model(model_path)
        y_pred = model.predict(test_generator)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_file_path = os.path.join(current_dir, '../models/final_target_scaler.pkl')
        scaler_path = os.path.abspath(data_file_path)
        scaler_y = joblib.load(scaler_path)

        # Perform the forecasting
        forecasted_values = forecast_growth(
            model=model, 
            initial_data=test_df_scaled_pca, 
            n_steps=n_forecast_steps, 
            scaler=scaler_y, 
            timesteps=n_timesteps,
            n_features=n_features
        )

        return test_df, y_pred, forecasted_values, n_forecast_steps
        

def display_repo_data(test_df, y_pred, forecasted_values, n_forecast_steps):

    if test_df is None or y_pred is None or forecasted_values is None or n_forecast_steps is None:
        return  # Exit the function early if any input is None
    
    st.subheader('Repository info')
    st.write('Organization name: ', test_df.iloc[0]['org_name'])
    st.write('Repo name: ', test_df.iloc[0]['repo_name'])
    st.write('Repo description: ', test_df.iloc[0]['description'])

    df_num_stars_sorted = test_df.sort_values(by=['repo_name', 'num_stars'], ascending=[True, True])
    last_rows_num_stars = df_num_stars_sorted.drop_duplicates(subset='repo_name', keep='last')
    

    df_num_forks_sorted = test_df.sort_values(by=['repo_name', 'num_forks'], ascending=[True, True])
    last_rows_num_forks = df_num_forks_sorted.drop_duplicates(subset='repo_name', keep='last')
    

    df_num_watchers_sorted = test_df.sort_values(by=['repo_name', 'num_watchers'], ascending=[True, True])
    last_rows_num_watchers = df_num_watchers_sorted.drop_duplicates(subset='repo_name', keep='last')
    

    df_num_prs_sorted = test_df.sort_values(by=['repo_name', 'num_pull_requests'], ascending=[True, True])
    last_rows_num_prs = df_num_prs_sorted.drop_duplicates(subset='repo_name', keep='last')
    

    df_num_oprs_sorted = test_df.sort_values(by=['repo_name', 'num_open_issues'], ascending=[True, True])
    last_rows_num_oprs = df_num_oprs_sorted.drop_duplicates(subset='repo_name', keep='last')
    

    df_num_releases_sorted = test_df.sort_values(by=['repo_name', 'num_releases'], ascending=[True, True])
    last_rows_num_releases = df_num_releases_sorted.drop_duplicates(subset='repo_name', keep='last')
    

    data_info = {
    'Metric': ['Stars', 'Forks', 'Watchers', 'Pull Requests', 'Open Pull Requests', 'Number of Releases'],
    'Value': [
        last_rows_num_stars.iloc[0]['num_stars'],
        last_rows_num_forks.iloc[0]['num_forks'],
        last_rows_num_watchers.iloc[0]['num_watchers'],
        last_rows_num_prs.iloc[0]['num_pull_requests'],
        last_rows_num_oprs.iloc[0]['num_open_issues'],
        last_rows_num_releases.iloc[0]['num_releases']
        ]
    }   
    df_info = pd.DataFrame(data_info)
    st.table(df_info)


    df_num_gs_sorted = test_df.sort_values(by=['repo_name', 'growth_score'], ascending=[True, True])
    last_rows_num_gs = df_num_gs_sorted.drop_duplicates(subset='repo_name', keep='last')
    actual_growth_score = last_rows_num_gs.iloc[0]['growth_score']
    # st.write('Growth Score: ', last_rows_num_gs.iloc[0]['growth_score'])
    st.markdown(f"<h3><b>Growth Score:</b> {last_rows_num_gs.iloc[0]['growth_score']:.2f}</h3>", unsafe_allow_html=True)
    
    st.info("The growth score is a metric that quantifies the development and popularity of a GitHub repository over time, considering factors like stars, forks, pull requests, and issues.")


    prediction_diff = y_pred[0][0] - actual_growth_score
    data_pred = {
    'Metric': ['Predicted Growth Score', 'Actual Growth Score', 'Prediction Difference'],
    'Value': [y_pred[0][0], actual_growth_score, prediction_diff]
    }
    df_pred = pd.DataFrame(data_pred)
    st.table(df_pred)

    plot_historical_and_forecasted_growth(test_df, forecasted_values, n_forecast_steps,)
    