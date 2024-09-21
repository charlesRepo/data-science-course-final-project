# CLI command: python3 -m streamlit run ./src/test.py
import streamlit as st # pip install streamlit
import pandas as pd
import numpy as np
import sys
import os

module_path = os.path.abspath(os.path.join('..', 'src'))
if module_path not in sys.path:
    sys.path.append(module_path)
from functions import add_growth_score_based_on_main_features 
from functions import add_repo_age_days_col
from functions import add_days_since_last_release_col
from functions import convert_topics_to_embeddings
from functions import indexify_release_dates
from functions import add_lag_features_based_on_target
from functions import scale_final_data, remove_outliers
from functions import reduce_dimentionality_pca
from functions import forecast_growth
from functions import plot_historical_and_forecasted_growth
from functions import get_repo_data
from functions import remove_unwanted_features

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import joblib
import matplotlib.pyplot as plt


st.subheader(f'Ai Repos Growth')

url = st.text_input("Enter a URL", "")


ok = st.button("Check Repo")
if ok and url:
    st.write(f"Processing the URL: {url}")
    #'https://github.com/Significant-Gravitas/AutoGPT'
    test_df = get_repo_data(url)
    test_df = add_growth_score_based_on_main_features(test_df, train=False)
    actual_growth_score = test_df['growth_score']

    # output description
    # number of stars
    # number of forks
    # list of topics


    test_df = add_repo_age_days_col(test_df)
    test_df = add_days_since_last_release_col(test_df)
    test_df = convert_topics_to_embeddings(test_df)
    test_df = remove_outliers(test_df)
    test_df = indexify_release_dates(test_df)
    test_df = add_lag_features_based_on_target(test_df)
    X = remove_unwanted_features(test_df)
    test_df_scaled = scale_final_data(X=X.values)
    test_df_scaled = reduce_dimentionality_pca(test_df_scaled)

    n_features = test_df_scaled.shape[1] 
    n_timesteps = 4
    n_forecast_steps = 24 

    test_generator = TimeseriesGenerator(test_df_scaled, np.zeros(len(test_df_scaled)), length=n_timesteps, batch_size=1)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(current_dir, '../models/best_rnn_model.keras')
    model_path = os.path.abspath(data_file_path)
    model = load_model(model_path)
    y_pred = model.predict(test_generator)
    st.text(f'predicted growth score: {y_pred[0][0]}')
    st.text(f'actual growth score: {actual_growth_score[1]}')
    st.text(f'prediction diff: {y_pred[0][0] - actual_growth_score[1]}')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(current_dir, '../data/final_target_scaler.pkl')
    scaler_path = os.path.abspath(data_file_path)
    scaler_y = joblib.load(scaler_path)

    # Perform the forecasting
    forecasted_values = forecast_growth(
        model=model, 
        initial_data=test_df_scaled, 
        n_steps=n_forecast_steps, 
        scaler=scaler_y, 
        timesteps=5,
        n_features=n_features
    )

    plot_historical_and_forecasted_growth(test_df, forecasted_values, n_forecast_steps,)

    

