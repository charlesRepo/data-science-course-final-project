# CLI command: python3 -m streamlit run ./src/test.py
import streamlit as st # pip install streamlit
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import joblib


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
from functions import get_single_repo_data
from functions import remove_unwanted_features
from functions import distribute_features_across_releases
from functions import apply_time_based_noise
from functions import apply_proportional_noise
from functions import remove_first_augmented_rows
import time


st.subheader(f'Ai-Related Repository Growth')

url = st.text_input("Enter a Github Repository URL", "")


ok = st.button("Fetch Repo data")
if ok and url:
    st.write(f"Processing the URL: {url}")
    #'https://github.com/Significant-Gravitas/AutoGPT'
    with st.spinner(text='In progress'):
        time.sleep(3)
        st.success("Repo's data fetched!")

    test_df = get_single_repo_data(url, st.secrets.GITHUB_TOKEN)

    if (test_df['num_releases'] <= 12).any() or test_df.empty:
    # Do something, like returning or filtering the DataFrame
        st.warning('Please provide a link to a repository which has more than 12 releases.')
    else:
        st.subheader('Repository info')
        st.write('Organization name: ', test_df.iloc[0]['org_name'])
        st.write('Repo name: ', test_df.iloc[0]['repo_name'])
        st.write('Repo description: ', test_df.iloc[0]['description'])

        df_num_stars_sorted = test_df.sort_values(by=['repo_name', 'num_stars'], ascending=[True, True])
        last_rows_num_stars = df_num_stars_sorted.drop_duplicates(subset='repo_name', keep='last')
        st.write('Stars: ', last_rows_num_stars.iloc[0]['num_stars'])

        df_num_forks_sorted = test_df.sort_values(by=['repo_name', 'num_forks'], ascending=[True, True])
        last_rows_num_forks = df_num_forks_sorted.drop_duplicates(subset='repo_name', keep='last')
        st.write('Forks: ', last_rows_num_forks.iloc[0]['num_forks'])

        df_num_watchers_sorted = test_df.sort_values(by=['repo_name', 'num_watchers'], ascending=[True, True])
        last_rows_num_watchers = df_num_watchers_sorted.drop_duplicates(subset='repo_name', keep='last')
        st.write('Watchers: ', last_rows_num_watchers.iloc[0]['num_watchers'])

        df_num_prs_sorted = test_df.sort_values(by=['repo_name', 'num_pull_requests'], ascending=[True, True])
        last_rows_num_prs = df_num_prs_sorted.drop_duplicates(subset='repo_name', keep='last')
        st.write('Pull Requests: ', last_rows_num_prs.iloc[0]['num_pull_requests'])

        df_num_oprs_sorted = test_df.sort_values(by=['repo_name', 'num_open_issues'], ascending=[True, True])
        last_rows_num_oprs = df_num_oprs_sorted.drop_duplicates(subset='repo_name', keep='last')
        st.write('Open Pull Requests: ', last_rows_num_oprs.iloc[0]['num_open_issues'])

        df_num_releases_sorted = test_df.sort_values(by=['repo_name', 'num_releases'], ascending=[True, True])
        last_rows_num_releases = df_num_releases_sorted.drop_duplicates(subset='repo_name', keep='last')
        st.write('Number of Releases: ', last_rows_num_releases.iloc[0]['num_releases'])

        test_df = test_df.sort_values(by='release_date', ascending=True).reset_index(drop=True)
        test_df = distribute_features_across_releases(test_df, ['num_stars', 'num_forks', 'num_watchers', 'num_pull_requests', 'num_open_issues', 'num_releases'])
        test_df = remove_first_augmented_rows(test_df)
        test_df = add_growth_score_based_on_main_features(test_df, train=False)        
        # test_df = apply_time_based_noise(test_df, ['num_stars', 'num_forks', 'num_watchers', 'num_pull_requests', 'num_open_issues'])
        # test_df = apply_proportional_noise(test_df, ['num_stars', 'num_forks', 'num_watchers', 'num_pull_requests', 'num_open_issues'])

        df_num_gs_sorted = test_df.sort_values(by=['repo_name', 'growth_score'], ascending=[True, True])
        last_rows_num_gs = df_num_gs_sorted.drop_duplicates(subset='repo_name', keep='last')
        st.write('Growth Score: ', last_rows_num_gs.iloc[0]['growth_score'])
        actual_growth_score = last_rows_num_gs.iloc[0]['growth_score']
        
        
        test_df = add_repo_age_days_col(test_df)
        test_df = add_days_since_last_release_col(test_df)
        test_df = convert_topics_to_embeddings(test_df)
        test_df = remove_outliers(test_df)
        test_df = indexify_release_dates(test_df)
        test_df = add_lag_features_based_on_target(test_df, num=5)
        X = remove_unwanted_features(test_df)
        test_df_scaled = scale_final_data(X=X.values)
        test_df_scaled_pca = reduce_dimentionality_pca(test_df_scaled)

        n_features = test_df_scaled_pca.shape[1] 
        n_timesteps = 6
        n_forecast_steps = 12 

        test_generator = TimeseriesGenerator(test_df_scaled_pca, np.zeros(len(test_df_scaled_pca)), length=n_timesteps, batch_size=1)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_file_path = os.path.join(current_dir, '../models/best_rnn_model.keras')
        model_path = os.path.abspath(data_file_path)
        model = load_model(model_path)
        y_pred = model.predict(test_generator)

        st.text(f'predicted growth score: {y_pred[0][0]}')
        st.text(f'actual growth score: {actual_growth_score}')
        st.text(f'prediction diff: {y_pred[0][0] - actual_growth_score}')
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

        plot_historical_and_forecasted_growth(test_df, forecasted_values, n_forecast_steps,)

    

