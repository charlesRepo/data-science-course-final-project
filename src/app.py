import streamlit as st
from functions import get_repo_data
from functions import display_repo_data

st.set_page_config(layout="wide")

if 'main_repo_data' not in st.session_state:
    st.session_state.main_repo_data = None

if 'comparison_repo_data' not in st.session_state:
    st.session_state.comparison_repo_data = None

st.title('AI-Related Repository Growth Forecast')
st.markdown('Visit this link for a [list of repositories](https://github.com/search?q=%28AI+OR+artificial+intelligence%29&type=repositories&s=stars&o=desc).', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Single", "Compare"])

with tab1:
    col1 = st.columns(1)[0]
    with col1:
        st.subheader('Main Repo Data')
        url = st.text_input("Enter a Github Repository URL", "", key="main_repo_url")
        if st.button("Fetch Repo data", key="main_repo_url_button"):
            if url:
                test_df, y_pred, forecasted_values, n_forecast_steps = get_repo_data(url)
                st.session_state.main_repo_data = (test_df, y_pred, forecasted_values, n_forecast_steps)
        if st.session_state.main_repo_data:
            test_df, y_pred, forecasted_values, n_forecast_steps = st.session_state.main_repo_data
            display_repo_data(test_df, y_pred, forecasted_values, n_forecast_steps)
        else:
            st.write("No data fetched yet for Main Repo.")
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Main Repo Data')
        url = st.text_input("Enter a Github Repository URL", "", key="main_repo_url_compare")
        if st.button("Fetch Repo data", key="main_repo_url_button_compare"):
            if url:
                test_df, y_pred, forecasted_values, n_forecast_steps = get_repo_data(url)
                st.session_state.main_repo_data = (test_df, y_pred, forecasted_values, n_forecast_steps)
        if st.session_state.main_repo_data:
            test_df, y_pred, forecasted_values, n_forecast_steps = st.session_state.main_repo_data
            display_repo_data(test_df, y_pred, forecasted_values, n_forecast_steps)
        else:
            st.write("No data fetched yet for Main Repo.")

    with col2:
        st.subheader('Comparison Repo Data')
        url_2 = st.text_input("Enter a Github Repository URL", "", key="comparison_repo_url")
        if st.button("Fetch Repo data", key="comparison_repo_url_button"):
            if url_2:
                # Fetch the data and update session state
                test_df, y_pred, forecasted_values, n_forecast_steps = get_repo_data(url_2)
                st.session_state.comparison_repo_data = (test_df, y_pred, forecasted_values, n_forecast_steps)
        if st.session_state.comparison_repo_data:
            test_df, y_pred, forecasted_values, n_forecast_steps = st.session_state.comparison_repo_data
            display_repo_data(test_df, y_pred, forecasted_values, n_forecast_steps)
        else:
            st.write("No data fetched yet for Comparison Repo.")