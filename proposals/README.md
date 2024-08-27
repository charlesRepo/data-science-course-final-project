# Final Project Proposal

# I. Introduction

This project will develop a neural network-based time series model to forecast which company will likely lead the AI ecosystem in the coming year, offering insights into the evolving AI landscape.

## II. Project Objective

The goal of this project is to analyze historical trends in AI and machine learning development through GitHub repository data and to develop a predictive model that forecasts which tech organization is likely to lead the AI ecosystem in the coming year. By leveraging metrics such as stars, forks, issues, and commit activity, the project aims to provide insights into the future direction of AI innovation.

The rapid advancement of AI technologies is driven by contributions from leading tech organizations, but predicting which company will dominate the AI landscape is challenging. This project seeks to answer the question: **Which tech organization is likely to lead the AI ecosystem in the next year based on trends in their GitHub activity?** By solving this problem, the project aims to provide valuable foresight into the dynamics of AI innovation and help identify the key players shaping the future of AI.

## III. Data Description

The data for this project will be sourced from GitHub, specifically through the GitHub API. The API provides access to repository information from leading tech organizations like Facebook Research, Google Research, OpenAI, Microsoft, and others. The data will include various metrics related to repository activity, such as stars, forks, issues, and commit frequency.

This is an example of some of the organizations on Github that are currently leading the AI advancement efforts, and which my analysis will be based on;
`orgsList = ['facebookresearch', 'googleresearch', 'openai', 'microsoft', 'midjourney', 'ibm', 'amzn', 'apple', 'NVIDIA']`

The data retrieved from the GitHub API will be in JSON format, containing key information about each repository. The structure of the data includes:

- **Repository Information**: Each repository is represented as an object with fields such as:
    - `name`: The name of the repository.
    - `full_name`: The full name of the repository, including the organization (e.g., `facebookresearch/DeepLearning`).
    - `html_url`: The URL of the repository on GitHub.
    - `created_at`: The date and time when the repository was created.
    - `updated_at`: The date and time of the last update to the repository.
    - `stargazers_count`: The number of stars the repository has received.
    - `forks_count`: The number of times the repository has been forked.
    - `open_issues_count`: The number of open issues in the repository.
    - `watchers_count`: The number of users watching the repository.
- **Additional Data (Fetched Separately)**:
    - `commits`: Number and frequency of commits to the repository.
    - `pull_requests`: Number of pull requests.
    - `contributors`: Number of contributors to the repository.
    - `closed_issues`: Number of issues that have been closed (requires separate API calls).

This project is primarily a **regression** problem. The goal is to predict a continuous value, such as the level of activity or influence of a tech organization (e.g., number of stars, forks, or contributions) over time. The output is a continuous variable rather than a discrete category, making it a regression task.

## IV. Methodology

### **1. Data Collection**:

- Gather data on AI and machine learning repositories and collect key metrics to measure repository engagement.

### **2. Data Preprocessing**:

- Convert JSON responses into a structured format and address inconsistencies by handling missing values.

### **3. Time Series Analysis**:

- Examine temporal trends in repository activity and decompose the data into its underlying components.

### **4. EDA**:

- Generate summary statistics and visualizations to uncover patterns and identify anomalies in the data.
- Plot the data to get better insight how the data is changing overtime

### **5. Feature Selection and Transformation**:

- Select important features based on analysis and normalize the data for consistent model input.

### **6. Clustering (optional):**

- Apply clustering techniques to group similar repositories or organizations based on their activity metrics, providing insights into patterns within the data.
- This addition can enhance your analysis and help in understanding different segments in the AI ecosystem.
- Optional: depending on available time

### **7. Model Development**:

- Create baseline models for comparison and design a neural network architecture suitable for time series prediction. Most probably due to the nature of the time series prediction, I will be using an LSTM (Long Short Term Memory) model in order to be able find more meaningful relationships in the historical data.

### **8. Model Training**:

- Split the data into training and validation sets and tune hyperparameters to optimize model performance.

### **9. Model Evaluation**:

- Evaluate the model using performance metrics and compare its effectiveness against baseline models.

### **10. Forecasting and Prediction**:

- Use the trained model to make future activity predictions and analyze the impact of changing input variables.

### **11. Result Interpretation and Visualization (Optional)**:

- Visualize predicted activity levels against actual data and provide insights on the implications for the AI landscape.
- Optional: depending on available time

### **12. Delivery**

## V.  **Expected Deliverables**

- Predictive model that will be able to forecast which company is going to dominate the AI advancements in the coming year
- A presentation showcasing my methodology, findings, and insights.
- A public repository on Github housing the code for the model development
- A dashboard in Tableau that uses my deployed model to effectively communicate my results and predictions.

## VI. Timeline and Tasks

| **Aug 27th** | Complete: **Proposal** |
| --- | --- |
| **Sept 3rd** | Complete: **Data Collection, Data Preprocessing, Time Series Analysis** |
| **Sept 10th** | Complete: **EDA, Feature Selection and Transformation** |
| **Sept 17th** | Complete: **Model Development, Model Training, Model Evaluation, Forecasting and Prediction** |
| **Sept 19th** | Complete: **Project + Draft Presentation** |
| Sept 24th | Complete: **Deliveries** |
| **Sept 26th & Sept 29th** | **Complete the whole project** |

## VII. Potential Challenges

There may be several challenges related to this project but here are some of the major ones:

### **Challenges**

**Data Collection Challenges**:

- Incomplete data may be returned, including missing commit histories or contributor information.
- Rate limits on GitHub API requests can hinder data collection efficiency.
- Changes to the GitHub API could disrupt data collection processes.
- The complex JSON format of returned data may require extensive parsing efforts.
- Public repositories vary in activity and quality, complicating trend analysis.
- The nature of the AI technology that is constantly evolving. A breakthrough in AI may not be necessarily related to the amount of effort in research and development, but rather to a lucky coincidence.

**Time Series Challenges**:

- There might be inconsistency in the time series data. There might be big gaps gaps between project creations and updates.
- Identifying seasonality and trends in repository activity can be complex and organization-specific.

**Interpretability Challenges**:

- Neural networks often lack transparency, making it difficult to explain predictions to stakeholders.

### Potential solutions

- Invest more time in data cleanup and EDA.
- Implement caching mechanisms to mitigate API rate limits and optimize request efficiency.
- Stay updated with GitHub API changes and be prepared to modify the data collection process accordingly.
- Collect as many data as possible to try to bridge the gap between time series dates.
- Develop robust data parsing methods to handle complex and nested JSON structures efficiently.