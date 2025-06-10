# README: Song Popularity Prediction Project

## Project Overview
This project explores the prediction of song popularity using a combination of audio features and textual data. The aim is to uncover the factors influencing a song's success by leveraging advanced machine learning techniques and robust data analysis. The insights from this project can empower stakeholders in the music industry, including independent artists and producers, to make data-driven decisions.

## Datasets
Two primary datasets were used for this project:

1. **Spotify 1 Million Tracks Dataset**
   - Source: Kaggle
   - Contains metadata and audio features for over 1 million songs (2000-2023).
   - Key Features: `danceability`, `energy`, `tempo`, `valence`, `popularity` (0-100 scale).

2. **Genius Song Lyrics Dataset**
   - Source: Scraped from Genius.com
   - Provides song lyrics and metadata, including views on lyric pages.
   - Key Features: `unique_word_count`, `sentiment scores`, `lyric_page_counter`.

## Workflow and File Descriptions

### 1. Data Cleaning and Preprocessing
- **1_Spotify_data_cleaning.ipynb**
  - Processes the Spotify dataset by handling missing values and normalizing columns.
  - Ensures consistency in audio features.

- **2_Genius_data_cleaning.ipynb**
  - Cleans and standardizes the Genius dataset, focusing on removing non-music entries, formatting lyrics, and deriving features such as `word_count` and `sentiment polarity`.

### 2. Dataset Integration
- **3_Dataset_joining.ipynb**
  - Merges the Spotify and Genius datasets based on standardized artist and track names.
  - Creates a unified dataset with 139,433 records and 39 columns.

### 3. Exploratory Data Analysis (EDA)
- **4_Exploratory_data_analysis.ipynb**
  - Provides insights into the dataset, including:
    - Popularity distribution.
    - Genre representation and trends over time.
    - Analysis of audio features (e.g., `energy`, `danceability`).
    - Lyrical complexity by genre.

### 4. Feature Engineering and Splitting
- **5_Distribution_split_and_random_forest.ipynb**
  - Transforms the target variable (`popularity`) into classes (e.g., low, neutral, high) using percentile-based thresholds.
  - Trains a Random Forest model as a baseline classifier.

### 5. Model Training and Evaluation
- **6_Multiple_model_training.ipynb**
  - Trains and evaluates various models:
    - Logistic Regression
    - K-Nearest Neighbors (KNN)
    - Multi-Layer Perceptron (MLP)
    - Support Vector Machines (SVM)
    - XGBoost

- **7_XGboost_on_balanced_dataset.ipynb**
  - Applies XGBoost on a balanced dataset (33% class distribution) to improve fairness.
  - Analyzes the trade-offs in accuracy and fairness.

- **8_XGboost_on_binary_classification.ipynb**
  - Implements binary classification for popularity prediction (high vs. low popularity).
  - Achieves improved performance with a simpler binary split.

## Key Insights
1. **Popularity Trends**
   - Songs have become increasingly popular over time, with significant changes in listener preferences.
   - Hip-hop and alt-rock remain dominant genres.

2. **Feature Importance**
   - Audio features like `danceability` and `energy` strongly correlate with popularity.
   - Lyrical features such as `unique_word_count` and sentiment polarity play a critical role in certain genres.

3. **Model Performance**
   - XGBoost emerged as the best-performing model with an accuracy of 74.28% on the imbalanced dataset.
   - Binary classification boosted performance to 84.42% accuracy, demonstrating the effectiveness of simplifying the target variable.

## Future Work
- Incorporate real-time data from streaming platforms for dynamic analysis.
- Explore deep learning models for improved feature representation.
- Analyze external factors such as social media trends and marketing strategies.

## Requirements
- Python 3.9+
- Key Libraries: Pandas, NumPy, Scikit-learn, XGBoost, Seaborn, Matplotlib

## How to Run
1. Clone this repository and navigate to the project folder.
2. Run the Jupyter notebooks in sequence to replicate the analysis and models.

## Authors
- **Aldridge Fonseca** 
- **Sushant Lokhande**

For questions or contributions, feel free to reach out!
