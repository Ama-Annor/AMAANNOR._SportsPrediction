#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import libraries
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# In[4]:


# loading Dataset needed

fifa_df = pd.read_csv('male_players (legacy).csv', na_values="")
fifa_df


# # Q1: Data Preparation & Feature Extraction Process

# In[5]:


# Function to clean data
first_run = True
scaler = StandardScaler()

# Function to clean data
def data_cleaning(df):
    df = df.select_dtypes(include=np.number)
    df = df.drop(columns=[col for col in df.columns if col not in [
        'overall', 'height_cm', 'weight_kg', 'age', 'physic', 'power_strength',
        'power_jumping', 'movement_agility', 'movement_balance', 'dribbling',
        'skill_dribbling', 'skill_ball_control', 'shooting', 'passing',
        'skill_long_passing', 'skill_fk_accuracy', 'attacking_crossing',
        'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing',
        'attacking_volleys', 'mentality_aggression', 'mentality_interceptions',
        'mentality_positioning', 'mentality_vision', 'mentality_penalties',
        'mentality_composure', 'movement_reactions', 'pace', 'movement_acceleration',
        'movement_sprint_speed', 'power_stamina', 'power_shot_power', 'power_long_shots',
        'defending']])
    df.dropna(thresh=np.floor(len(df) * 0.50), axis=1, inplace=True)
    return df

# Function to impute and scale data
def imp_scale(df):
    global first_run  # Use the global flag
    imputer = SimpleImputer(strategy='median')
    imputed_data = imputer.fit_transform(df)
    imputed_df = pd.DataFrame(imputed_data, columns=df.columns)

    scaler.fit(imputed_df)
    standardized_data = scaler.transform(imputed_df)
    standardized_df = pd.DataFrame(standardized_data, columns=imputed_df.columns)

    return standardized_df

# The data is cleaned and preprocessed
cleaned_data = data_cleaning(fifa_df)
imputed_and_scaled_data = imp_scale(cleaned_data)


# # Q2: Create feature subsets that show max correlation with the dependent variable

# In[6]:


def calculate_corr(df):
    # Calculate the correlation between each feature and the dependent variable (player rating)
    correlations = df.corr()['overall']
    
    # Remove the correlation of the dependent variable with itself
    correlations = correlations.drop('overall')

    # Select the top 10-15 features with the highest correlation
    top_features = correlations.nlargest(11)

    # Print the top 10-15 correlations between features and the dependent variable
    print("Top 10-15 correlations between features and the dependent variable:")
    print("---------------------------------------------------------------")
    return top_features

calculate_corr(data_cleaning(fifa_df))


# # Q3: Create and train a suitable machine learning model with cross-validation that can predict a player's rating.

# In[7]:


# Import necessary modules for model training and evaluation

from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

#The data is cleaned and preprocessed
cleaned_data = data_cleaning(fifa_df)
imputed_and_scaled_data = imp_scale(cleaned_data)
imputed_and_scaled_data

# Split the data into training and testing sets
X = imputed_and_scaled_data.drop('overall', axis=1)
y = imputed_and_scaled_data['overall']

# Spliting into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Individual models

# Train a Random Forest Regressor on the training data
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Train a Random Forest Regressor on the training data
xgb = XGBRegressor()
xgb.fit(X_train, y_train)

# Train a Gradient Boosting Regressor on the training data
gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)

# Evaluate individual models using cross-validation

# Evaluate the Random Forest Regressor using cross-validation
rf_scores = cross_val_score(rf, X, y, cv=3)

# Evaluate the XGBoost Regressor using cross-validation
xgb_scores = cross_val_score(xgb, X, y, cv=3)

# Evaluate the Gradient Boosting Regressor using cross-validation
gb_scores = cross_val_score(gb, X, y, cv=3)

# Print cross-validation scores for individual models
print("The following are the cross-validation scores for individual models:")

print(f"Random Forest Regressor: Mean score = {round(rf_scores.mean(), 2)}, Scores = {rf_scores}")

print(f"XGBoost Regressor: Mean score = {round(xgb_scores.mean(), 2)}, Scores = {xgb_scores}")

print(f"Gradient Boosting Regressor: Mean score = {round(gb_scores.mean(), 2)}, Scores = {gb_scores}")


# # Q4: Measure the model's performance and fine-tune it as a process of optimization.

# In[8]:


# Import necessary modules

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor

cleaned_data = data_cleaning(fifa_df)
imputed_and_scaled_data = imp_scale(cleaned_data)
imputed_and_scaled_data

scaler.fit(imputed_and_scaled_data)
joblib.dump(scaler,'scaler.pkl')

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate the Mean Absolute error
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")

    # Calculate Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

def fine_tune_model(model, X_train, y_train, X_test, y_test):
    
    # Gradient Boosting Regressor - creating parameter distribution
    param_grid = {'n_estimators': randint(100, 500),
        'learning_rate': [0.01, 0.1, 0.5, 1],
        'max_depth': randint(3, 9),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 10),
        'subsample': [0.5, 0.8, 1.0],
        'max_features': [1, 'sqrt', 'log2']}

    # Fine-tune Gradient Boosting Regressor with parameter distribution using RandomizedSearchCV
    random_search = RandomizedSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_iter=10, n_jobs = -1)
    random_search.fit(X_train, y_train)

    # Get the best-performing model
    best_model = random_search.best_estimator_

    # Evaluate the fine-tuned model
    evaluate_model(best_model, X_test, y_test)
    return best_model

#A  model instance
model = GradientBoostingRegressor()

# Split the data into training and testing sets
X = imputed_and_scaled_data.drop('overall', axis=1)
y = imputed_and_scaled_data['overall']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Fit the model to the training data
model.fit(X_train, y_train)

#Evaluating the initial model
evaluate_model(model, X_test, y_test)

#The model Fine-tuned
fine_tune_model(model, X_train, y_train, X_test, y_test)


# # Q5: Use the data from another season(players_22) which was not used during the training to test how good is the model. 

# In[12]:


# Import necessary modules 

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# make predictions on all features, store them and compare them to the actual values
testing_data = pd.read_csv('C:/Users/HP/OneDrive/Desktop/Intro to AI/players_22-1.csv')

# The testing data is cleaned and preprocesses
cleaned_testing_data = data_cleaning(testing_data)

imputed_and_scaled_testing_data = imp_scale(cleaned_testing_data)

# Make predictions on the testing data
X_testing = imputed_and_scaled_testing_data.drop('overall', axis=1)

y_testing = imputed_and_scaled_testing_data['overall']

#The fine-tuned model is used to make predictions
best_model = fine_tune_model(model, X_train, y_train, X_test, y_test)

predictions = best_model.predict(X_testing)

# Store the predicted ratings and actual values
#Predicted ratings
predicted_ratings = pd.DataFrame({'Predicted Rating': predictions})

#Actual ratings
actual_ratings = pd.DataFrame({'Actual Rating': y_testing})

print(f"""
Mean Absolute Error = {mean_absolute_error(predictions,y_testing)},
Mean Squared Error = {mean_squared_error(predictions,y_testing)},
Root Mean Squared Error = {np.sqrt(mean_squared_error(predictions,y_testing))},
R2 Score = {r2_score(predictions,y_testing)}
""")

# Compare the predicted ratings with the actual ratings
comparison = np.sqrt(mean_squared_error(predicted_ratings, actual_ratings))
print("---------------------------------------------------------------")
print(f"The Root Mean Squared Error (RMSE) between the predicted and actual ratings DataFrames is: {round(comparison, 2)}")


# In[ ]:


#Save the best model to a file using joblib
joblib.dump(best_model, 'best_enemble_model.pkl')


# In[ ]:




