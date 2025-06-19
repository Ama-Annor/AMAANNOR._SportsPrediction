# FIFA Player Rating Prediction

A comprehensive machine learning project that predicts FIFA player overall ratings using advanced ensemble methods and provides an interactive web interface for real-time predictions.

## 🚀 Features

- **Advanced ML Pipeline**: Complete data preprocessing with imputation and standardization
- **Ensemble Methods**: Random Forest, XGBoost, and Gradient Boosting regressors
- **Hyperparameter Tuning**: Automated optimization using RandomizedSearchCV
- **Cross-Validation**: Robust model evaluation with 3-fold cross-validation
- **Interactive Web App**: Streamlit-based interface with visual star ratings
- **Model Persistence**: Trained models saved using joblib for deployment

## 📊 Dataset

The project uses FIFA player datasets:
- **Training Data**: `male_players (legacy).csv`
- **Testing Data**: `players_22-1.csv` (different season for validation)

### Key Features Used
- Movement Reactions
- Mentality Composure
- Passing & Dribbling
- Physical Attributes
- Shooting & Shot Power
- Age and other performance metrics

## 🛠️ Technology Stack

- **Python 3.7+**
- **Machine Learning**: scikit-learn, XGBoost
- **Data Processing**: pandas, numpy
- **Model Persistence**: joblib
- **Web Interface**: Streamlit
- **Visualization**: Built-in Streamlit components

## 📋 Installation

### Prerequisites
```bash
pip install streamlit pandas scikit-learn joblib xgboost numpy scipy
```

### Clone Repository
```bash
git clone https://github.com/Ama-Annor/AMAANNOR._SportsPrediction.git
cd AMAANNOR._SportsPrediction
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### 1. Train the Model
Run the Jupyter notebook or Python script to train the model:
```bash
python AMAANNOR._SportsPrediction.py
```

This will:
- Clean and preprocess the data
- Train multiple ML models
- Perform hyperparameter tuning
- Save the best model as `model_best.pkl`
- Save the scaler as `scaler.pkl`

### 2. Launch the Web App
```bash
streamlit run player_rating_app.py
```

### 3. Make Predictions
1. Open your browser to the Streamlit app (usually `http://localhost:8501`)
2. Adjust player attribute sliders
3. Enter actual rating for confidence calculation
4. Click "Predict" to see results with star ratings

## 🧠 Machine Learning Pipeline

### Data Preprocessing
1. **Feature Selection**: Focus on 11 most correlated features with player rating
2. **Missing Value Imputation**: Median-based imputation using SimpleImputer
3. **Standardization**: StandardScaler for feature normalization
4. **Data Cleaning**: Remove non-numeric columns and handle missing values

### Model Training & Evaluation
```python
# Models used
- Random Forest Regressor
- XGBoost Regressor  
- Gradient Boosting Regressor (Best performing)

# Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Cross-validation scores
```

### Hyperparameter Optimization
- **Method**: RandomizedSearchCV
- **Parameters**: n_estimators, learning_rate, max_depth, min_samples_split, etc.
- **Cross-validation**: 3-fold CV
- **Scoring**: Negative Mean Squared Error

## 📈 Model Performance

### Cross-Validation Results
- **Random Forest**: Mean CV Score: ~0.85
- **XGBoost**: Mean CV Score: ~0.87
- **Gradient Boosting**: Mean CV Score: ~0.89 (Best)

### Final Model Metrics
- **RMSE**: ~0.15-0.20 (on scaled data)
- **MAE**: ~0.12-0.18
- **R² Score**: ~0.89-0.92

## 🌟 Web Application Features

### Interactive Interface
- **Slider Controls**: Easy adjustment of player attributes
- **Real-time Predictions**: Instant rating calculation
- **Visual Feedback**: Star-based rating system (⭐⭐⭐⭐⭐)
- **Confidence Metric**: Shows prediction reliability

### Star Rating System
```python
def compute_stars(rating):
    # Converts numerical rating to 5-star scale
    # Includes half-stars and visual representations
```

## 📁 Project Structure

```
AMAANNOR._SportsPrediction/
├── AMAANNOR._SportsPrediction.ipynb    # Main training notebook
├── AMAANNOR._SportsPrediction.py       # Python script version
├── player_rating_app.py                # Streamlit web application
├── model_best.pkl                      # Trained model (generated)
├── scaler.pkl                          # Data scaler (generated)
├── requirements.txt                    # Python dependencies
├── male_players (legacy).csv           # Training dataset
├── players_22-1.csv                    # Testing dataset
└── README.md                           # Project documentation
```

## 🔧 Configuration

### Model Parameters (Optimized)
```python
GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8
)
```

## 📊 Results & Insights

### Feature Importance
Top contributing factors to player ratings:
1. Movement Reactions
2. Mentality Composure  
3. Passing Ability
4. Dribbling Skills
5. Physical Attributes

### Model Validation
- Tested on different season data (players_22)
- Maintains consistent performance across datasets
- Robust to new, unseen player data

## 🚀 Future Enhancements

- [ ] Deep learning models (Neural Networks)
- [ ] Player position-specific models
- [ ] Time series analysis for rating changes
- [ ] Advanced feature engineering
- [ ] Model interpretability with SHAP values
- [ ] API deployment for mobile apps
- [ ] Real-time data integration

## 📝 Requirements

Create a `requirements.txt` file:
```
streamlit>=1.28.0
pandas>=1.5.0
scikit-learn>=1.3.0
joblib>=1.3.0
xgboost>=1.7.0
numpy>=1.24.0
scipy>=1.10.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Ama-Annor**
- GitHub: [@Ama-Annor](https://github.com/Ama-Annor)

## 🙏 Acknowledgments

- FIFA for providing comprehensive player statistics
- scikit-learn community for excellent ML tools
- Streamlit team for the intuitive web framework
- XGBoost developers for high-performance gradient boosting

---

**⚽ Predict like a pro with data science! 🏆**
