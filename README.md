# 🌤️ Solar Radiation Prediction

A machine learning project for predicting solar radiation levels using weather conditions and time-based features.

---

## 📌 Overview

This project uses meteorological data collected from the **HI-SEAS weather station in Hawaii, USA** between **September and December 2016**.

The objective is to predict **solar radiation (W/m²)** using weather measurements and engineered time features. Several regression algorithms were evaluated, and **LightGBM** achieved the best overall performance.

---

## 📂 Dataset

* **Source:** Kaggle – Solar Radiation Prediction Dataset
* **Location:** Hawaii, USA
* **Period:** September – December 2016
* **Number of records:** 32,686
* **Sampling frequency:** Approximately every 5 minutes

### Original Features

| Column                 | Description                       |
| ---------------------- | --------------------------------- |
| UNIXTime               | Unix timestamp                    |
| Data                   | Date of observation               |
| Time                   | Time of observation               |
| Radiation              | Solar radiation (Target variable) |
| Temperature            | Air temperature (°F)              |
| Pressure               | Atmospheric pressure (Hg)         |
| Humidity               | Relative humidity (%)             |
| WindDirection(Degrees) | Wind direction (0–360°)           |
| Speed                  | Wind speed (mph)                  |
| TimeSunRise            | Sunrise time                      |
| TimeSunSet             | Sunset time                       |

---

## 🛠 Data Preparation

### Data Cleaning

* Converted column names to lowercase.
* Verified missing values and duplicates.
* Removed unnecessary columns.

### Feature Engineering

Created several time-related features:

* `hour`
* `month`
* `minute`
* `second`
* `sunrise_minutes`
* `sunset_minutes`
* `minutes_since_sunrise`
* `minutes_to_sunset`

After experimentation, some low-importance features were removed, resulting in the final dataset used for training.

---

## 📊 Exploratory Data Analysis

Several analyses were performed:

* Distribution of numerical variables.
* Correlation analysis.
* Radiation distribution.
* Average radiation throughout the day.
* Feature importance visualization.

### Key Findings

* **Temperature** has the strongest positive correlation with solar radiation.
* Radiation peaks around **11 AM – 1 PM**.
* More than half of the readings correspond to nighttime or very low radiation.
* Engineered features based on sunrise and sunset significantly improved performance.

---

## 🤖 Models Evaluated

Seven regression algorithms were compared:

1. Linear Regression
2. K-Nearest Neighbors
3. Decision Tree
4. Random Forest
5. CatBoost
6. XGBoost
7. LightGBM

Model evaluation was performed using **5-fold Cross Validation**.

---

## 🏆 Best Model

### LightGBM Regressor

Performance:

* **Train R²:** 0.9566
* **Test R²:** 0.8205
* **MSE:** 15,986.92
* **MAE:** 58.51

LightGBM showed the best balance between accuracy and generalization.

---

## ⚙ Hyperparameter Tuning

`RandomizedSearchCV` was used to optimize LightGBM hyperparameters.

Best parameters:

```python
n_estimators = 200
learning_rate = 0.03
max_depth = 5
num_leaves = 30
```

---

## 📈 Feature Importance

The most influential features were:

1. `minutes_since_sunrise`
2. `minutes_to_sunset`
3. `humidity`
4. `pressure`
5. `temperature`
6. `winddirection(degrees)`
7. `hour`

These results indicate that the position of the sun relative to sunrise and sunset is more important than raw clock time.

---

## 🌐 Streamlit Application

The project includes two interactive pages:

### 📄 Overview Page

Contains:

* Dataset summary
* Key metrics
* Data preview
* Feature descriptions

### 🔬 Analysis Page

Provides answers to questions such as:

* What affects solar radiation the most?
* When is radiation at its peak?
* Is the dataset balanced?
* Which features are most important?
* Which model performs best?

---

## 📦 Libraries Used

* Pandas
* NumPy
* Plotly
* Streamlit
* Scikit-Learn
* LightGBM
* XGBoost
* CatBoost
* Joblib

---

## 🚀 Future Improvements

* Incorporate weather forecast APIs.
* Experiment with deep learning models.
* Build a real-time prediction system.
* Deploy the application to Streamlit Cloud.

---

## Author

**Mohammed Abdelhay**

Faculty of Agricultural Engineering
Data Science & Machine Learning Enthusiast
