# ML + DM 2 Final Project: Krithika Natarajan and Maya Prasad
## Overview
Two models to help Boston renters identify favorable rental timing and predict zip-level rental prices. Boston's rental market is highly seasonal and fragmented across neighborhoods, making national tools like Zillow insufficient for local decision-making.

- **Bayesian Hierarchical Model**: Predicts zip-level ZORI rental prices across Boston neighborhoods, 
  capturing local rent dynamics via nested random effects and academic calendar seasonality.
- **CNN Classifier**: Classifies whether current market conditions represent a favorable time to rent 
  for a given zip code, using a 12-month sequential window of temporal and spatial features.

## Problem Statement
Renters in Greater Boston face significant challenges in understanding when and where to rent. 
The market is highly fragmented as pricing dynamics in university-adjacent neighborhoods like 
Allston differ drastically from downtown or suburban areas. These local trends are not 
captured by national tools like Zillow.

Boston's rental market is shaped by dense, local signals: university schedules driving 
seasonal move-in and move-out cycles, neighborhood demographics, and historical demand. 
A single national model systematically misrepresents these conditions, motivating a 
Boston-specific approach.

This project delivers two linked outputs to Boston renters: a prediction of expected 
rental price given a property's location and characteristics, and a classification of 
whether current market conditions represent a favorable time to rent in a given zip code.

## Data
- **Zillow Observed Rent Index (ZORI)**: Monthly rental price index at the zip code level across Greater Boston, sourced from Zillow.
  
## Models
### 1. Bayesian Model
  A Bayesian model that forecasts future rental prices for a given zip code by forming a prior belief from historical Zillow rental data and updating that belief using neighborhood-level supplementary features. This approach allows the model to incorporate existing market knowledge while remaining responsive to local signals like university proximity and seasonal demand shifts.

### 2. CNN
  A CNN-based classifier that takes a zip code and its historical rental trend as well as some other factors as input and outputs a binary recommendation (good time to rent or not) based on whether current prices are favorable relative to the area's recent trend. This model is designed to give renters a simple, actionable signal rather than requiring them to interpret raw price data themselves.

#### Code
- `cnn_utilities.py`: 
  - `MinMaxScaler`: 
  - `load_data`: 
  - `get_date_columns`:
  - `scale_prices`:
  - `calculate_volatility`:
  - `calculate_rate_of_change`:
  - `extract_month_from_date_string`:
  - `create_seasonal_features`:
  - `create_september_tracker`
- `manual_neural_network.py`: 
  - `relu`/ `relu_grad`:
  - `tanh_grad`:
  - `sigmoid`,`sigmoid_grad`:
  - `binary_crossentropy`:
  - `Manual_NN`:
    - `forward`:
    - `backward`:
    - `train`:
    - `evaluate`:
    - `predict`:
- `cnn_predictions.py`: 
  - `create_sequences`:
  - `process_data`:
  - `model_eval`:
  - `prepare_training_data`:
  - `build_features`:
  - `predict_zip`:
  - `main`:
- `bayesian_model.Rmd`: Bayesian hierarchical model using `brms` in R to predict zip-level rental prices across Boston neighborhoods.
  - **Data Preparation**: pivots ZORI data to long format and encodes date features
  - **Feature Engineering**: computes volatility, September intensity, and sin/cos seasonality
  - **EDA**: zip-level and neighborhood-level rent trend visualizations
  - **Model Definition**: Gamma-log formula with prior specification
  - **Model Fitting**: 80/20 train/test split, 4 chains, 2000 iterations via BRMS, adapt_delta=0.99 & max_tree=15
  - **Evaluation**: RMSE, Bayes $R^2$, posterior predictive checks
