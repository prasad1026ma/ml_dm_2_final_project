# ML + DM 2 Final Project: Krithika Natarajan and Maya Prasad

## Problem Statement
Renters in the Greater Boston area face significant challenges in understanding when and where to rent. The Boston rental market is highly fragmented as pricing dynamics in university-adjacent neighborhoods like Allston differ drastically from downtown or suburban areas, and these local trends are not captured by national tools like Zillow's market metrics. This project aims to build a set of models that help renters make more informed decisions by predicting future rental prices and classifying whether a given zip code represents a good renting opportunity at a given point in time.


## Data
- Zillow Observed Rent Index (ZORI): Monthly rental price index at the zip code level across Greater Boston
- Zillow Observed Rent Forecast (ZORF): A month-ahead, quarter-ahead and year-ahead forecast of the Zillow Observed Rent Index (ZORI).
- Census Demographics: Population, income, and housing characteristics by zip code via data.census.gov
- Zillow Observed Renter Demand Index (ZORDI): Demand-side rental pressure index from Zillow
- Spatial Features: Binary flag close_to_university derived from proximity to colleges and universities
- Temporal Features: Boolean variable is_before_september / is_after_september to capture Boston's student-driven seasonal rental cycle
- Weather Patterns: Seasonal weather data as a supplementary temporal signal
  
## Models
### 1. Bayesian Model
  A Bayesian model that forecasts future rental prices for a given zip code by forming a prior belief from historical Zillow rental data and updating that belief using neighborhood-level supplementary features. This approach allows the model to incorporate existing market knowledge while remaining responsive to local signals like university proximity and seasonal demand shifts.

### 2. CNN
  A CNN-based classifier that takes a zip code and its historical rental trend as well as some other factors as input and outputs a binary recommendation (good time to rent or not) based on whether current prices are favorable relative to the area's recent trend. This model is designed to give renters a simple, actionable signal rather than requiring them to interpret raw price data themselves.

