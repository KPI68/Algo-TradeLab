# project2-team5

## Team and Resource
Oby Nwafor: PM

Felipe Jdanov

Hazel Akongoh

Kala Pi

Flatform: Jupyter Notebookds + Google Colab

## Project Topic 

Algorithm Trading Utilizing Machine Learning

## Goal

* Using data sources from market historically, create models incorporate trading algorithm, to predict trade actions
* Starting $10,000, simulate the trading actions for 15 days, calculate the net worth at the end
* Tune the models for premium results

## Data

Historical data of tickers from Alpaca

## Cross-reference
| Algo                   |Model|Train On|Measurements|ipynb|
|------------------------|-----|-------------|--------|---|
|Simple up/down buy/sell on close price |Tensorflow Neural Net   |Scaled Numeric Columns|ML scores + Trade Action Yield|neuralnet_algo_trade|
|Simple Moving Average on close price    |Support Vector Machine + Calibrated Classifier|Simple Moving Average of close price short+long |Calibration Plot + Comparison PLot|calibrating_classifier_felipe|
|Simple up/down buy/sell on vwap |Support Vector Machine Classifier, Regression   |Daily return, Scaled Numeric Columns, Numeric Columns pct_change |Trade Action Yield|algo-trade|

## Details

Refer the corresponding Python Jupyter Notebooks

## Next Steps

* Try better curve
* Try better model
* Try better trade algorithm  


