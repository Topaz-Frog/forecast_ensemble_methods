# Forecast Combination Methods

## Description
This project implements a variety of forecast combination methods to improve the accuracy of time series forecasting. By leveraging multiple forecasting models and combining their predictions, the aim is to produce a more robust and accurate forecast compared to using individual models.

### Purpose
The primary purpose of this project is to provide a comprehensive toolkit for combining different forecast models' predictions to minimize forecasting errors.
### Features

- **Error Calculation**: Functions to calculate Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- **Combination Methods**:
  - **Simple Average**: Averages the predictions of multiple models.
  - **Median**: Uses the median of predictions.
  - **Trimmed Mean**: Averages the central part of the predictions, trimming the extreme values.
  - **Winsorized Mean**: Limits extreme values before averaging.
  - **Bates-Granger**: Weighs models inversely to their RMSE.
  - **Inverse Rank**: Combines models based on the inverse of their rank by RMSE.
  - **Ordinary Least Squares (OLS)**: Uses linear regression for combination.
  - **Least Absolute Deviation (LAD)**: Combines forecasts using LAD regression.
  - **Positive Weights**: Ensures all weights are non-negative.
  - **Constrained Least Squares**: Adds constraints to the weights used in OLS.
  - **Complete Subset Regression**: Combines forecasts by averaging over subsets of models.
  - **Eigenvector Approaches**: Uses eigenvector methods to combine forecasts.
  - **Bias-Corrected Eigenvector**: Adds bias correction to the standard eigenvector method.
  - **Trimmed Eigenvector**: Trims the models based on RMSE before applying eigenvector methods.
  - **Trimmed Bias-Corrected Eigenvector**: Adds bias correction to the trimmed eigenvector method.

### How it Works
1. **Data Input**: The project expects a pandas DataFrame containing the real values and predictions from various models.
2. **Error Calculation**: MAE and RMSE are used to evaluate the accuracy of each model.
3. **Combination**: Various methods are applied to combine the forecasts, aiming to reduce overall error.
4. **Output**: The combined forecast is returned as a pandas Series.

## Usage

Below is an example of how to use the code in the `__main__.py` file to run forecast combination methods.

### Example

1. **Prepare Your Data**:
    - Ensure you have a DataFrame `df` with real prices and predictions from various models.
    - Example DataFrame structure:
      ```python
      import pandas as pd

      data = {
          'Real price': [100, 150, 200, 250],
          'Model 1': [102, 148, 198, 252],
          'Model 2': [99, 151, 201, 249],
          'Model 3': [101, 149, 197, 248]
      }
      df = pd.DataFrame(data)
      ```

2. **Choose a Combination Method**:
    - Call the desired combination method, e.g., `simple_average_combination`.
    - Example:
      ```python
      from forecast_ensemble_methods import simple_average_combination

      combined_forecast = simple_average_combination(df, ['Model 1', 'Model 2', 'Model 3'])
      print(combined_forecast)
      ```

3. **Run the Script**:
    - Execute the main script to see the results.
    - Example command:
      ```bash
      python __main__.py
      ```

Replace `simple_average_combination` with any other combination method you want to use. Ensure that the data columns and method parameters match your dataset and requirements.
