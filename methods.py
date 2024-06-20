import pandas as pd
import numpy as np
import math
import scipy
import itertools
from gurobi_optimods.regression import LADRegression
import warnings
warnings.filterwarnings("ignore")

pd.options.display.float_format = '{:20,.2f}'.format

REAL = 'Real price'

def format_elapsed_time(start, end):
    elapsed_seconds = end - start
    if elapsed_seconds < 60:
        return f"{elapsed_seconds:.2f} seconds"
    else:
        elapsed_minutes = elapsed_seconds / 60
        return f"{elapsed_minutes:.2f} minutes"

################ error calculation #################

def calculate_mae(real_values, forecast_values):
    errors = [abs(a - p) for a, p in zip(real_values, forecast_values)]
    mae = sum(errors) / len(errors)
    return mae

def calculate_rmse(real_values, forecast_values):
    rmse = np.sqrt(np.mean((np.array(real_values) - np.array(forecast_values)) ** 2))
    return float("{:.6f}".format(rmse))
    

############### combination methods #################

def simple_average_combination(df, columns):
    return df[columns].mean(axis=1)

def median_combination(df, columns):
    return df[columns].median(axis=1)

def trimmed_mean_combination(df, columns, lam_trim):
    def trimmed_mean(series):
        values = series.sort_values().values
        P = len(values)
        trimmed_values = values[int(lam_trim * P) : int((1 - lam_trim) * P) +1]
        return trimmed_values.mean() if len(trimmed_values) > 0 else float('nan')
    
    return df[columns].apply(trimmed_mean, axis=1)

def winsorized_mean_combination(df, columns, lam_trim):
    def calc_winsorized_mean(row):
        values = [row[col] for col in columns]
        values.sort()

        P = len(values)
        min_allowed = values[math.floor(lam_trim * P)]
        max_allowed = values[P - math.floor(lam_trim * P) - 1]

        temp_values = [min(value, max_allowed) for value in values]
        temp_values = [max(value, min_allowed) for value in temp_values]

        return np.mean(temp_values)
    return df[columns].apply(calc_winsorized_mean, axis=1)

def bates_granger_combination(df, columns, days):
    real = df[REAL].values
    T = 24 * days
    res = []
    temp_series = df[REAL][T+24: len(real)]
    for idx in range(T+24, len(real)):
        RMSEs = {col: calculate_rmse(real[idx-T-24 : idx-24], df[col][idx-T-24 : idx-24]) for col in columns}

        inv_RMSEs = {col: 1 / rmse for col, rmse in RMSEs.items()}

        total_inv_RMSEs = sum(inv_RMSEs.values())
        normalized_inv_RMSEs = {col: rmse / total_inv_RMSEs for col, rmse in inv_RMSEs.items()}

        res.append(sum(df[col][idx] * weight for col, weight in normalized_inv_RMSEs.items()))

    res_series = pd.Series(res)
    res_series.index = temp_series.index
    return res_series

def inverse_rank_combination(df, columns, days):
    real = df[REAL].values
    T = 24 * days
    res = []
    temp_series = df[REAL][T+24: len(real)]
    for idx in range(T+24, len(real)):
        RMSEs = {col: calculate_rmse(real[idx-T-24 : idx-24], df[col][idx-T-24 : idx-24]) for col in columns}

        ranks = pd.Series(RMSEs).rank(method='min', ascending=True)
        inverse_ranks = 1 / ranks
        normalized_weights = inverse_ranks / inverse_ranks.sum()

        forecast = df[columns].iloc[idx].mul(normalized_weights).sum()
        
        res.append(forecast)

    res_series = pd.Series(res)

    res_series.index = temp_series.index
    return res_series

def ordinary_least_squares_combination(df, columns, days):
    real = df[REAL].values
    T = 24 * days
    res = []
    temp_series = df[REAL][T+24: len(real)]
    for idx in range(T+24, len(real)):
        data = df.iloc[idx-T-24 : idx-24]
        Y = data[REAL]

        X0 = np.ones(np.shape(Y))
        X1 = data[columns].values
        X = np.column_stack((X0, X1))

        arr = np.linalg.inv(X.T @ X)
        betas = arr @ (X.T @ Y)

        X_future = df[columns].iloc[idx].to_numpy()
        X_future = np.insert(X_future, 0, 1)

        forecast = X_future @ betas
        res.append(forecast)

    res_series = pd.Series(res)

    res_series.index = temp_series.index
    return res_series

def least_absolute_deviation(df, columns, days):
    real = df[REAL].values
    T = 24 * days
    res = []
    temp_series = df[REAL][T+24: len(real)]
    for idx in range(T+24, len(real)):
        data = df.iloc[idx-T-24 : idx-24]
        Y = data[REAL].values

        X0 = np.ones(np.shape(Y))
        X1 = data[columns].values
        X = np.column_stack((X0, X1))

        lad = LADRegression()
        lad.fit(X, Y, verbose=False)

        X_future = df[columns].iloc[idx].to_numpy()
        X_future = np.insert(X_future, 0, 1)

        forecast = lad.predict(X_future[np.newaxis, :])[0]
        res.append(forecast)

    res_series = pd.Series(res)

    res_series.index = temp_series.index
    return res_series

def positive_weights_combination(df, columns, days):
    def objective_function(w):
        return np.sum((Y - X.dot(w)) ** 2)
    
    real = df[REAL].values
    T = 24 * days
    res = []
    temp_series = df[REAL][T+24: len(real)]
    
    for idx in range(T+24, len(real)):
        data = df.iloc[idx-T-24 : idx-24]
        Y = data[REAL].values

        X0 = np.ones(np.shape(Y))
        X1 = data[columns].values
        X = np.column_stack((X0, X1))

        constraints = [
            {'type': 'ineq', 'fun': lambda w: w[1:]}
        ]
        
        w0 = np.zeros(len(columns) + 1)
        w0[1:] = 1 / len(columns)
        
        result = scipy.optimize.minimize(objective_function, w0, bounds=[(None, None)] + [(0, None)] * len(columns), constraints=constraints)
        betas = result.x

        X_future = df[columns].iloc[idx].to_numpy()
        X_future = np.insert(X_future, 0, 1)

        forecast = X_future @ betas
        res.append(forecast)

    res_series = pd.Series(res)

    res_series.index = temp_series.index
    return res_series

def constrained_least_squares_combination(df, columns, days):
    def objective_function(w):
        return np.sqrt(np.mean((Y - X.dot(w)) ** 2))
    
    def constraint_sum_to_one(w):
        return np.sum(w[1:]) - 1
    
    real = df[REAL].values
    T = 24 * days
    res = []
    temp_series = df[REAL][T+24: len(real)]
    
    for idx in range(T+24, len(real)):
        data = df.iloc[idx-T-24 : idx-24]
        Y = data[REAL].values

        X0 = np.ones(np.shape(Y))
        X1 = data[columns].values
        X = np.column_stack((X0, X1))

        constraints = [
            {'type': 'eq', 'fun': constraint_sum_to_one},
            {'type': 'ineq', 'fun': lambda w: w[1:]}
        ]
        
        w0 = np.zeros(len(columns) + 1)
        w0[1:] = 1 / len(columns)
        
        result = scipy.optimize.minimize(objective_function, w0, bounds=[(None, None)] + [(0, None)] * len(columns), constraints=constraints)
        betas = result.x

        X_future = df[columns].iloc[idx].to_numpy()
        X_future = np.insert(X_future, 0, 1)

        forecast = X_future @ betas
        res.append(forecast)

    res_series = pd.Series(res)

    res_series.index = temp_series.index
    return res_series

def complete_subset_regression_combination(df, columns, days, n):
    real = df[REAL].values
    T = 24 * days
    res = []
    temp_series = df[REAL][T+24: len(real)]
    combinations = list(itertools.combinations(columns, n))
    
    for idx in range(T+24, len(real)):
        data = df.iloc[idx-T-24 : idx-24]
        Y = data[REAL].values
        
        forecasts = []
        
        for combo in combinations:
            X0 = np.ones(np.shape(Y))
            X1 = data[list(combo)].values
            X = np.column_stack((X0, X1))
            
            betas = np.linalg.solve(X.T @ X, X.T @ Y)
            
            X_future = df[list(combo)].iloc[idx].to_numpy()
            X_future = np.insert(X_future, 0, 1)
            
            forecast = X_future @ betas
            forecasts.append(forecast)
        
        res.append(np.mean(forecasts))
    
    res_series = pd.Series(res, index=temp_series.index)
    return res_series

def standard_eigenvector_approach_combination(df, columns, days):
    real = df[REAL].values
    T = 24 * days
    res = []
    temp_series = df[REAL][T+24: len(real)]
    
    for idx in range(T+24, len(real)):
        data = df.iloc[idx-T-24 : idx-24]
        Y = data[REAL].values
        
        errors = data[columns].subtract(Y, axis=0) # ostensibly there should be a mean squared error but I don't see how that should work
        squared_errors = np.square(errors) # squared errors give much better results
        mspe_matrix = np.cov(squared_errors, rowvar=False)
        
        eigenvalues, eigenvectors = np.linalg.eig(mspe_matrix)
        
        min_index = np.argmin(eigenvalues)
        weights = eigenvectors[:, min_index]
        weights = np.abs(weights)
        # weights = weights ** 2 # condition from C. Hsiao, S.K. Wan - 3rd page
        weights /= weights.sum()

        
        X_future = df[columns].iloc[idx].to_numpy()
        combined_forecast = X_future @ weights
        
        res.append(combined_forecast)
    
    res_series = pd.Series(res, index=temp_series.index)
    return res_series

def biased_corrected_eigenvector_approach_combination(df, columns, days):
    real = df[REAL].values
    T = 24 * days
    res = []
    temp_series = df[REAL][T+24: len(real)]
    for idx in range(T+24, len(real)):
        data = df.iloc[idx-T-24 : idx-24]
        Y = data[REAL].values
        
        errors = data[columns].subtract(Y, axis=0)
        squared_errors = np.square(errors)
        
        mspe_matrix = np.cov(squared_errors, rowvar=False)
        column_means = np.mean(mspe_matrix, axis=0)
        centered_mspe_matrix = mspe_matrix - column_means[:, None]
        
        eigenvalues, eigenvectors = np.linalg.eig(centered_mspe_matrix)
        
        min_index = np.argmin(eigenvalues)
        weights = eigenvectors[:, min_index]
        weights = np.abs(weights)
        # weights = weights ** 2 # condition from C. Hsiao, S.K. Wan - 3rd page
        weights /= weights.sum()
        
        X_future = df[columns].iloc[idx].to_numpy()
        combined_forecast = X_future @ weights
        
        res.append(combined_forecast)
    
    res_series = pd.Series(res, index=temp_series.index)
    return res_series

def trimmed_eigenvector_combination(df, columns, days, lam):
    real = df[REAL].values
    T = 24 * days
    res = []
    temp_series = df[REAL][T+24: len(real)]
    
    for idx in range(T+24, len(real)):
        data = df.iloc[idx-T-24 : idx-24]
        Y = data[REAL].values

        rmses = {col: calculate_rmse(Y, data[col].values) for col in columns}
        sorted_models = sorted(rmses, key=rmses.get, reverse=True)
        cutoff = int(len(sorted_models) * lam)
        trimmed_columns = sorted_models[cutoff:]

        errors = data[trimmed_columns].subtract(Y, axis=0)
        squared_errors = np.square(errors)
        mspe_matrix = np.cov(squared_errors, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eig(mspe_matrix)
        min_index = np.argmin(eigenvalues)
        weights = eigenvectors[:, min_index]
        weights = np.abs(weights) / np.sum(np.abs(weights))
        
        X_future = df[trimmed_columns].iloc[idx].to_numpy()
        combined_forecast = X_future @ weights
        
        res.append(combined_forecast)
    
    res_series = pd.Series(res, index=temp_series.index)
    return res_series

def trimmed_bias_corrected_eigenvector_combination(df, columns, days, lam):
    real = df[REAL].values
    T = 24 * days
    res = []
    temp_series = df[REAL][T+24: len(real)]
    
    for idx in range(T+24, len(real)):
        data = df.iloc[idx-T-24 : idx-24]
        Y = data[REAL].values

        rmses = {col: calculate_rmse(Y, data[col].values) for col in columns}
        sorted_models = sorted(rmses, key=rmses.get, reverse=True)
        cutoff = int(len(sorted_models) * lam)
        trimmed_columns = sorted_models[cutoff:]

        errors = data[trimmed_columns].subtract(Y, axis=0)
        squared_errors = np.square(errors)
        mspe_matrix = np.cov(squared_errors, rowvar=False)

        column_means = np.mean(mspe_matrix, axis=0)
        centered_mspe_matrix = mspe_matrix - column_means[None, :]

        eigenvalues, eigenvectors = np.linalg.eig(centered_mspe_matrix)
        min_index = np.argmin(eigenvalues)
        weights = eigenvectors[:, min_index]
        weights = np.abs(weights) / np.sum(np.abs(weights))
        
        X_future = df[trimmed_columns].iloc[idx].to_numpy()
        combined_forecast = X_future @ weights
        
        res.append(combined_forecast)
    
    res_series = pd.Series(res, index=temp_series.index)
    return res_series