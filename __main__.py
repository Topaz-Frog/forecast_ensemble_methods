import methods as meth
import numpy as np
import pandas as pd
import pickle
import sys
import time
import warnings
# from multiprocessing import Pool
warnings.filterwarnings("ignore")

pd.options.display.float_format = '{:20,.6f}'.format

countries = ['NP', 'BE', 'DE', 'FR', 'PJM']
# countries = ['NP']

DNNs = ['DNN 1', 'DNN 2', 'DNN 3', 'DNN 4']
LEARs = ['LEAR 56', 'LEAR 84', 'LEAR 1092', 'LEAR 1456']
DNNs_and_LEARs = ['DNN 1', 'DNN 2', 'DNN 3', 'DNN 4', 'LEAR 56', 'LEAR 84', 'LEAR 1092', 'LEAR 1456']
all_columns = [DNNs, LEARs, DNNs_and_LEARs]
# all_columns = [DNNs_and_LEARs]
all_columns_names = ['DNNs', 'LEARs', 'DNNs_and_LEARs']
# all_columns_names = ['DNNs_and_LEARs']
windows = [1, 7, 14]
# windows = [1, 7]

def __combine__():
    all_forecasts = {}
    for country in countries:
        print()
        print('#######################################', country, '########################################')
        df = pd.read_csv(f'Forecasts_{country}_DNN_LEAR_ensembles.csv', index_col=0)
        # df = df.iloc[:10*24,:]
        real = df[meth.REAL].values
        ids = (len(real) - (24 * (max(windows) + 1))) * (-1)
        # ids = (len(real) - (24 * 7)) * (-1)
        country_forecasts = {}

        for column_idx in range(len(all_columns)):
            columns = all_columns[column_idx]
            max_cols = (len(columns) - 1) / len(columns)
            print('###################################################################################')
            print('########################################', all_columns_names[column_idx], '########################################')

            combination_results = {}

            ######################################## avg ####################################################
            start = time.time()
            combination_results['simple average'] = meth.simple_average_combination(df, columns)
            end = time.time()
            print(' simple average:', str(meth.calculate_rmse(real[ids:], combination_results['simple average'][ids:]))
                , '\t\t\t\t\ttime:', meth.format_elapsed_time(start, end))

            ######################################## median #################################################
            start = time.time()
            combination_results['median'] = meth.median_combination(df, columns)
            end = time.time()
            print('         median:', str(meth.calculate_rmse(real[ids:], combination_results['median'][ids:]))
                , '\t\t\t\t\ttime:', meth.format_elapsed_time(start, end))

            ##################################### trimmed mean ##############################################
            start = time.time()
            best_lam = 0.0
            best_rmse = float(sys.maxsize)
            trimmed_means = {}
            for lambda_trimmer in np.arange(1/len(columns), 0.5, 1/len(columns)):
                forecast = meth.trimmed_mean_combination(df, columns, lambda_trimmer)
                forecast_rmse = meth.calculate_rmse(real[ids:], forecast[ids:])
                trimmed_means[lambda_trimmer] = forecast
                if forecast_rmse < best_rmse:
                    best_lam = lambda_trimmer
                    best_rmse = forecast_rmse
            combination_results['trimmed mean'] = trimmed_means[best_lam]
            end = time.time()
            print('   trimmed mean:', str(meth.calculate_rmse(real[ids:], combination_results['trimmed mean'][ids:]))
                , '\t best lambda:', str(best_lam)
                , '\t\ttime:', meth.format_elapsed_time(start, end))
            
            with open(f'trimmed_means__{country}_{all_columns_names[column_idx]}', 'wb') as f:
                pickle.dump(trimmed_means, f)

            #################################### winsorized mean ###########################################
            start = time.time()
            best_lam = 0.0
            best_rmse = float(sys.maxsize)
            winsorized_means = {}
            for lambda_trimmer in np.arange(0, 0.5, 1/len(columns)):
                forecast = meth.winsorized_mean_combination(df, columns, lambda_trimmer)
                forecast_rmse = meth.calculate_rmse(real[ids:], forecast[ids:])
                winsorized_means[lambda_trimmer] = forecast
                if forecast_rmse < best_rmse:
                    best_rmse = forecast_rmse
                    best_lam = lambda_trimmer
            combination_results['winsorized mean'] = winsorized_means[best_key]
            end = time.time()
            print('winsorized mean:', str(meth.calculate_rmse(real[ids:], combination_results['winsorized mean'][ids:]))
                , '\t best lambda:', str(best_key)
                , '\ttime:', meth.format_elapsed_time(start, end))
            
            with open(f'winsorized_means__{country}_{all_columns_names[column_idx]}', 'wb') as f:
                pickle.dump(winsorized_means, f)

            #################################### bates / granger ###########################################
            start = time.time()
            best_rmse = float(sys.maxsize)
            best_window = 1
            bates_grangers = {}
            for days in windows:
                forecast = meth.bates_granger_combination(df, columns, days)
                forecast_rmse = meth.calculate_rmse(real[ids:], forecast[ids:])
                bates_grangers[days] = forecast
                if forecast_rmse < best_rmse:
                    best_window = days
                    best_rmse = forecast_rmse
            combination_results['bates / granger'] = bates_grangers[best_window]
            end = time.time()
            print('bates / granger:', str(meth.calculate_rmse(real[ids:], combination_results['bates / granger'][ids:]))
                , '\t best window:', str(best_window)
                , '\t\ttime:', meth.format_elapsed_time(start, end))
            
            with open(f'bates_grangers__{country}_{all_columns_names[column_idx]}', 'wb') as f:
                pickle.dump(bates_grangers, f)

            ####################################### inverse rank ###########################################
            start = time.time()
            best_rmse = float(sys.maxsize)
            best_window = 1
            inverse_ranks = {}
            for days in windows:
                forecast = meth.inverse_rank_combination(df, columns, days)
                forecast_rmse = meth.calculate_rmse(real[ids:], forecast[ids:])
                inverse_ranks[days] = forecast
                if forecast_rmse < best_rmse:
                    best_window = days
                    best_rmse = forecast_rmse
            combination_results['inverse rank'] = inverse_ranks[best_window]
            end = time.time()
            print('   inverse rank:', str(meth.calculate_rmse(real[ids:], combination_results['inverse rank'][ids:]))
                , '\t best window:', str(best_window)
                , '\t\ttime:', meth.format_elapsed_time(start, end))
            
            with open(f'inverse_ranks__{country}_{all_columns_names[column_idx]}', 'wb') as f:
                pickle.dump(inverse_ranks, f)

            ######################################## OLS ####################################################
            start = time.time()
            best_rmse = float(sys.maxsize)
            best_window = 1
            OLSs = {}
            for days in windows:
                forecast = meth.ordinary_least_squares_combination(df, columns, days)
                forecast_rmse = meth.calculate_rmse(real[ids:], forecast[ids:])
                OLSs[days] = forecast
                if forecast_rmse < best_rmse:
                    best_window = days
                    best_rmse = forecast_rmse
            combination_results['OLS'] = OLSs[best_window]
            end = time.time()
            print('            OLS:', str(meth.calculate_rmse(real[ids:], combination_results['OLS'][ids:]))
                , '\t best window:', str(best_window)
                , '\t\ttime:', meth.format_elapsed_time(start, end))
            
            with open(f'OLSs__{country}_{all_columns_names[column_idx]}', 'wb') as f:
                pickle.dump(OLSs, f)

            ######################################## LAD ####################################################
            start = time.time()
            best_rmse = float(sys.maxsize)
            best_window = 1
            LADs = {}
            for days in windows:
                forecast = meth.least_absolute_deviation(df, columns, days)
                forecast_rmse = meth.calculate_rmse(real[ids:], forecast[ids:])
                LADs[days] = forecast
                if forecast_rmse < best_rmse:
                    best_window = days
                    best_rmse = forecast_rmse
            combination_results['LAD'] = LADs[best_window]
            end = time.time()
            print('            LAD:', str(meth.calculate_rmse(real[ids:], combination_results['LAD'][ids:]))
                , '\t best window:', str(best_window)
                , '\t\ttime:', meth.format_elapsed_time(start, end))
            
            with open(f'LADs__{country}_{all_columns_names[column_idx]}', 'wb') as f:
                pickle.dump(LADs, f)

            ######################################### PW ####################################################
            start = time.time()
            best_rmse = float(sys.maxsize)
            best_window = 1
            PWs = {}
            for days in windows:
                forecast = meth.positive_weights_combination(df, columns, days)
                forecast_rmse = meth.calculate_rmse(real[ids:], forecast[ids:])
                PWs[days] = forecast
                if forecast_rmse < best_rmse:
                    best_window = days
                    best_rmse = forecast_rmse
            combination_results['PW'] = PWs[best_window]
            end = time.time()
            print('             PW:', str(meth.calculate_rmse(real[ids:], combination_results['PW'][ids:]))
                , '\t best window:', str(best_window)
                , '\t\ttime:', meth.format_elapsed_time(start, end))
            
            with open(f'PWs__{country}_{all_columns_names[column_idx]}', 'wb') as f:
                pickle.dump(PWs, f)

            ######################################## CLS ####################################################
            start = time.time()
            best_rmse = float(sys.maxsize)
            best_window = 1
            CLSs = {}
            for days in windows:
                forecast = meth.constrained_least_squares_combination(df, columns, days)
                forecast_rmse = meth.calculate_rmse(real[ids:], forecast[ids:])
                CLSs[days] = forecast
                if forecast_rmse < best_rmse:
                    best_window = days
                    best_rmse = forecast_rmse
            combination_results['CLS'] = CLSs[best_window]
            end = time.time()
            print('            CLS:', str(meth.calculate_rmse(real[ids:], combination_results['CLS'][ids:]))
                , '\t best window:', str(best_window)
                , '\t\ttime:', meth.format_elapsed_time(start, end))
            
            with open(f'CLSs__{country}_{all_columns_names[column_idx]}', 'wb') as f:
                pickle.dump(CLSs, f)

            ######################################## CSR ####################################################
            start = time.time()
            best_rmse = float(sys.maxsize)
            best_key = ""
            CSRs = {}
            for days in windows:
                for n in range(1, len(columns)):
                    forecast = meth.complete_subset_regression_combination(df, columns, days, n)
                    forecast_rmse = meth.calculate_rmse(real[ids:], forecast[ids:])
                    key = str(days) + " " + str(n)
                    CSRs[key] = forecast
                    if forecast_rmse < best_rmse:
                        best_key = key
                        best_rmse = forecast_rmse
            combination_results['CSR'] = CSRs[best_key]
            end = time.time()
            print('            CSR:', str(meth.calculate_rmse(real[ids:], combination_results['CSR'][ids:]))
                , '\t best window:', str(best_key)
                , '\t\ttime:', meth.format_elapsed_time(start, end))
            
            with open(f'CSRs__{country}_{all_columns_names[column_idx]}', 'wb') as f:
                pickle.dump(CSRs, f)

            ######################################## SEA ####################################################
            start = time.time()
            best_rmse = float(sys.maxsize)
            best_window = 1
            SEAs = {}
            for days in windows:
                forecast = meth.standard_eigenvector_approach_combination(df, columns, days)
                forecast_rmse = meth.calculate_rmse(real[ids:], forecast[ids:])
                SEAs[days] = forecast
                if forecast_rmse < best_rmse:
                    best_window = days
                    best_rmse = forecast_rmse
            combination_results['SEA'] = SEAs[best_window]
            end = time.time()
            print('            SEA:', str(meth.calculate_rmse(real[ids:], combination_results['SEA'][ids:]))
                , '\t best window:', str(best_window)
                , '\t\ttime:', meth.format_elapsed_time(start, end))
            
            with open(f'SEAs__{country}_{all_columns_names[column_idx]}', 'wb') as f:
                pickle.dump(SEAs, f)

            ######################################## BCEA ###################################################
            start = time.time()
            best_rmse = float(sys.maxsize)
            best_window = 1
            BCEAs = {}
            for days in windows:
                forecast = meth.biased_corrected_eigenvector_approach_combination(df, columns, days)
                forecast_rmse = meth.calculate_rmse(real[ids:], forecast[ids:])
                BCEAs[days] = forecast
                if forecast_rmse < best_rmse:
                    best_window = days
                    best_rmse = forecast_rmse
            combination_results['BCEA'] = BCEAs[best_window]
            end = time.time()
            print('           BCEA:', str(meth.calculate_rmse(real[ids:], combination_results['BCEA'][ids:]))
                , '\t best window:', str(best_window)
                , '\t\ttime:', meth.format_elapsed_time(start, end))
            
            with open(f'BCEAs__{country}_{all_columns_names[column_idx]}', 'wb') as f:
                pickle.dump(BCEAs, f)

            ######################################## TEA ####################################################
            start = time.time()
            best_rmse = float(sys.maxsize)
            best_key = ""
            lams = np.arange(0, max_cols, 1/len(columns))
            TEAs = {}
            for days in windows:
                for lam in lams:
                    forecast = meth.trimmed_eigenvector_combination(df, columns, days, lam)
                    forecast_rmse = meth.calculate_rmse(real[ids:], forecast[ids:])
                    key = str(days) + " " + str(lam)
                    TEAs[key] = forecast
                    if forecast_rmse < best_rmse:
                        best_key = key
                        best_rmse = forecast_rmse
            combination_results['TEA'] = TEAs[best_key]
            end = time.time()
            print('            TEA:', str(meth.calculate_rmse(real[ids:], combination_results['TEA'][ids:]))
                , '\t best window:', str(best_key)
                , '\t\ttime:', meth.format_elapsed_time(start, end))
            
            with open(f'TEAs__{country}_{all_columns_names[column_idx]}', 'wb') as f:
                pickle.dump(TEAs, f)

            ######################################## TBCEA ##################################################
            start = time.time()
            best_rmse = float(sys.maxsize)
            best_key = ""
            lams = np.arange(0, max_cols, 1/len(columns))
            TBCEAs = {}
            for days in windows:
                for lam in lams:
                    forecast = meth.trimmed_bias_corrected_eigenvector_combination(df, columns, days, lam)
                    forecast_rmse = meth.calculate_rmse(real[ids:], forecast[ids:])
                    key = str(days) + " " + str(lam)
                    TBCEAs[key] = forecast
                    if forecast_rmse < best_rmse:
                        best_key = key
                        best_rmse = forecast_rmse
            combination_results['TBCEA'] = TBCEAs[best_key]
            end = time.time()
            print('          TBCEA:', str(meth.calculate_rmse(real[ids:], combination_results['TBCEA'][ids:]))
                , '\t best window:', str(best_key)
                , '\t\ttime:', meth.format_elapsed_time(start, end))
            
            with open(f'TBCEAs__{country}_{all_columns_names[column_idx]}', 'wb') as f:
                pickle.dump(TBCEAs, f)

            country_forecasts[all_columns_names[column_idx]] = combination_results
        all_forecasts[country] = country_forecasts

    with open('all_forecasts', 'wb') as f:
        pickle.dump(all_forecasts, f)
    print('saved')

    return all_forecasts

def __load__():
    with open('all_forecasts', 'rb') as f:
        all_forecasts_loaded = pickle.load(f)
    return all_forecasts_loaded

def __prepare_mae_rmse_excel__(all_forecasts_loaded):
    created_combined = False

    for country in all_forecasts_loaded:
        for columns_name in all_forecasts_loaded[country]:
            temp_df = pd.read_csv(f'Forecasts_{country}_DNN_LEAR_ensembles.csv', index_col=0)
            temp_real = temp_df[meth.REAL].values
            ids = (len(temp_real) - (24 * (14 + 1))) * (-1)
            # print(country, columns_name)
            forecasts = all_forecasts_loaded[country][columns_name]
            forecasts_df = pd.DataFrame(forecasts)

            mae_values = {}

            for column in forecasts_df.columns:
                mae_values[column] = meth.calculate_mae(temp_real[ids:], forecasts_df[column][ids:])

            mae_df = pd.DataFrame([mae_values], index=[f'MAE {country} {columns_name}'])

            mae_df = mae_df.T
            mae_df['Rank MAE'] = mae_df[f'MAE {country} {columns_name}'].rank(method='dense', ascending=True)
            mae_df['Improve MAE'] = mae_df[f'MAE {country} {columns_name}'] / mae_df[f'MAE {country} {columns_name}'].iloc[[0]].values

            rmse_values = {}
            for column in forecasts_df.columns:
                rmse_values[column] = meth.calculate_rmse(temp_real[ids:], forecasts_df[column][ids:])
            rmse_df = pd.DataFrame([rmse_values], index=[f'RMSE {country} {columns_name}'])

            rmse_df = rmse_df.T
            rmse_df['Rank RMSE'] = rmse_df[f'RMSE {country} {columns_name}'].rank(method='dense', ascending=True)
            rmse_df['Improve RMSE'] = rmse_df[f'RMSE {country} {columns_name}'] / rmse_df[f'RMSE {country} {columns_name}'].iloc[[0]].values
            
            rmse_mae_df = pd.concat([mae_df, rmse_df], axis=1)

            if created_combined:
                combined_df = pd.concat([combined_df, rmse_mae_df], axis=1)
            else:
                combined_df = rmse_mae_df
                created_combined = True
                
    combined_df.to_excel('combined_df.xlsx')
    print(combined_df.head())

# __combine__()
all_forecasts_loaded = __load__()
__prepare_mae_rmse_excel__(all_forecasts_loaded)

# for country in all_forecasts_loaded:
#     for columns_name in all_forecasts_loaded[country]:
#         print(country, columns_name)
#         forecasts = all_forecasts_loaded[country][columns_name]
