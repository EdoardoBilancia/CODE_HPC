import numpy as np
from pactools.grid_search import GridSearchCVProgressBar
import MLfunctions
import scipy.io
import os
import sys
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score,cross_validate
import lightgbm as lgb
from sklearn.decomposition import PCA
import neurokit2 as nk

def find_respiration_peaks(rsp,fs):
    cleaned = nk.rsp_clean(rsp, sampling_rate=fs)
    df, peaks_dict = nk.rsp_peaks(cleaned) 
    info = nk.rsp_fixpeaks(peaks_dict)
    return info["RSP_Peaks"],cleaned


def Nested_CV (X,feat_names, y, model, param_grid, outer_kfolds, inner_kfolds, scoring):
    # Outer loop
    outer_scores = []
    outer_best_params = []
    reg = GridSearchCVProgressBar(model, param_grid=param_grid, cv=inner_kfolds, verbose=10, scoring=scoring, refit=True)
    #nested_score = cross_val_score(reg, X=X, y=np.mean(Y, axis=1), cv=outer_cv)
    #calculate nested_score but in a for loop using always outer_cv
    nested_score = []
    for train_index, test_index in outer_kfolds.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        reg.fit(X_train, y_train,feature_name=feat_names)
        nested_score.append(reg.score(X_test, y_test))
    nested_score = np.array(nested_score)

    return reg, nested_score

def Nested_CV_sklearn(X,feat_names, y, model, param_grid, outer_kfolds, inner_kfolds, scoring):
    reg = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_kfolds,scoring=scoring, verbose=10)
    nested_score = cross_validate(reg, X=X, y=y, cv=outer_kfolds, return_estimator=True)
    mean_score = []
    std_score = []
    grid_search_cv_object = nested_score["estimator"]
    for j in range(outer_kfolds.n_splits):
        mean_score.append([])
        std_score.append([])
        for i in range(len(grid_search_cv_object[j].cv_results_["params"])):
            CV_results = grid_search_cv_object[j].cv_results_
            mean_score[j].append(CV_results['mean_test_score'][i])
            std_score[j].append(CV_results["std_test_score"][i])

    mean_mean_score = np.mean(mean_score,axis=0)
    mean_std_score = np.mean(std_score,axis=0)
    best_model_overall_idx = np.argmax(mean_std_score)
    best_params = nested_score["estimator"][0].cv_results_["params"][best_model_overall_idx]
    LGB_regr = lgb.LGBMRegressor(verbose = -1,random_state=0,objective="regression", scoring="r2")
    LGB_regr = LGB_regr.set_params(**best_params)
    LGB_regr.fit(X, y,feature_name=feat_names)
    return LGB_regr, mean_mean_score[best_model_overall_idx],mean_std_score[best_model_overall_idx]

def CV (X,feat_names, y, model, param_grid, cv, scoring):
    # Outer loop
    reg = GridSearchCVProgressBar(model, param_grid=param_grid, cv=cv, verbose=10, scoring=scoring, refit=True)
    reg.fit(X, y,feature_name=feat_names)
    return reg

def apply_PCA(X,X_test,n_components):
    #apply PCA to the data
    pca = PCA(n_components=n_components)
    pca.fit(X)
    X = pca.transform(X)
    X_test = pca.transform(X_test)
    print("Explained variance: " + str(pca.explained_variance_ratio_))
    print("Number of components: " + str(pca.n_components_))
    return X,X_test


def non_uniform_sampling(t,data,win,fs,r_peaks,K):
    subportion_0 = np.array(data[0:int(win*fs)]).squeeze()
    t_subportion_0 = np.array(t[0:int(win*fs)]).squeeze()
    central_point = np.mean(t_subportion_0)
    new_data = [subportion_0]
    new_t = [t_subportion_0]
    max_dist = np.max(np.diff(r_peaks))
    print("Number of samples: " + str(len(subportion_0)))
    while central_point + (win/2) < t[-100]:
        #find the closest peak after the central point
        dist = np.min(np.abs(r_peaks - central_point))
        ovelap_percentage = (dist/max_dist)*100
        next_t = new_t[-1][0] + ((ovelap_percentage)*win)/(100*K)
        subportion = np.array(data[int(next_t*fs)+3:int((next_t + win)*fs)+3]).squeeze()
        t_subportion = np.array(t[int(next_t*fs)+3:int((next_t + win)*fs)+3]).squeeze()
        if len(subportion) < len(subportion_0):
            subportion = np.array(data[int(next_t*fs)+3:int((next_t + win)*fs)+4]).squeeze()
            t_subportion = np.array(t[int(next_t*fs)+3:int((next_t + win)*fs)+4]).squeeze()
        if len(subportion) > len(subportion_0):
            subportion = subportion[0:len(subportion_0)]
            t_subportion = t_subportion[0:len(subportion_0)]
        new_data.append(subportion)
        new_t.append(t_subportion)
        central_point = np.mean(t_subportion)
        #print("Number of samples: " + str(len(subportion)))
    return np.array(new_data),np.array(new_t)




def load_and_preprocess(subject,times):
    data = {}
    directory = 'C:/Users\\edoar\\OneDrive\\Desktop\\TESI\\codice2\\02.macefield_10_2022_mat'
    for file in os.listdir(directory):
        if subject in file:
            with open(directory +"\\" +file, 'rb') as myfile:
                data[subject] = (scipy.io.loadmat(myfile))
    print(subject + " loaded" + "\nIntervals: " + str(times))
    #preprocess data
    global SB_data,fs      
    fs = int(data[subject]['fs_uNG'][0])     
    SB_data = {}
    h=0
    global train_start, train_end, test_start, test_end
    train_start = times[0][0]
    train_end = times[0][1]
    test_start = times[1][0]
    test_end = times[1][1]      
    for intervals in times:
        h +=1
        #filtered_array = np.array(filter(lambda row: intervals[0]<row[0]<intervals[1], data[subj]['ECG_peaks_times']))
        b = data[subject]['ECG_peaks_times'][:,0]
        condition = (b>int(intervals[0])) & (b<int(intervals[1]))
        intervals[0] = int(float(intervals[0]))
        intervals[1] = int(float(intervals[1]))
        data_uNG = data[subject]['data_uNG'][intervals[0]*int(data[subject]['fs_uNG']):intervals[1]*int(data[subject]['fs_uNG'])]
        t_uNG = data[subject]['t_uNG'][intervals[0]*int(data[subject]['fs_uNG']):intervals[1]*int(data[subject]['fs_uNG'])] -int(intervals[0])
        fs = int(data[subject]['fs_uNG'][0])
        nyquist = 0.5 * fs
        b, a = signal.butter(2, 300/nyquist, btype="highpass")
        data_uNG = signal.filtfilt(b, a, np.array(data_uNG).ravel())
        print("High pass filtered uNG")
        data_ung_filt = data_uNG
        #RMS_002 = MLfunctions.get_RMS(np.array(data_uNG).flatten(),data[subject]['t_uNG'],0.002)
        #RMS_005 = MLfunctions.get_RMS(np.array(data_uNG).flatten(),data[subject]['t_uNG'],0.005)
        #RMS_010 = MLfunctions.get_RMS(np.array(data_uNG).flatten(),data[subject]['t_uNG'],0.010)
        #RMS_020 = MLfunctions.get_RMS(np.array(data_uNG).flatten(),data[subject]['t_uNG'],0.020)
        RMS_040 = MLfunctions.get_RMS(np.array(data_uNG).flatten(),data[subject]['t_uNG'],0.040)
        RMS_100 = MLfunctions.get_RMS(np.array(data_uNG).flatten(),data[subject]['t_uNG'],0.100)
        RMS_200 = MLfunctions.get_RMS(np.array(data_uNG).flatten(),data[subject]['t_uNG'],0.200)
        RMS_400 = MLfunctions.get_RMS(np.array(data_uNG).flatten(),data[subject]['t_uNG'],0.400)
        RMS_800 = MLfunctions.get_RMS(np.array(data_uNG).flatten(),data[subject]['t_uNG'],0.800)
        RMS_1500 = MLfunctions.get_RMS(np.array(data_uNG).flatten(),data[subject]['t_uNG'],1.500)
        print("RMS calculated")

        s = data[subject]["data_resp"][intervals[0]*int(data[subject]['fs_resp']):intervals[1]*int(data[subject]['fs_resp'])].ravel()
        r = data[subject]["t_resp"][intervals[0]*int(data[subject]['fs_resp']):intervals[1]*int(data[subject]['fs_resp'])].ravel() -int(intervals[0])
        interp = interpolate.interp1d(r,s,bounds_error=None,fill_value = "extrapolate")
        data_resp = interp(t_uNG).flatten()

        s = data[subject]["data_resp_therm"][intervals[0]*int(data[subject]['fs_resp_therm']):intervals[1]*int(data[subject]['fs_resp_therm'])].ravel()
        r = data[subject]["t_resp_therm"][intervals[0]*int(data[subject]['fs_resp_therm']):intervals[1]*int(data[subject]['fs_resp_therm'])].ravel() -int(intervals[0])
        interp = interpolate.interp1d(r,s,bounds_error=None,fill_value = "extrapolate")
        data_resp_therm = interp(t_uNG).flatten()

        s = data[subject]["data_ECG"][intervals[0]*int(data[subject]['fs_ECG']):intervals[1]*int(data[subject]['fs_ECG'])].ravel()
        r = data[subject]["t_ECG"][intervals[0]*int(data[subject]['fs_ECG']):intervals[1]*int(data[subject]['fs_ECG'])].ravel() -int(intervals[0])
        interp = interpolate.interp1d(r,s,bounds_error=None,fill_value = "extrapolate")
        data_ECG = interp(t_uNG).flatten()

        s = data[subject]["data_ECG_HR"][intervals[0]*int(data[subject]['fs_ECG']):intervals[1]*int(data[subject]['fs_ECG'])].ravel()
        interp = interpolate.interp1d(r,s,bounds_error=None,fill_value = "extrapolate")
        data_ECG_HR = interp(t_uNG).flatten()

        print("Interpolation done")
        peaks_therm,cleaned_resp_therm = find_respiration_peaks(data_resp_therm,fs)
        peaks_neg_therm, _ = find_respiration_peaks(-data_resp_therm,fs)

        peaks_belt,cleaned_resp_belt = find_respiration_peaks(data_resp,fs)
        peaks_neg_belt, _ = find_respiration_peaks(-data_resp,fs)

        b_RMS, a_RMS = signal.butter(2, 0.040/nyquist, btype="highpass")
        #RMS_002 = signal.filtfilt(b_RMS, a_RMS, RMS_002)
        #RMS_005 = signal.filtfilt(b_RMS, a_RMS, RMS_005)
        #RMS_010 = signal.filtfilt(b_RMS, a_RMS, RMS_010)
        #RMS_020 = signal.filtfilt(b_RMS, a_RMS, RMS_020)
        RMS_040 = signal.filtfilt(b_RMS, a_RMS, RMS_040)
        RMS_100 = signal.filtfilt(b_RMS, a_RMS, RMS_100)
        RMS_200 = signal.filtfilt(b_RMS, a_RMS, RMS_200)
        RMS_400 = signal.filtfilt(b_RMS, a_RMS, RMS_400)
        RMS_800 = signal.filtfilt(b_RMS, a_RMS, RMS_800)
        #RMS_1500 = signal.filtfilt(b_RMS, a_RMS, RMS_1500)
        print("RMS filtered")

        b_resp,a_resp = signal.butter(2, np.array([0.040 , 40])/nyquist, btype="bandpass")
        data_resp = signal.filtfilt(b_resp, a_resp, data_resp)
        data_resp_therm = signal.filtfilt(b_resp, a_resp, data_resp_therm)

        SB_data[subject + "_" + str(h)] = {
            'data_uNG' : data_uNG,
            'data_uNG_RMS' : data[subject]['data_uNG_RMS'][intervals[0]*int(data[subject]['fs_uNG']):intervals[1]*int(data[subject]['fs_uNG'])],
            'data_ECG' :data_ECG,
            'data_ECG_HR' : data_ECG_HR*60,
            'data_BP' : data[subject]['data_BP'][intervals[0]*int(data[subject]['fs_BP']):intervals[1]*int(data[subject]['fs_BP'])],
            'data_resp' : data_resp,
            'data_resp_therm' : data_resp_therm,
            'metadata' : data[subject]['metadata'],
            'cleaned_resp' : cleaned_resp_belt,
            'cleaned_resp_therm' : cleaned_resp_therm,
            #'RMS_002' : RMS_002,
            #'RMS_005' : RMS_005,
            #'RMS_010' : RMS_010,
            #'RMS_020' : RMS_020,
            'RMS_040' : RMS_040,
            'RMS_100' : RMS_100,
            'RMS_200' : RMS_200,
            'RMS_400' : RMS_400,
            'RMS_800' : RMS_800,
            'RMS_1500' : RMS_1500,
            #'ECG_peaks_times' : new_array,
            'resp_peaks' : peaks_belt,
            'resp_therm_peaks' : peaks_therm,
            'resp_neg_peaks' : peaks_neg_belt,
            'resp_neg_therm_peaks' : peaks_neg_therm,
            't_uNG' : t_uNG,
            't_ECG' :  t_uNG,
            't_BP' : data[subject]['t_BP'][intervals[0]*int(data[subject]['fs_BP']):intervals[1]*int(data[subject]['fs_BP'])] -int(intervals[0]),
            't_resp' : t_uNG,
            't_resp_therm' : t_uNG,
            'fs_uNG' : data[subject]['fs_uNG'],
            'fs_ECG' : data[subject]['fs_uNG'],
            'fs_BP' : data[subject]['fs_BP'],
            'fs_resp' : data[subject]['fs_uNG'],
            'fs_resp_therm' : data[subject]['fs_uNG'],
            'dt_uNG' : data[subject]['dt_uNG'],
            'dt_ECG': data[subject]['dt_ECG'],
            'dt_BP' : data[subject]['dt_BP'],
            'dt_resp' : data[subject]['dt_resp'],
            'dt_resp_therm' : data[subject]['dt_resp_therm']
                        }
    return SB_data,fs

def load_and_preprocess_HR(subject,times,directory):
    data = {}
    for file in os.listdir(directory):
        if subject in file:
            with open(directory +"\\" +file, 'rb') as myfile:
                data[subject] = (scipy.io.loadmat(myfile))
    print(subject + " loaded" + "\nIntervals: " + str(times))
    #preprocess data
    global SB_data,fs      
    fs = int(data[subject]['fs_uNG'][0])     
    SB_data = {}
    h=0
    global train_start, train_end, test_start, test_end
    train_start = times[0][0]
    train_end = times[0][1]
    test_start = times[1][0]
    test_end = times[1][1]      
    for intervals in times:
        h +=1
        #filtered_array = np.array(filter(lambda row: intervals[0]<row[0]<intervals[1], data[subj]['ECG_peaks_times']))
        b = data[subject]['ECG_peaks_times'][:,0]
        condition = (b>int(intervals[0])) & (b<int(intervals[1]))
        intervals[0] = int(float(intervals[0]))
        intervals[1] = int(float(intervals[1]))
        data_uNG = data[subject]['data_uNG'][intervals[0]*int(data[subject]['fs_uNG']):intervals[1]*int(data[subject]['fs_uNG'])]
        t_uNG = data[subject]['t_uNG'][intervals[0]*int(data[subject]['fs_uNG']):intervals[1]*int(data[subject]['fs_uNG'])] -int(intervals[0])
        fs = int(data[subject]['fs_uNG'][0])
        nyquist = 0.5 * fs
        b, a = signal.butter(2, 300/nyquist, btype="highpass")
        data_uNG = signal.filtfilt(b, a, np.array(data_uNG).ravel())
        print("High pass filtered uNG")
        data_ung_filt = data_uNG
        #RMS_002 = MLfunctions.get_RMS(np.array(data_uNG).flatten(),data[subject]['t_uNG'],0.002)
        #RMS_005 = MLfunctions.get_RMS(np.array(data_uNG).flatten(),data[subject]['t_uNG'],0.005)
        #RMS_010 = MLfunctions.get_RMS(np.array(data_uNG).flatten(),data[subject]['t_uNG'],0.010)
        RMS_020 = MLfunctions.get_RMS(np.array(data_uNG).flatten(),data[subject]['t_uNG'],0.020)
        RMS_040 = MLfunctions.get_RMS(np.array(data_uNG).flatten(),data[subject]['t_uNG'],0.040)
        RMS_100 = MLfunctions.get_RMS(np.array(data_uNG).flatten(),data[subject]['t_uNG'],0.100)
        RMS_200 = MLfunctions.get_RMS(np.array(data_uNG).flatten(),data[subject]['t_uNG'],0.200)
        RMS_400 = MLfunctions.get_RMS(np.array(data_uNG).flatten(),data[subject]['t_uNG'],0.400)
        RMS_800 = MLfunctions.get_RMS(np.array(data_uNG).flatten(),data[subject]['t_uNG'],0.800)
        RMS_1500 = MLfunctions.get_RMS(np.array(data_uNG).flatten(),data[subject]['t_uNG'],1.500)
        RMS_3000 = MLfunctions.get_RMS(np.array(data_uNG).flatten(),data[subject]['t_uNG'],3.000)
        print("RMS calculated")

        s = data[subject]["data_resp"][intervals[0]*int(data[subject]['fs_resp']):intervals[1]*int(data[subject]['fs_resp'])].ravel()
        r = data[subject]["t_resp"][intervals[0]*int(data[subject]['fs_resp']):intervals[1]*int(data[subject]['fs_resp'])].ravel() -int(intervals[0])
        interp = interpolate.interp1d(r,s,bounds_error=None,fill_value = "extrapolate")
        data_resp = interp(t_uNG).flatten()

        s = data[subject]["data_resp_therm"][intervals[0]*int(data[subject]['fs_resp_therm']):intervals[1]*int(data[subject]['fs_resp_therm'])].ravel()
        r = data[subject]["t_resp_therm"][intervals[0]*int(data[subject]['fs_resp_therm']):intervals[1]*int(data[subject]['fs_resp_therm'])].ravel() -int(intervals[0])
        interp = interpolate.interp1d(r,s,bounds_error=None,fill_value = "extrapolate")
        data_resp_therm = interp(t_uNG).flatten()

        s = data[subject]["data_ECG"][intervals[0]*int(data[subject]['fs_ECG']):intervals[1]*int(data[subject]['fs_ECG'])].ravel()
        r = data[subject]["t_ECG"][intervals[0]*int(data[subject]['fs_ECG']):intervals[1]*int(data[subject]['fs_ECG'])].ravel() -int(intervals[0])
        interp = interpolate.interp1d(r,s,bounds_error=None,fill_value = "extrapolate")
        data_ECG = interp(t_uNG).flatten()

        s = data[subject]["data_ECG_HR"][intervals[0]*int(data[subject]['fs_ECG']):intervals[1]*int(data[subject]['fs_ECG'])].ravel()
        interp = interpolate.interp1d(r,s,bounds_error=None,fill_value = "extrapolate")
        data_ECG_HR = interp(t_uNG).flatten()

        s = data[subject]["data_BP"][intervals[0]*int(data[subject]['fs_BP']):intervals[1]*int(data[subject]['fs_BP'])].ravel()
        r = data[subject]["t_BP"][intervals[0]*int(data[subject]['fs_BP']):intervals[1]*int(data[subject]['fs_BP'])].ravel() -int(intervals[0])
        interp = interpolate.interp1d(r,s,bounds_error=None,fill_value = "extrapolate")
        data_BP = interp(t_uNG).flatten()

        print("Interpolation done")
        #peaks_therm,cleaned_resp_therm = find_respiration_peaks(data_resp_therm,fs)
        #peaks_neg_therm, _ = find_respiration_peaks(-data_resp_therm,fs)

        #peaks_belt,cleaned_resp_belt = find_respiration_peaks(data_resp,fs)
        #peaks_neg_belt, _ = find_respiration_peaks(-data_resp,fs)

        b_RMS, a_RMS = signal.butter(2, np.array([0.5 , 5])/nyquist, btype="bandpass")
        #RMS_002 = signal.filtfilt(b_RMS, a_RMS, RMS_002)
        #RMS_005 = signal.filtfilt(b_RMS, a_RMS, RMS_005)
        #RMS_010 = signal.filtfilt(b_RMS, a_RMS, RMS_010)
        RMS_020 = signal.filtfilt(b_RMS, a_RMS, RMS_020)
        RMS_040 = signal.filtfilt(b_RMS, a_RMS, RMS_040)
        RMS_100 = signal.filtfilt(b_RMS, a_RMS, RMS_100)
        RMS_200 = signal.filtfilt(b_RMS, a_RMS, RMS_200)
        RMS_400 = signal.filtfilt(b_RMS, a_RMS, RMS_400)
        RMS_800 = signal.filtfilt(b_RMS, a_RMS, RMS_800)
        RMS_1500 = signal.filtfilt(b_RMS, a_RMS, RMS_1500)
        RMS_3000 = signal.filtfilt(b_RMS, a_RMS, RMS_3000)
        print("RMS filtered")

        b_resp,a_resp = signal.butter(2, np.array([0.040 , 40])/nyquist, btype="bandpass")
        data_resp = signal.filtfilt(b_resp, a_resp, data_resp)
        data_resp_therm = signal.filtfilt(b_resp, a_resp, data_resp_therm)

        new_array = data[subject]['ECG_peaks_times'][condition] -int(intervals[0])

        SB_data[subject + "_" + str(h)] = {
            'data_uNG' : data_uNG,
            'data_uNG_RMS' : data[subject]['data_uNG_RMS'][intervals[0]*int(data[subject]['fs_uNG']):intervals[1]*int(data[subject]['fs_uNG'])],
            'data_ECG' :data_ECG,
            'data_ECG_HR' : data_ECG_HR*60,
            'data_BP' : data_BP,
            'data_resp' : data_resp,
            'data_resp_therm' : data_resp_therm,
            'metadata' : data[subject]['metadata'],
            #'cleaned_resp' : cleaned_resp_belt,
            #'cleaned_resp_therm' : cleaned_resp_therm,
            #'RMS_002' : RMS_002,
            #'RMS_005' : RMS_005,
            #'RMS_010' : RMS_010,
            'RMS_020' : RMS_020,
            'RMS_040' : RMS_040,
            'RMS_100' : RMS_100,
            'RMS_200' : RMS_200,
            'RMS_400' : RMS_400,
            'RMS_800' : RMS_800,
            'RMS_1500' : RMS_1500,
            'RMS_3000' : RMS_3000,
            'ECG_peaks_times' : new_array,
            #'resp_peaks' : peaks_belt,
            #'resp_therm_peaks' : peaks_therm,
            #'resp_neg_peaks' : peaks_neg_belt,
            #'resp_neg_therm_peaks' : peaks_neg_therm,
            't_uNG' : t_uNG,
            't_ECG' :  t_uNG,
            't_BP' : data[subject]['t_BP'][intervals[0]*int(data[subject]['fs_BP']):intervals[1]*int(data[subject]['fs_BP'])] -int(intervals[0]),
            't_resp' : t_uNG,
            't_resp_therm' : t_uNG,
            'fs_uNG' : data[subject]['fs_uNG'],
            'fs_ECG' : data[subject]['fs_uNG'],
            'fs_BP' : data[subject]['fs_BP'],
            'fs_resp' : data[subject]['fs_uNG'],
            'fs_resp_therm' : data[subject]['fs_uNG'],
            'dt_uNG' : data[subject]['dt_uNG'],
            'dt_ECG': data[subject]['dt_ECG'],
            'dt_BP' : data[subject]['dt_BP'],
            'dt_resp' : data[subject]['dt_resp'],
            'dt_resp_therm' : data[subject]['dt_resp_therm']
                        }
    return SB_data,fs







def remove_outliers(features,times,target,M):
    #features is a 2d matrix, from that compute the IQR of each feature and detect the outliers
    #times is the time vector
    #target is the target vector
    #M is the number of standard deviations from the mean
    #return the new features and target
    #remove outliers
    Q1 = np.percentile(features, 25, axis=0)
    Q3 = np.percentile(features, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - M*IQR
    upper_bound = Q3 + M*IQR
    #would like to remove only if more then 5 features are outliers

    num_outliers = np.sum((features < lower_bound) | (features > upper_bound), axis=1)
    mask = num_outliers < 5
    print("Number of observations: " + str(len(features)))
    print("Number of observations removed: " + str(len(features) - np.sum(mask)))
    features = features[mask]
    target = target[mask]
    times = times[mask]
    return features, times, target, mask


def extrapolate_peak_freq(data, fs):
    # Compute the power spectrum of the signal
    freq, power = signal.periodogram(data, fs=fs)

    # Find the index of the peak in the power spectrum
    power = power[freq > 0.08]
    freq = freq[freq > 0.08]
    peak_index = np.argmax(power)

    # Extrapolate the peak using quadratic interpolation
    if peak_index > 0 and peak_index < len(power) - 1:
        prev_power = power[peak_index - 1]
        curr_power = power[peak_index]
        next_power = power[peak_index + 1]

        # Compute the interpolated frequency
        interpolated_freq = freq[peak_index] + 0.5 * (next_power - prev_power) / (2 * curr_power - next_power - prev_power)

        return interpolated_freq
    else:
        return freq[peak_index]
    

def metrics_resp(concat_Y_test,concat_X_test,interpolated_predicts,concat_T_test,fs,target_test,predicts):

    peaks_real = find_respiration_peaks(concat_Y_test,fs)[0]
    results_half = signal.peak_widths(concat_Y_test, peaks_real, rel_height=0.4)
    results_full = signal.peak_widths(concat_Y_test, peaks_real, rel_height=0.8)

    peaks_predicted = find_respiration_peaks(interpolated_predicts.squeeze(),fs)[0]
    results_half_predicted = signal.peak_widths(interpolated_predicts.squeeze(), peaks_predicted, rel_height=0.4)
    results_full_predicted = signal.peak_widths(interpolated_predicts.squeeze(), peaks_predicted, rel_height=0.8)

    peaks_RMS = find_respiration_peaks(concat_X_test,fs)[0]
    results_half_RMS = signal.peak_widths(concat_X_test, peaks_RMS, rel_height=0.4)
    results_full_RMS = signal.peak_widths(concat_X_test, peaks_RMS, rel_height=0.8)
    if len(peaks_real) != len(peaks_predicted):
        print("Number of peaks real: " + str(len(peaks_real)))
        print("Number of peaks predicted: " + str(len(peaks_predicted)))
        print("Number of peaks RMS: " + str(len(peaks_RMS)))
    r2_error_test = 1 - r2_score(concat_Y_test, interpolated_predicts.squeeze())
    r2_error_test_no_interp = 1 - r2_score(target_test, predicts)
    rmse_test = np.sqrt(mean_squared_error(concat_Y_test, interpolated_predicts.squeeze()))
    rmse_test_no_interp = np.sqrt(mean_squared_error(target_test, predicts))
    if len(peaks_real) == len(peaks_predicted):
        peak_width_error = np.mean((abs(results_half[0] - results_half_predicted[0])) / fs).squeeze()
        peak_height_error = np.mean(abs(interpolated_predicts[peaks_predicted] - concat_X_test[peaks_real]))
        peak_distance_error = np.mean(abs(np.array(concat_T_test[peaks_predicted]) - np.array(concat_T_test[peaks_real])))
    else:
        peak_width_error = 0
        peak_height_error = 0
        peak_distance_error = 0

    r2_error_no_pred = 1 - r2_score(concat_Y_test, concat_X_test)
    r2_error_no_pred_no_interp = 1 - r2_score(concat_Y_test, concat_X_test)
    rmse_no_pred = np.sqrt(mean_squared_error(concat_Y_test,concat_X_test))
    rmse_no_pred_no_interp = np.sqrt(mean_squared_error(concat_Y_test, concat_X_test))
    if len(peaks_real) == len(peaks_RMS):
        peak_width_error_no_pred = np.mean((abs(results_half[0] - results_half_RMS[0]))/ fs).squeeze()
        peak_height_error_no_pred = np.mean(abs(concat_Y_test[peaks_real] - concat_X_test[peaks_RMS]))
        peak_distance_error_no_pred = np.mean(abs(np.array(concat_T_test[peaks_real]) - np.array(concat_T_test[peaks_RMS])))
    else:
        peak_width_error_no_pred = 0
        peak_height_error_no_pred = 0
        peak_distance_error_no_pred = 0

    data = {
    'Metrics with Predictor': ['R2 Error Score Test', 'R2 Error Score Test no Interpolation',
                               'RMSE Score Test', 'RMSE Score Test no Interpolation',
                               'Peak Width Error', 'Peak Height Error', 'Peak Distance Error'],
    'RMS as Predictor': [r2_error_no_pred, r2_error_no_pred_no_interp, rmse_no_pred, rmse_no_pred_no_interp,
                          peak_width_error_no_pred, peak_height_error_no_pred, peak_distance_error_no_pred],
    'Predictor': [r2_error_test, r2_error_test_no_interp, rmse_test, rmse_test_no_interp,
                  peak_width_error, peak_height_error, peak_distance_error]
    }
    df = pd.DataFrame(data)
    print(df)
    fig,ax = plt.subplots(figsize=(10,5))
    df.plot(kind='barh', x='Metrics with Predictor', y=['RMS as Predictor', 'Predictor'], figsize=(10, 5),rot = 30,ax=ax)
    ax.vlines(1,ax.get_ylim()[0],ax.get_ylim()[1],color="red",linestyles="dashed",label="Chance level")
    ax.legend(loc="upper right")

    fig,ax = plt.subplots(figsize=(10,5))
    ax.plot(concat_Y_test,label="Respiration")
    ax.plot(np.array(np.arange(0,len(concat_Y_test)-1,1))[peaks_real],concat_Y_test[peaks_real],"x",label= "real peaks")
    ax.hlines(*results_half[1:], color="C1")
    ax.hlines(*results_full[1:], color="C1")

    ax.plot(interpolated_predicts,label="Predicted respiration")
    ax.plot(np.array(np.arange(0,len(interpolated_predicts)-1,1))[peaks_predicted],interpolated_predicts[peaks_predicted],"o",label= "predicted peaks")
    ax.hlines(*results_half_predicted[1:], color="C2")
    ax.hlines(*results_full_predicted[1:], color="C2") 

    ax.plot(concat_X_test.squeeze(),label="RMS")
    ax.plot(np.array(np.arange(0,len(concat_X_test)-1,1))[peaks_RMS],concat_X_test[peaks_RMS],"v",label= "RMS peaks")
    ax.hlines(*results_half_RMS[1:], color="C3")
    ax.hlines(*results_full_RMS[1:], color="C3")
    plt.legend()

    fig,ax = plt.subplots(3,1,figsize=(10,10),sharex=True)
    ax[0].plot(concat_Y_test,label="Respiration")
    ax[0].plot(np.array(np.arange(0,len(concat_Y_test)-1,1))[peaks_real],concat_Y_test[peaks_real],"x",label= "real peaks")
    ax[0].hlines(*results_half[1:], color="C1")
    ax[0].hlines(*results_full[1:], color="C1")
    ax[0].legend()

    ax[1].plot(interpolated_predicts,label="Predicted respiration")
    ax[1].plot(np.array(np.arange(0,len(interpolated_predicts)-1,1))[peaks_predicted],interpolated_predicts[peaks_predicted],"o",label= "predicted peaks")
    ax[1].hlines(*results_half_predicted[1:], color="C2")
    ax[1].hlines(*results_full_predicted[1:], color="C2") 
    ax[1].legend()
    #ax[1].plot(interpolated_predicts_filt,label="Predicted respiration filtered")

    ax[2].plot(concat_X_test.squeeze(),label="RMS")
    ax[2].plot(np.array(np.arange(0,len(concat_X_test)-1,1))[peaks_RMS],concat_X_test[peaks_RMS],"v",label= "RMS peaks")
    ax[2].hlines(*results_half_RMS[1:], color="C3")
    ax[2].hlines(*results_full_RMS[1:], color="C3")
    ax[2].legend()

    fig,ax = plt.subplots(1,1,figsize=(10,10),sharex=True)
    ax.plot(concat_Y_test,label="Respiration")
    #ax.plot(interpolated_predicts_filt,label="Predicted respiration filtered")
    ax.legend()

    plt.show()



def normalize_Train_test(train,test):
    #normalize the train and test set
    mean = np.mean(train,axis=0)
    std = np.std(train,axis=0)
    train = (train - mean) / std
    test = (test - mean) / std
    return train,test