#%%
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os
from scipy import signal
import sys
import logging
from scipy.integrate import simps
from scipy import interpolate
from scipy import stats
import sklearn.ensemble as ens
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score,cross_validate, RandomizedSearchCV, StratifiedKFold,TimeSeriesSplit
sys.path.append("support_scripts/")
import MLfunctions,MLfunctions_2
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna.integration.lightgbm as lgbopt
from sklearn.model_selection import train_test_split
import neurokit2 as nk

directory = '../../Data/MAT/'
data = {}

import logging    # first of all import the module

logging.basicConfig(filename='execution.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s')
logging.warning('Start of A new execution')


plt.style.use("seaborn-v0_8-colorblind")
plt.rcParams.update({'font.size': 12,'figure.autolayout': True})
#here we can define the segments
segment_time_SB = {
    #"MP130721" : [[1120,1295],[1400,1500]],
    #"SD110521 left vagus - insertion _ atrial_" : [[62,294],[295,336]],
    #"SD110521 left vagus - insertion _ insp act" : [[200,328],[329,369]],
    "SD110521 left vagus - insertion _ insp act" : [[100,240],[240,280]],
    #"SD110521 left vagus n- multiunit actvity" : [[0,400],[400,500]],
    #"LM 220721" : [[800,880],[880,900]], 
    #"LM 220721" : [[1250,1309],[1310,1350]], #bello!
    #"LM 220721" : [[2060,2110],[880,900]], #Brutto!
    #"LM 220721" : [[1435,1510],[880,900]], #Normal Breath only ma Bello! in pratica fase bella 
    #"LM 220721" : [[325,375],[880,900]], #questo è brutto
    #"EB 200521 right vagus" : [[545,679],[680,720]], #bellooo!!
    #"EB 200521 right vagus" : [[1045,1180],[120,240]] #bello!! #with electrode adjustments in the middle very similiar response
    #"EB 200521 right vagus" : [[1700,1900],[120,240]] #ancora molto bello
    #questi sono più brutti
    #"PC 150721 right vagus" :  [[410,540],[622,662]], #poche realizzazioni belle (a 0.4s)
    #"BT 230721 right vagus" : [[425,584],[585,625]],
    #"DR 130521 insp related 2" : [[70,159],[160,200]],
    #"DR 130521 insp related 1" : [[260,340],[341,381]]
    #"DR 130521 insp related 1" : [[940,1029],[1030,1070]]
    #"LC 170621 left vagus": [[120,240],[1030,1070]]
    #"LC 170621 left vagus": [[1070,1180],[120,240]]
    #"SD110521 left vagus n- multiunit actvity" : [[65,205],[120,240]],
    #"CC 240621 right vagus" : [[310,410],[120,240]],
    #"CC 240621 right vagus" : [[1195,1299],[1300,1340]]
    #"CC 240621 right vagus" : [[990,1060],[1061,1100]],
    #"TD270421 right vagus nerve" : [[405,505],[120,240]],
    #"TD270421 right vagus nerve" : [[1410,1490],[120,240]]
    #"TD270421 right vagus nerve" : [[1610,1710],[120,240]]
    #"AA 230621 left vagus" :  [[825,875],[120,240]],
    #"AA 230621 left vagus" :  [[970,1060],[1061,1100]],
    #"AA 230621 left vagus" :  [[1100,1210],[1211,1241]]
    #"AF 030821" : [[325,409],[410,450]],
    #"EF 110821" : [[640,779],[780,820]],
    #"EF 110821" : [[1855,2050]]
    #"RS 100821" : [[120,240],[241,281]],
    #'LS270721' : [[2560,2800],[2801,2900]]
}


first_key = list(segment_time_SB.keys())[0]
SB_data, fs = MLfunctions_2.load_and_preprocess_HR(str(first_key), segment_time_SB[first_key],directory)


sub = str(first_key)
subj = sub + "_1"
subj_test = sub + "_2"

uNG_types = ["RMS_040","RMS_400","RMS_1500"] #this will be the input
data_ECG = "data_ECG" #this will be the target

neurokit_fitler = False
window_size = 0.5  # Window size in seconds
overlap_percentage = 65  # Overlap percentage between windows
N_past = 15  # Number of past samples to consider for the features

logging.warning("subject: " + sub)
logging.warning("time interval: " + str(segment_time_SB[first_key]) + " seconds")
logging.warning("uNG_types: " + str(uNG_types))
logging.warning("window_size: " + str(window_size))
logging.warning("overlap_percentage: " + str(overlap_percentage))
logging.warning("N_past: " + str(N_past))


if neurokit_fitler:
    SB_data[subj][data_ECG] = nk.ecg_clean(SB_data[subj][data_ECG],sampling_rate=fs,method="biosppy")
    SB_data[subj_test][data_ECG] = nk.ecg_clean(SB_data[subj_test][data_ECG],sampling_rate=fs,method="biosppy")
    

#normalize data
for uNG_type in uNG_types:
    SB_data[subj][uNG_type],SB_data[subj_test][uNG_type] = MLfunctions_2.normalize_Train_test(SB_data[subj][uNG_type],SB_data[subj_test][uNG_type])
    SB_data[subj][data_ECG],SB_data[subj_test][data_ECG] = MLfunctions_2.normalize_Train_test(SB_data[subj][data_ECG],SB_data[subj_test][data_ECG])

#plot the data

fig, ax = plt.subplots(len(uNG_types), 1, figsize=(100, 30), sharex=True)
for i, uNG_type in enumerate(uNG_types):
    ax[i].plot(SB_data[subj]["t_uNG"], SB_data[subj][uNG_type], label=uNG_type,alpha = 0.5)
    ax[i].set_ylabel("Normalized amplitude")
    ax[i].spines[['top', 'right']].set_visible(False)
    ax[i].plot(SB_data[subj]["t_uNG"], SB_data[subj][data_ECG], label=data_ECG)
    ax[i].legend(loc="upper right")


plt.suptitle(str(segment_time_SB[first_key]) + " Post-processed data train " + sub)
plt.savefig("Input singals " + sub + ".png")

""" fig,ax = plt.subplots(2,1,figsize=(10,5))
ax[0].plot(SB_data[subj]["t_uNG"],SB_data[subj][data_ECG],label = data_ECG,zorder = 10)
for uNG_type in uNG_types:
    ax[0].plot(SB_data[subj]["t_uNG"],SB_data[subj][uNG_type],label = uNG_type,alpha = 0.5,zorder = 0)
ax[0].set_xlabel("Time [s]")
ax[0].set_ylabel("Normalized amplitude")
ax[0].legend(loc="upper right")
ax[0].set_title("Training set" )
ax[1].plot(SB_data[subj_test]["t_uNG"],SB_data[subj_test][data_ECG],label = data_ECG,zorder = 10)
for uNG_type in uNG_types:
    ax[1].plot(SB_data[subj_test]["t_uNG"],SB_data[subj_test][uNG_type],label = uNG_type,alpha = 0.5,zorder = 0)
ax[1].set_xlabel("Time [s]")
ax[1].set_ylabel("Normalized amplitude")
ax[1].legend(loc="upper right")
ax[1].set_title("Test set")
ax[0].spines[['top','right']].set_visible(False)
ax[1].spines[['top','right']].set_visible(False)
plt.suptitle(str(segment_time_SB[first_key]) + " Post-processed data " + sub)
plt.show() """

subportions_target = MLfunctions.non_overlapping_subportions(SB_data[subj][data_ECG], window_size, fs, overlap_percentage)
subportions_time = np.array(MLfunctions.non_overlapping_subportions(SB_data[subj]["t_uNG"], window_size, fs, overlap_percentage)).squeeze()
subportions_test_target = MLfunctions.non_overlapping_subportions(SB_data[subj_test][data_ECG], window_size, fs, overlap_percentage)
subportions_test_time = np.array(MLfunctions.non_overlapping_subportions(SB_data[subj_test]["t_uNG"], window_size, fs, overlap_percentage)).squeeze()
# Create subportions
features = {}
features_test = {}
for uNG_type in uNG_types:
    subportions_Input = MLfunctions.non_overlapping_subportions(SB_data[subj][uNG_type], window_size, fs, overlap_percentage)
    subportions_test_Input = MLfunctions.non_overlapping_subportions(SB_data[subj_test][uNG_type], window_size, fs, overlap_percentage)
    features[uNG_type] = np.array(MLfunctions.features_extractor_with_pasts(subportions_Input,N_past))
    features_test[uNG_type] = np.array(MLfunctions.features_extractor_with_pasts(subportions_test_Input,N_past))

# Extract features

features = np.hstack((features[uNG_types[0]],features[uNG_types[1]],features[uNG_types[2]]))
features_test = np.hstack((features_test[uNG_types[0]],features_test[uNG_types[1]],features_test[uNG_types[2]]))

for i in range(features.shape[1]):
    features[:,i],features_test[:,i] = MLfunctions_2.normalize_Train_test(features[:,i],features_test[:,i])


#feature_names_uNG = MLfunctions.features_extractor_with_pasts_names(9)
#feature_names_uNG = [name + "_uNG" for name in feature_names_uNG]
feature_names = {}
for names in uNG_types:
    feature_names[names] = MLfunctions.features_extractor_with_pasts_names(N_past)
    feature_names[names] = [name + "_" + names for name in feature_names[names]]

feature_names = np.hstack((feature_names[uNG_types[0]],feature_names[uNG_types[1]],feature_names[uNG_types[2]]))
feature_names = list(feature_names)
#in this case the target is the mean of the signal in every window

#target = np.array(np.mean(subportions_target,axis = 1))
#target_test = np.array(np.mean(subportions_test_target,axis = 1))

#in this case the target is the central point of the signal in every window

target = np.array([x[int(window_size * fs / 2)] for x in subportions_target])
target_test = np.array([x[int(window_size * fs / 2)] for x in subportions_test_target])

# Create a DataFrame from your vectors

#in this case the target is the max of the signal in every window (in this way i don't lose any peak)


#target = np.array(np.max(subportions_target,axis = 1))
#target_test = np.array(np.max(subportions_test_target,axis = 1))

#plot the target versus the ECG signal
fig,ax = plt.subplots(2,1,figsize=(10,5))
ax[0].plot(SB_data[subj]["t_uNG"],SB_data[subj][data_ECG],label = data_ECG)
ax[0].plot(np.mean(subportions_time,axis=1),target,label = "target_train")
ax[0].set_title("Training set")
ax[1].plot(SB_data[subj_test]["t_uNG"],SB_data[subj_test][data_ECG],label = data_ECG)
ax[1].plot(np.mean(subportions_test_time,axis=1),target_test,label = "target_test")
ax[1].set_title("Test set")
ax[0].legend()
ax[1].legend()
ax[0].set_xlabel("Time [s]")
ax[0].set_ylabel("Normalized amplitude")
ax[1].set_xlabel("Time [s]")
ax[1].set_ylabel("Normalized amplitude")
ax[0].spines[['top','right']].set_visible(False)
ax[1].spines[['top','right']].set_visible(False)
plt.suptitle("Target vs Respiration original signal")
plt.savefig("Target vs Respiration original signal " + sub + ".png")

concat_T,indexes_unique = np.unique(np.hstack(subportions_time),return_index=True)
concat_Y = np.hstack(subportions_target)[indexes_unique]
concat_X = np.hstack(subportions_Input)[indexes_unique]
concat_T_test,indexes_unique_test = np.unique(np.hstack(subportions_test_time),return_index=True)
concat_Y_test = np.hstack(subportions_test_target)[indexes_unique_test]
concat_X_test = np.hstack(subportions_test_Input)[indexes_unique_test]

#print("Number of features: ", features.shape[1])
#print("Number of observations: ", features.shape[0])
#print("Number of test observations: ", features_test.shape[0])

logging.warning("Number of features: " + str(features.shape[1]))
logging.warning("Number of observations Train: " + str(features.shape[0]))
logging.warning("Number of observations Test: " + str(features_test.shape[0]))


#%%

model= lgb.LGBMRegressor(verbose = -1,
                        random_state=0,
                        objective="regression", 
                        scoring="r2")

model.fit(features, target,feature_name=feature_names)

target_pred = model.predict(features_test)
print("R2: ",r2_score(target_test,target_pred))
print("RMSE: ",np.sqrt(mean_squared_error(target_test,target_pred)))
RMSE_score_unoptimized = np.sqrt(mean_squared_error(target_test,target_pred))
R2_score_unoptimized = r2_score(target_test,target_pred)
# %%

import optuna
inner_cv = KFold(n_splits=8, shuffle=True,random_state = 0)
outer_cv = KFold(n_splits=3, shuffle=True,random_state = 0)

# Random Forest
def create_model(trial):
    param = {
        "objective": "regression",
        "metric": "",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 6),
        "n_estimators": trial.suggest_int("n_estimators", 10, 250, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 1e-1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 2e-2, 8e-1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 2e-2, 8e-1, log=True),
        "max_bin": trial.suggest_int("max_bin", 16, 512),
        "num_leaves": trial.suggest_int("num_leaves", 5, 15),
        "min_split_gain": trial.suggest_float("min_split_gain", 5e-2, 1e-1, log=True),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1e-3, 1e-1, log=True),
        "n_jobs": -1,
        "random_state": 0,
        "verbose": -1,
    }


    LGB_regr = lgb.LGBMRegressor(random_state=0)
    return LGB_regr.set_params(**param)


def objective(trial, x_train , x_test , y_train , y_test):
    LGB_regr = create_model(trial)
    LGB_regr.fit(x_train, y_train,feature_name=feature_names)
    target_pred = LGB_regr.predict(x_test)
    return np.sqrt(mean_squared_error(y_test,target_pred))#r2_score(y_test,target_pred)


def objective_cv(trial):

    # Get the MNIST dataset.
    scores = []
    for train_index, test_index in inner_cv.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = target[train_index], target[test_index]

        RMSE_result = objective(trial, X_train, X_test, y_train, y_test)
        scores.append(RMSE_result)
    return np.mean(scores)

optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.
logging.info("Optimizing the model")
study = optuna.create_study(direction="minimize")
study.optimize(objective_cv, n_trials=100, n_jobs=1)
best_model = create_model(study.best_trial)
best_model.fit(features, target,feature_name=feature_names)
target_pred = best_model.predict(features_test)
fig,ax = plt.subplots(1,1,figsize=(100,5))
ax = optuna.visualization.matplotlib.plot_optimization_history(study)
plt.gca().set_xlabel("Number of trials")
plt.gca().set_ylabel("RMSE")
plt.suptitle("Optimization history " + sub)
plt.savefig("Optimization history " + sub + ".png")


logging.warning("RMSE_score_test_unoptimized: ", RMSE_score_unoptimized)
logging.warning("R2_score_test_unoptimized: ", R2_score_unoptimized)
logging.warning("RMSE_score_best_validation: ", study.best_value)
logging.warning("RMSE_score_test_optimized: ", np.sqrt(mean_squared_error(target_test,target_pred)))
logging.warning("R2_score_test_optimized: ", r2_score(target_test,target_pred))


#plot the feature importance
fig,ax = plt.subplots(1,1,figsize=(10,5))
lgb.plot_importance(best_model,ax=ax)
plt.savefig("Feature importance " + sub + ".png")
# %%
interp = interpolate.interp1d(concat_T,concat_Y,bounds_error=None,fill_value = "extrapolate")
concat_Y = np.array(interp(SB_data[subj]["t_uNG"])).squeeze()
interp = interpolate.interp1d(concat_T,concat_X,bounds_error=None,fill_value = "extrapolate")
concat_X = np.array(interp(SB_data[subj]["t_uNG"])).squeeze()
interp = interpolate.interp1d(concat_T_test,concat_Y_test,bounds_error=None,fill_value = "extrapolate")
concat_Y_test = np.array(interp(SB_data[subj_test]["t_uNG"])).squeeze()
interp = interpolate.interp1d(concat_T_test,concat_X_test,bounds_error=None,fill_value = "extrapolate")
concat_X_test = np.array(interp(SB_data[subj_test]["t_uNG"])).squeeze()

interp = interpolate.interp1d(concat_T,concat_T,bounds_error=None,fill_value = "extrapolate")
concat_T = np.array(interp(SB_data[subj]["t_uNG"])).squeeze()
interp = interpolate.interp1d(concat_T_test,concat_T_test,bounds_error=None,fill_value = "extrapolate")
concat_T_test = np.array(interp(SB_data[subj_test]["t_uNG"])).squeeze()

#predicts = []
#times = []
#for i in range(len(features_test)):
#    predicts.append(best_model.predict(features_test[i].reshape(1, -1)))
#    times.append(np.mean(np.array(subportions_test_time).squeeze()[i]))

predicts = best_model.predict(features_test)
times = np.mean(np.array(subportions_test_time),axis=1)

interp = interpolate.interp1d(np.array(times).squeeze(),np.array(predicts).squeeze(),bounds_error=None,fill_value = "extrapolate")
interpolated_predicts = interp(SB_data[subj_test]["t_uNG"])
#b, a = signal.butter(2, 1/(fs*0.5), btype="lowpass")
#interpolated_predicts = signal.filtfilt(b,a,np.array(interpolated_predicts).squeeze())

fig,ax = plt.subplots(1,1,figsize=(100,15))
ax.plot(SB_data[subj_test]["t_uNG"],SB_data[subj_test][data_ECG],label = "ECG - normalized")
ax.plot(SB_data[subj_test]["t_uNG"],interpolated_predicts,label = "Predicted ECG interpolated - normalized")
ax.spines[['top','right']].set_visible(False)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Normalized amplitude")
ax.legend()
plt.suptitle("Test Results " + sub)
plt.savefig("Test Results " + sub + ".png")

#MLfunctions_2.metrics_resp(concat_Y_test,concat_X_test,interpolated_predicts,concat_T_test,fs,target_test,predicts)
#plt.suptitle("Test Results")

#%%

#predicts = []
#times = []
#for i in range(len(features)):
#    predicts.append(best_model.predict(features[i].reshape(1, -1)))
#    times.append(np.mean(np.array(subportions_time).squeeze()[i]))

predicts = best_model.predict(features)
times = np.mean(np.array(subportions_time),axis=1)

interp = interpolate.interp1d(np.array(times).squeeze(),np.array(predicts).squeeze(),bounds_error=None,fill_value = "extrapolate")
interpolated_predicts = interp(SB_data[subj]["t_uNG"])
#b, a = signal.butter(2, 1/(fs*0.5), btype="lowpass")
#interpolated_predicts = signal.filtfilt(b,a,np.array(interpolated_predicts).squeeze())
fig,ax = plt.subplots(1,1,figsize=(30,20))
ax.plot(SB_data[subj]["t_uNG"],SB_data[subj][data_ECG],label = "ECG - normalized")
ax.plot(SB_data[subj]["t_uNG"],interpolated_predicts,label = "Predicted ECG - normalized")
ax.spines[['top','right']].set_visible(False)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Normalized amplitude")
ax.legend()
plt.suptitle("Train Results " + sub)
plt.savefig("Train Results " + sub + ".png")

logging.shutdown()

# %%
