import numpy as np
from scipy.signal import medfilt, lfilter
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy import signal
from numpy.fft import fft, ifft


def normalize(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    normalized_signal = (signal - mean) / std
    return normalized_signal

def normalize_min_max(signal):
    min = np.min(signal)
    max = np.max(signal)
    normalized_signal = (signal - min) / (max - min)
    return normalized_signal

def non_overlapping_subportions(signal, window_size, sampling_frequency, overlap_percentage):
    signal_length = len(signal)
    samples_per_window = int(window_size * sampling_frequency)
    overlap = int(samples_per_window * overlap_percentage / 100)
    num_windows = int((signal_length - overlap) / (samples_per_window - overlap)) + 1
    subportions = []
    start_position = 0
    for i in range(num_windows):
        if len(signal[start_position:start_position + samples_per_window]) == samples_per_window:
            #window = np.ones(samples_per_window)
            windowed_signal = signal[start_position:start_position + samples_per_window]
            subportions.append(windowed_signal)
            start_position += samples_per_window - overlap

    return subportions

#function to divide the signal in subportions without overlapping
def non_overlapping_subportions_2(signal, window_size, sampling_frequency):
    signal_length = len(signal)
    samples_per_window = int(window_size * sampling_frequency)
    num_windows = int(signal_length / samples_per_window) + 1
    subportions = []
    start_position = 0
    for i in range(num_windows):
        if len(signal[start_position:start_position + samples_per_window]) == samples_per_window:
            #window = np.ones(samples_per_window)
            windowed_signal = signal[start_position:start_position + samples_per_window]
            subportions.append(windowed_signal)
            start_position += samples_per_window
        else:
            subportions.append(signal[start_position:])
    return subportions        


def remove_outliers_IQR(signal):
    Q1 = np.percentile(signal, 25)
    Q3 = np.percentile(signal, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_signal = signal[np.where((signal >= lower_bound) & (signal <= upper_bound))]
    return filtered_signal , np.where((signal >= lower_bound) & (signal <= upper_bound))

def outliers_IQR_to_nan(signal):
    Q1 = np.percentile(signal, 25)
    Q3 = np.percentile(signal, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = np.where((signal < lower_bound) | (signal > upper_bound))[0]
    signal[outliers] = np.nan

    return signal,np.where((signal >= lower_bound) & (signal <= upper_bound))


def remove_outliers_median_filter(signal, window_size=3):
    filtered_signal = medfilt(signal, window_size)
    return filtered_signal

def features_extractor(data,fs):
    N = len(data)
    sotto_segnali = []
    for i in range(N):
        sign = data[i].ravel()
        mean = np.mean(sign)
        power = np.mean(pow(sign,2))
        max_t = np.argmax(sign,keepdims = True)
        min_t = np.argmin(sign,keepdims = True)
        max_val = sign[max_t]
        min_val = sign[min_t]
        skew = stats.skew(sign)
        kurt = stats.kurtosis(sign)
        RMS = np.sqrt(np.mean(sign**2))
        var = np.var(sign)
        #find the peaks and computed the mean difference in time between them
        peaks, _ = signal.find_peaks(sign,distance=fs/2)
        if len(peaks) > 1:
            mean_diff = np.mean(np.diff(peaks))
        else:
            mean_diff = 0
        
        fft_sign = fft(sign)
        fft_sign = fft_sign[0:int(N/2)]
        fft_sign = np.abs(fft_sign)
        fft_sign = fft_sign/np.sum(fft_sign)
        freq = np.linspace(0,fs/2,int(N/2))
        if len(freq) != len(fft_sign):
            freq = freq[0:len(fft_sign)]
        mean_freq = np.sum(freq*fft_sign)
        std_freq = np.sqrt(np.sum((freq-mean_freq)**2*fft_sign))
        max_freq = freq[np.argmax(fft_sign)]
        min_freq = freq[np.argmin(fft_sign)]
        skew_freq = stats.skew(fft_sign)
        kurt_freq = stats.kurtosis(fft_sign)
        max_val_freq = np.max(fft_sign)
        min_val_freq = np.min(fft_sign)
        #final = [mean,power,float(max_t),float(min_t),max_val[0],min_val[0],skew[0],kurt[0],RMS,var,mean_freq,std_freq,max_freq,min_freq,skew_freq,kurt_freq,max_val_freq,min_val_freq,mean_diff]
        final = [mean,power,float(max_t),float(min_t),max_val[0],min_val[0],skew[0],kurt[0],RMS,var]
        sotto_segnali.append(final)
    return sotto_segnali


def features_extractor_with_past(data,points_back):
    N = len(data)
    sotto_segnali = []
    past = np.zeros(points_back)
    for i in range(N):
        sign = data[i].ravel()
        mean = np.mean(sign)
        power = np.mean(pow(sign,2))
        max_t = np.argmax(sign,keepdims = True)
        min_t = np.argmin(sign,keepdims = True)
        max_val = sign[max_t]
        min_val = sign[min_t]
        skew = stats.skew(sign)
        kurt = stats.kurtosis(sign)
        RMS = np.sqrt(np.mean(sign**2))
        var = np.var(sign)
        #insert in the subpoortions the past points of mean for number of points_back for each signal
        final = [mean,power,float(max_t),float(min_t),max_val[0],min_val[0],skew[0],kurt[0],RMS,var,*past]
        sotto_segnali.append(final)
        past = np.roll(past,-1)
        past[-1] = mean
    return  sotto_segnali

def features_extractor_with_pasts(data,points_back):
    N = len(data)
    sotto_segnali = []
    past_mean = np.zeros(points_back)
    past_power = np.zeros(points_back)
    past_max_t = np.zeros(points_back)
    past_min_t = np.zeros(points_back)
    past_max_val = np.zeros(points_back)
    past_min_val = np.zeros(points_back)
    past_skew = np.zeros(points_back)
    past_kurt = np.zeros(points_back)
    past_RMS = np.zeros(points_back)
    past_var = np.zeros(points_back)
    past_mean_slope = np.zeros(points_back)
    past_power_slope = np.zeros(points_back)
    past_max_t_slope = np.zeros(points_back)
    past_min_t_slope = np.zeros(points_back)
    past_max_val_slope = np.zeros(points_back)
    past_min_val_slope = np.zeros(points_back)
    past_skew_slope = np.zeros(points_back)
    past_kurt_slope = np.zeros(points_back)
    past_RMS_slope = np.zeros(points_back)
    past_var_slope = np.zeros(points_back)
    for i in range(N):
        sign = data[i].ravel()
        mean = np.mean(sign)
        power = np.mean(pow(sign,2))
        max_t = np.argmax(sign,keepdims = True)
        min_t = np.argmin(sign,keepdims = True)
        max_val = sign[max_t]
        min_val = sign[min_t]
        skew = stats.skew(sign)
        kurt = stats.kurtosis(sign)
        RMSs = np.sqrt(np.mean(sign**2))
        var = np.var(sign)
        sign_slope = np.diff(sign)
        mean_slope = np.mean(sign_slope)
        power_slope = np.mean(pow(sign_slope,2))
        max_t_slope = np.array(np.argmax(sign_slope,keepdims = True)).squeeze()
        min_t_slope = np.array(np.argmin(sign_slope,keepdims = True)).squeeze()
        max_val_slope = sign_slope[max_t_slope].squeeze()
        min_val_slope = sign_slope[min_t_slope].squeeze()
        skew_slope = stats.skew(sign_slope)
        kurt_slope = stats.kurtosis(sign_slope)
        RMS_slope = np.sqrt(np.mean(sign_slope**2))
        var_slope = np.var(sign_slope)
        #insert in the subpoortions the past points of mean for number of points_back for each signal
        #final = [mean,power,float(max_t),float(min_t),max_val[0],min_val[0],skew[0],kurt[0],RMSs,var,*past_mean,*past_power,*past_max_t,*past_min_t,*past_max_val,*past_min_val,*past_skew,*past_kurt,*past_RMS,*past_var]
        final = [mean,power,float(max_t),float(min_t),max_val[0],min_val[0],skew,kurt,RMSs,var,mean_slope,power_slope,float(max_t_slope),float(min_t_slope),max_val_slope,min_val_slope,skew_slope,kurt_slope,RMS_slope,var_slope,*past_mean,*past_power,*past_max_t,*past_min_t,*past_max_val,*past_min_val,*past_skew,*past_kurt,*past_RMS,*past_var,*past_mean_slope,*past_power_slope,*past_max_t_slope,*past_min_t_slope,*past_max_val_slope,*past_min_val_slope,*past_skew_slope,*past_kurt_slope,*past_RMS_slope,*past_var_slope]
        sotto_segnali.append(final)
        past_mean = np.roll(past_mean,-1)
        past_power = np.roll(past_power,-1)
        past_max_t = np.roll(past_max_t,-1)
        past_min_t = np.roll(past_min_t,-1)
        past_max_val = np.roll(past_max_val,-1)
        past_min_val = np.roll(past_min_val,-1)
        past_skew = np.roll(past_skew,-1)
        past_kurt = np.roll(past_kurt,-1)
        past_RMS = np.roll(past_RMS,-1)
        past_var = np.roll(past_var,-1)
        past_mean_slope = np.roll(past_mean_slope,-1)
        past_power_slope = np.roll(past_power_slope,-1)
        past_max_t_slope = np.roll(past_max_t_slope,-1)
        past_min_t_slope = np.roll(past_min_t_slope,-1)
        past_max_val_slope = np.roll(past_max_val_slope,-1)
        past_min_val_slope = np.roll(past_min_val_slope,-1)
        past_skew_slope = np.roll(past_skew_slope,-1)
        past_kurt_slope = np.roll(past_kurt_slope,-1)
        past_RMS_slope = np.roll(past_RMS_slope,-1)
        past_var_slope = np.roll(past_var_slope,-1)
        past_mean[-1] = mean
        past_power[-1] = power
        past_max_t[-1] = max_t
        past_min_t[-1] = min_t
        past_max_val[-1] = max_val
        past_min_val[-1] = min_val
        past_skew[-1] = skew
        past_kurt[-1] = kurt
        past_RMS[-1] = RMSs
        past_var[-1] = var
        past_mean_slope[-1] = mean_slope
        past_power_slope[-1] = power_slope
        past_max_t_slope[-1] = max_t_slope
        past_min_t_slope[-1] = min_t_slope
        past_max_val_slope[-1] = max_val_slope
        past_min_val_slope[-1] = min_val_slope
        past_skew_slope[-1] = skew_slope
        past_kurt_slope[-1] = kurt_slope
        past_RMS_slope[-1] = RMS_slope
        past_var_slope[-1] = var_slope
    return  sotto_segnali

def features_extractor_with_pasts_less(data,points_back):
    N = len(data)
    sotto_segnali = []
    past_mean = np.zeros(points_back)
    past_max_val = np.zeros(points_back)
    past_min_val = np.zeros(points_back)
    past_skew = np.zeros(points_back)
    past_kurt = np.zeros(points_back)
    past_var = np.zeros(points_back)
    past_mean_slope = np.zeros(points_back)
    past_max_t_slope = np.zeros(points_back)
    past_min_t_slope = np.zeros(points_back)
    past_skew_slope = np.zeros(points_back)
    past_kurt_slope = np.zeros(points_back)
    past_var_slope = np.zeros(points_back)
    for i in range(N):
        sign = data[i].ravel()
        mean = np.mean(sign)
        max_t = np.argmax(sign,keepdims = True)
        min_t = np.argmin(sign,keepdims = True)
        max_val = sign[max_t]
        min_val = sign[min_t]
        skew = stats.skew(sign,keepdims = True)
        kurt = stats.kurtosis(sign,keepdims = True)
        var = np.var(sign)
        #sign_slope = np.diff(sign)
        #mean_slope = np.mean(sign_slope)
        #power_slope = np.mean(pow(sign_slope,2))
        #max_t_slope = np.array(np.argmax(sign_slope,keepdims = True)).squeeze()
        #min_t_slope = np.array(np.argmin(sign_slope,keepdims = True)).squeeze()
        #max_val_slope = sign_slope[max_t_slope].squeeze()
        #min_val_slope = sign_slope[min_t_slope].squeeze()
        #skew_slope = stats.skew(sign_slope,keepdims = True)[0]
        #kurt_slope = stats.kurtosis(sign_slope,keepdims = True)[0]
        #RMS_slope = np.sqrt(np.mean(sign_slope**2))
        #var_slope = np.var(sign_slope)
        #insert in the subpoortions the past points of mean for number of points_back for each signal
        final = [mean,max_val[0],min_val[0],skew[0],kurt[0],var,*past_mean,*past_max_val,*past_min_val,*past_skew,*past_kurt,*past_var]
        #final = [mean,power,float(max_t),float(min_t),max_val[0],min_val[0],skew[0],kurt[0],RMSs,var,mean_slope,power_slope,float(max_t_slope),float(min_t_slope),max_val_slope,min_val_slope,skew_slope,kurt_slope,RMS_slope,var_slope,*past_mean,*past_power,*past_max_t,*past_min_t,*past_max_val,*past_min_val,*past_skew,*past_kurt,*past_RMS,*past_var,*past_mean_slope,*past_power_slope,*past_max_t_slope,*past_min_t_slope,*past_max_val_slope,*past_min_val_slope,*past_skew_slope,*past_kurt_slope,*past_RMS_slope,*past_var_slope]
        sotto_segnali.append(final)
        past_mean = np.roll(past_mean,-1)
        past_max_val = np.roll(past_max_val,-1)
        past_min_val = np.roll(past_min_val,-1)
        past_skew = np.roll(past_skew,-1)
        past_kurt = np.roll(past_kurt,-1)
        past_var = np.roll(past_var,-1)
        """past_mean_slope = np.roll(past_mean_slope,-1)
        past_power_slope = np.roll(past_power_slope,-1)
        past_max_t_slope = np.roll(past_max_t_slope,-1)
        past_min_t_slope = np.roll(past_min_t_slope,-1)
        past_max_val_slope = np.roll(past_max_val_slope,-1)
        past_min_val_slope = np.roll(past_min_val_slope,-1)
        past_skew_slope = np.roll(past_skew_slope,-1)
        past_kurt_slope = np.roll(past_kurt_slope,-1)
        past_RMS_slope = np.roll(past_RMS_slope,-1)
        past_var_slope = np.roll(past_var_slope,-1)"""
        past_mean[-1] = mean
        past_max_val[-1] = max_val
        past_min_val[-1] = min_val
        past_skew[-1] = skew
        past_kurt[-1] = kurt
        past_var[-1] = var
        #past_mean_slope[-1] = mean_slope
        #past_power_slope[-1] = power_slope
        #past_max_t_slope[-1] = max_t_slope
        #past_min_t_slope[-1] = min_t_slope
        #past_max_val_slope[-1] = max_val_slope
        #past_min_val_slope[-1] = min_val_slope
        #past_skew_slope[-1] = skew_slope
        #past_kurt_slope[-1] = kurt_slope
        #past_RMS_slope[-1] = RMS_slope
        #past_var_slope[-1] = var_slope
    return  sotto_segnali

def features_extractor_with_pasts_names_less(points_back):
    names = ["mean","max_val","min_val","skew","kurt","var"]
    for i in range(points_back):
        names.append(f"mean_past_{i}")
    for i in range(points_back):
        names.append(f"max_val_past_{i}")
    for i in range(points_back):
        names.append(f"min_val_past_{i}")
    for i in range(points_back):
        names.append(f"skew_past_{i}")
    for i in range(points_back):
        names.append(f"kurt_past_{i}")
    for i in range(points_back):
        names.append(f"var_past_{i}")
    return names

def bpm_mean_RR(rr_series,fs,window_length):
    #rr_series = median_filter(rr_series,10)
    peaks = np.array(rr_series)
    ok = ~np.isnan(peaks)
    xp = ok.ravel().nonzero()[0]
    fp = peaks[~np.isnan(peaks)]
    x  = np.isnan(peaks).ravel().nonzero()[0]

    peaks[np.isnan(peaks)] = np.interp(x, xp, fp).astype(int)
    bpm = []
    for i in range(len(peaks)):
        rr_mean_in_window = np.mean(peaks[i : i + window_length])
        bpm.append(60*fs / (rr_mean_in_window*1000))
    return bpm

def features_extractor_names():
    #names = ["mean","power","max_t","min_t","max_val","min_val","skew","kurt","RMS","var","mean_freq","std_freq","max_freq","min_freq","skew_freq","kurt_freq","max_val_freq","min_val_freq","mean_diff"]
    names = ["mean","power","max_t","min_t","max_val","min_val","skew","kurt","RMS","var"]
    return names

def features_extractor_with_pasts_names(points_back):
    names = ["mean","power","max_t","min_t","max_val","min_val","skew","kurt","RMS","var","mean_slope","power_slope","max_t_slope","min_t_slope","max_val_slope","min_val_slope","skew_slope","kurt_slope","RMS_slope","var_slope"]
    for i in range(points_back):
        names.append(f"mean_past_{i}")
    for i in range(points_back):
        names.append(f"power_past_{i}")
    for i in range(points_back):
        names.append(f"max_t_past_{i}")
    for i in range(points_back):
        names.append(f"min_t_past_{i}")
    for i in range(points_back):
        names.append(f"max_val_past_{i}")
    for i in range(points_back):
        names.append(f"min_val_past_{i}")
    for i in range(points_back):
        names.append(f"skew_past_{i}")
    for i in range(points_back):
        names.append(f"kurt_past_{i}")
    for i in range(points_back):
        names.append(f"RMS_past_{i}")
    for i in range(points_back):
        names.append(f"var_past_{i}")
    for i in range(points_back):
        names.append(f"mean_slope_past_{i}")
    for i in range(points_back):
        names.append(f"power_slope_past_{i}")
    for i in range(points_back):
        names.append(f"max_t_slope_past_{i}")
    for i in range(points_back):
        names.append(f"min_t_slope_past_{i}")
    for i in range(points_back):
        names.append(f"max_val_slope_past_{i}")
    for i in range(points_back):
        names.append(f"min_val_slope_past_{i}")
    for i in range(points_back):
        names.append(f"skew_slope_past_{i}")
    for i in range(points_back):
        names.append(f"kurt_slope_past_{i}")
    for i in range(points_back):
        names.append(f"RMS_slope_past_{i}")
    for i in range(points_back):
        names.append(f"var_slope_past_{i}")
    return names

def create_subsets(dataset, labels, test_ratio=0.2, val_ratio=0.1, randomize=True, ratio=0.5):
    # Shuffle and split dataset into train, test, and validation subsets
    if randomize:
        dataset, labels = shuffle(dataset, labels, random_state=42)
    train_dataset, test_dataset, train_labels, test_labels = train_test_split(dataset, labels, test_size=test_ratio, stratify=labels, random_state=42)
    val_dataset, test_dataset, val_labels, test_labels = train_test_split(test_dataset, test_labels, test_size=val_ratio, stratify=test_labels, random_state=42)

    # Balance the classes
    rus = RandomUnderSampler(sampling_strategy=ratio)
    train_dataset, train_labels = rus.fit_resample(train_dataset, train_labels)
    val_dataset, val_labels = rus.fit_resample(val_dataset, val_labels)
    test_dataset, test_labels = rus.fit_resample(test_dataset, test_labels)

    return train_dataset, train_labels, val_dataset, val_labels, test_dataset, test_labels


def display_dataset_info(dataset, labels):

    # Convert dataset to a NumPy array if necessary
    if isinstance(dataset, list):
        dataset = np.array(dataset)

    # Check if labels is a NumPy array
    if not isinstance(labels, np.ndarray):
        raise TypeError("labels must be a numpy.ndarray")

    # Calculate class counts and display class balance
    unique_labels, class_counts = np.unique(labels, return_counts=True)
    print("Class Balance:")
    for label, count in zip(unique_labels, class_counts):
        print(f" - {label}: {count}")

    # Create a bar plot of class distribution
    plt.figure()
    plt.bar(unique_labels, class_counts)
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.show()

    # Display number of features
    num_features = dataset.shape[1]
    print("Number of Features:", num_features)
    

def correlate(X,Y,fs,method = "same"):
    X = (X -np.mean(X))
    std_a = np.linalg.norm(X)
    X = X / std_a
    Y = (Y-np.mean(Y))
    std_b = np.linalg.norm(Y)
    Y = Y/ std_b
    correlation = signal.correlate(X, Y, mode=method)
    #correlation = signal.correlate(X, np.hstack((Y[1:], Y)), mode='valid')
    #correlation = ifft(fft(X) * fft(Y).conj()).real
    lags = signal.correlation_lags(len(X), len(Y), mode=method)/fs
    return correlation,lags

def getMissingNumbers(arr): 
    poss = []
    # Traverse the array arr[] 
    i = 0
    for num in arr: 
        if int(num) != int(i):
            poss.append(i)
            i +=1
        i +=1
    return poss
  
def get_RMS(data, t, window_width):
    dt = t[1] - t[0]
    window_width_idcs = int(np.floor(window_width / dt))

    # Calculate the squared data
    squared_data = np.power(data, 2)

    # Calculate the moving mean of the squared data
    data_rms = np.sqrt(np.convolve(squared_data, np.ones(window_width_idcs), mode='same')/window_width_idcs)

    return data_rms

def get_RMS_2(data, t, window_width):

    dt = t[1] - t[0]
    window_width_idcs = int(window_width / dt)
    
    if window_width_idcs < 2:
        raise ValueError('The window width must be at least 2.')
        
    squared_data = np.power(data, 2)
    window = np.ones(window_width_idcs) / window_width_idcs
    mean_squared_data = lfilter(window, 1, squared_data)
    data_rms = np.sqrt(mean_squared_data)

    return data_rms

def media_mobile(tempo, valori, finestra):
    
    if len(tempo) != len(valori):
        raise ValueError("I vettori di tempo e valori devono avere la stessa lunghezza.")
    if finestra < 1:
        raise ValueError("La dimensione della finestra deve essere maggiore o uguale a 1.")

  # Calcolo della media mobile

    media_mobile = []
    for i in range(len(tempo)):
        if i < finestra - 1:
            media_mobile.append(np.nan)
        else:
            media = sum(valori[i - (finestra - 1) : i + 1]) / finestra
            media_mobile.append(media)

    return media_mobile