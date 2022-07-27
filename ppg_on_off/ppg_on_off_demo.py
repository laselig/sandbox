"""
CONFIDENTIAL
__________________
2021 Happy Health Incorporated
All Rights Reserved.
NOTICE:  All information contained herein is, and remains
the property of Happy Health Incorporated and its suppliers,
if any.  The intellectual and technical concepts contained
herein are proprietary to Happy Health Incorporated
and its suppliers and may be covered by U.S. and Foreign Patents,
patents in process, and are protected by trade secret or copyright law.
Dissemination of this information or reproduction of this material
is strictly forbidden unless prior written permission is obtained
from Happy Health Incorporated.
Authors: Lucas Selig <lucas@happy.ai>
"""
import numpy as np
import pandas as pd
import scipy.signal as dsp
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from happyalgos.stress.live import (
    HPYLiveStressRunner,
    HPYLiveStressPersonalizer,
    HPYLiveStressOptions,
    HPYLiveStressState,
    HPYLiveStressPersonalization,
)
# MV = magic values
MV1 = 0.00082
MV2 = 1.06662
MV3 = 0.93380
MV4 = 0.00002
MV5 = 0.00007

my_dictionary = {"ts": [1, 2, 3],
                "asdlfkjasdflkj": [12, 34]
                 }

def window_ranger(ts_start, ts_end, stride, window):
    """Makes window ranges given a start and end time.

    Parameters
    ----------
    ts_start : float
        Starting time
    ts_end : float
        Ending time
    stride : float
        Size of step to succcessive window
    window : float
        Size of window

    Returns
    -------
    np.array (N x 2)
        Array of windows start/end points, inclusive on left, exclusive on right.
    """
    window_beginings = np.arange(ts_start, ts_end, stride)
    window_endings = window_beginings + window
    windows_ranges = np.vstack([window_beginings, window_endings]).T
    return windows_ranges
def make_initial_prediction(acc_mean, acc_var):
    if(acc_var <= MV1):
        if(acc_mean <= MV2):
            if(acc_mean <= MV3):
                if(acc_var <= MV4):
                    return 0
                else:
                    return 1
            else:
                return 0
        else:
            if(acc_var <= MV5):
                return 0
            else:
                return 1
    else:
        return 1
def make_final_prediction(X, sleep_threshold, awake_threshold):
    n = X.shape[0]
    y_pred = np.zeros(n)
    clf_pred_history = np.zeros(n)

    burnin = 20
    lookback = 20
    majority_pct_sleep = sleep_threshold
    majority_pct_awake = awake_threshold

    for i in range(n):
        print("pred calc window:", i)
        pred = [make_initial_prediction(X[i][0], X[i][1])]
        clf_pred_history[i] = pred[0]
        if(i >= burnin):
            last_20 = clf_pred_history[i-lookback:i]
            if(pred[0] == 0): # wants to put us in asleep
                if((lookback - np.count_nonzero(last_20)) / 20 >= majority_pct_sleep): # how badly
                    y_pred[i] = 0
                else:
                    y_pred[i] = 1

            if(pred[0] == 1): #wants to put us in awake
                if(np.count_nonzero(last_20) / 20 >= majority_pct_awake): # how badly
                    y_pred[i] = 1
                else:
                    y_pred[i] = 0

        else:
            y_pred[i] = pred[0]
    return y_pred


acc_raw = pd.read_parquet("/Users/lselig/selig-sandbox/data/DB5FA386-D75A-482B-A35A-5EFA94E58AC5_d81371000060_03092022_acc_raw.parquet")
# for plotting
labels = pd.read_parquet("/Users/lselig/selig-sandbox/data/DB5FA386-D75A-482B-A35A-5EFA94E58AC5_d81371000060_03092022_windowed_labels.parquet")

# apply filter to signal
b, a = dsp.firwin(3, cutoff=0.2), [1]
acc_filt = dsp.lfilter(b, a, acc_raw.mag.values)
acc_raw["acc_filt"] = acc_filt
# plt.plot(acc_raw.ts, acc_raw.mag.values, label = "raw")
# plt.plot(acc_raw.ts, acc_filt, label = "filt")
# plt.legend()
# plt.show()

# break signal into 20 second non overlapping windows
window_ranges = window_ranger(
    acc_raw.iloc[0].ts,
    acc_raw.iloc[-1].ts,
    20,
    20)
    # fig, axs = plt.subplots(2, 1, figsize = (15, 9))
acc_mu = np.ones(labels.shape[0])
acc_var = np.ones(labels.shape[0])

# calculate features
for i, window in enumerate(window_ranges):
    print("feature calc window:", i)
    start = window[0]
    end = window[1]
    acc_slice = acc_raw[(acc_raw.ts >= start) & (acc_raw.ts <= end)].acc_filt.values
    acc_mu[i] = np.nanmean(acc_slice)
    acc_var[i] = np.nanvar(acc_slice)

# align and remove nans
labels = labels.label.values
labels[np.where(np.isnan(acc_mu))] = np.nan

acc_mu = acc_mu[~np.isnan(acc_mu)]
labels = labels[~np.isnan(labels)]
acc_var = acc_var[~np.isnan(acc_var)]


# calculate predictions
predictions = make_final_prediction(np.vstack((acc_mu, acc_var)).T, 0.555555, 0.333333)

# visualize
fig, axs = plt.subplots(3, 1, figsize = (15, 9), sharex = True)
axs[0].plot(acc_mu)
axs[0].set_ylabel("acc mean")
axs[1].plot(acc_var)
axs[1].set_ylabel("acc var")
axs[2].plot(labels, label = "True")
axs[2].set_ylabel("isAwake")
axs[2].plot(predictions, label = "Pred")
axs[2].legend()
class_report = classification_report(labels,
                                     predictions,
                                     target_names=["Asleep", "Awake"],
                                     output_dict=True,
                                     digits=4)
print(class_report)
plt.show()
