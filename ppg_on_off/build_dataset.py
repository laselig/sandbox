from happyalgos.integrity.sqi import HPYSQIRunner
from dateutil import parser
from collections import namedtuple, defaultdict
import matplotlib.dates as md
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import happyalgos.utils.buffer as buf
from happyalgos.integrity.sqi import HPYSQIRunner
from sklearn.preprocessing import minmax_scale
import seaborn as sns
from dataclasses import asdict
import sqlite3
import binascii
import struct
import datetime
from datetime import datetime, timedelta,timezone
import json, glob, pickle
from pathlib import Path
import os
from happyalgos.sleep.preprocess import HPYSleepPreprocessor
from happyalgos.sleep.features import HPYSleepFeatureExtractor
import happyalgos.sleep.constants as const
from happyalgos.sleep.utils import resample_data, scale_features
from happyalgos.sleep.preprocess.accel import preprocess_acceleration
from happyalgos.sleep.preprocess.eda import preprocess_eda
from happyalgos.sleep.preprocess.temperature import preprocess_temperature
from happyalgos.sleep.preprocess.timestamps import preprocess_timestamps
from happyalgos.utils.filtering import window_ranger, window_maker
import scipy.signal as dsp

path = '/Volumes/GoogleDrive/Shared drives/Partners - Studies Data sets/Happy_App_Sleep_Window_Nans_final/'
save_path = "/Users/lselig/Desktop/P8_SLEEP_WINDOW/"
labels = pd.read_csv('/Volumes/GoogleDrive/Shared drives/Partners - Studies Data sets/Happy_App_Sleep_Window_Nans/sw_labels.csv')
sleep_labels = labels.iloc[np.where(labels.has_missing_sleep_data== False)[0],:].reset_index()
sleep_labels=sleep_labels[['user_id','date','start','end','type']]
trials = glob.glob(path+'*/*/*')
tz_table = pd.read_csv('/Users/lselig/Desktop/user_id_timezone.csv')
save_mat_flag=True
plot_flag=False
# save_mat_path = '/Volumes/GoogleDrive/Shared drives/Sleep and ON-OFF/FEATURES/SLEEP_WINDOW/Happy_app_2022/'
count=0
def get_happy_data(path,labels_table,tz_offset):
    eda_df = pd.read_parquet(path +'/eda.parquet')
    acc_df = pd.read_parquet(path +'/acc.parquet')
    skin_temp_df = pd.read_parquet(path +'/skintemp.parquet')
    start_ts = np.min([eda_df.etime[0],acc_df.etime[0],skin_temp_df.etime[0]])
    end_ts = np.max([eda_df.etime.iloc[-1],acc_df.etime.iloc[-1],skin_temp_df.etime.iloc[-1]])
    ts=np.arange(start_ts,end_ts,1)
    labels = np.ones(ts.shape)
    labels_table.sort_values(by='type',inplace=True,ascending= False)
    for i in range(len(labels_table)):
        this_start_t = labels_table.iloc[i].start
        this_end_t = labels_table.iloc[i].end
        if labels_table.iloc[i].type == 'sleep_window':
            labels[(ts>=this_start_t) & (ts<this_end_t)] = 0
        else:
            labels[(ts >= this_start_t) & (ts < this_end_t)] = 1
    labels_df=pd.DataFrame({'etime':ts,'label':labels})

    eda_df["etime"] = eda_df["etime"] + 3600 * tz_offset
    acc_df["etime"] = acc_df["etime"]  + 3600 * tz_offset
    skin_temp_df["etime"] = skin_temp_df["etime"] + 3600 * tz_offset
    labels_df["etime"] = labels_df["etime"] + 3600 * tz_offset

    eda_df.sort_values(by=['etime'])
    acc_df.sort_values(by=['etime'])
    skin_temp_df.sort_values(by=['etime'])
    labels_df.sort_values(by=['etime'])

    return eda_df, acc_df, skin_temp_df,labels_df

def get_tz_hours(str,date_str):
    tz_dict_hrs = {"UTC-6:00 CST": -6,
                   "UTC-7:00 MST": -7,
                   "UTC-8:00 PST": -8,
                   "UTC−5:00 EST\n": -5,
                   "UTC−5:00 EST": -5,
                   "UTC-5:00 EST": -5,
                   "UTC+8:00": 8,
                   "America/New York": -5,
                   'America/Chicago': -6,
                   'America/New_York': -5,
                   'America/Denver': -7,
                   'America/Phoenix': -7,
                   'America/Los_Angeles': -8,
                   'Asia/Taipei': +8,
                   'US/Pacific': -8,
                   'America/Toronto': -5,
                   'US/Central': -6,
                   'Pacific/Honolulu': -10}
    try:
        user_tz = tz_dict_hrs[str].values[0]
    except:
        # if we aren't tracking their TZ assume to be CST
        user_tz = -6

    date_datetime =datetime.strptime(date_str, '%m%d%Y')
    if date_datetime > datetime(2022, 3, 13):
        user_tz = user_tz+1

    return user_tz

def remove_nan(eda_df,acc_df,skin_temp_df):
    eda_df = eda_df.loc[~np.isnan(eda_df.current).values]
    acc_df = acc_df.loc[~np.isnan(acc_df.x).values]
    skin_temp_df = skin_temp_df.loc[~np.isnan(skin_temp_df.temperature).values]
    return eda_df,acc_df,skin_temp_df

def get_labels(labels,window_ranges):
    new_labels = np.zeros((len(window_ranges),2))
    for i in range(len(window_ranges)):
        window=window_ranges[i,:]
        new_labels[i,1] = np.round(np.mean(labels[(labels[:,0]>=window[0]) &(labels[:,0]<window[1]),1]))
        new_labels[i, 0] = window[0]
    return new_labels



for trial_path in trials:
    user_id = trial_path.split('/')[-3]
    date_str = trial_path.split('/')[-1]
    date = int(date_str)
    this_trial_labels = sleep_labels.loc[(sleep_labels.user_id==user_id) & (sleep_labels.date==date)].reset_index(drop=True)
    # this_trial_labels.reset_index(inplace=True,drop=True)
    if len(this_trial_labels)==0:
        continue
    print(str(count)+' ' + user_id +' ' + date_str)
    count=count+1
    save_trial_path = save_path + user_id  +date_str[0:2] + '_' +date_str[2:4] + '_' +date_str[-4:]
    if not os.path.isdir(save_trial_path):
        os.mkdir(save_trial_path)
    if np.sum(tz_table.user_id==user_id.lower())==0:
        tz=0
    else:
        tz = get_tz_hours(tz_table.timezone.loc[tz_table.user_id==user_id.lower()].values[0],date_str)

    eda_df,acc_df,skin_temp_df,labels_df = get_happy_data( trial_path,this_trial_labels,tz)
    # eda_df.to_csv(save_trial_path +'/eda.csv',index=False)
    # skin_temp_df.to_csv(save_trial_path +'/skin_temp.csv',index=False)
    # labels_df.to_csv(save_trial_path +'/labels.csv',index=False)
    # acc_df.to_csv(save_trial_path +'/acc.csv',index=False)

    # if plot_flag:
    #     fig, (ax0, ax1,ax2, ax3) = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
    #     fig.suptitle(user_id + ' ' + date_str)
    #     ax0.step([datetime.fromtimestamp(ft, tz=timezone.utc) for ft in labels_df.etime], labels_df.label, label='Oura',where='post')
    #     ax1.plot(pd.to_datetime(eda_df.etime, unit = "s"), eda_df.conductance, label='v0',color='blue')
    #     ax1.set_ylabel('eda')
    #     ax2.plot(pd.to_datetime(acc_df.etime, unit = "s"), acc_df.mag, label='v0',color='blue')
    #     ax2.set_ylabel('acc mag')
    #     ax3.plot(pd.to_datetime(skin_temp_df.etime, unit = "s"), skin_temp_df.temperature, label='v0',color='blue')
    #     ax3.set_ylabel('temperature')
    #     plt.close('all')
    eda_df,acc_df,skin_temp_df = remove_nan(eda_df,acc_df,skin_temp_df)
    # shifting data to UTC (was saved in central time)
    # eda_ts = eda_df['etime'].values.astype(np.float64)
    # eda = eda_df['conductance'].values
    # plt.plot(eda_ts,eda)
    acceleration_ts = (acc_df['etime'].values.astype(np.float64))
    acceleration = acc_df[['x', 'y', 'z']].values
    # temperature_ts = (skin_temp_df['etime'].values.astype(np.float64))
    # temperature = skin_temp_df['temperature'].values
    labels = np.array(labels_df)

    # resample acc from 104 to 52/26 and temp to 0.25
    acceleration_ts, acceleration = resample_data(acceleration_ts, acceleration, const.ACC_FS)
    acc_mag = acceleration[:, 0] ** 2 + acceleration[:, 1] ** 2 + acceleration[:, 2] ** 2
    # acceleration_x = acceleration[:, 0]
    # acceleration_y = acceleration[:, 1]
    # acceleration_z = acceleration[:, 2]
    #r
    # base = acceleration_x**2 + acceleration_y**2
    # base[base == 0] = const.EPSILON
    # angle = const.HALF_A_CIRCLE * np.arctan(acceleration_z / np.sqrt(base)) / np.pi
    # acceleration_angle = np.abs(np.diff(angle))
    # acceleration_angle = np.append(acceleration_angle, 0.0)
    #
    b, a = dsp.firwin(3, cutoff=0.2), [1]
    acc_file = dsp.lfilter(b, a, acc_mag)

    # plt.plot(acceleration_ts, acc_mag, label = "raw")
    # plt.plot(acceleration_ts, acc_file, label = "filt")
    # plt.legend()
    # plt.show()

    acc_df = pd.DataFrame({"ts": acceleration_ts,
                           "ss": acc_file})
    #                        "x": acceleration_x,
    #                        "y": acceleration_y,
    #                        "z": acceleration_z,
    #                        "ang": acceleration_angle})
    # # temperature_ts, temperature = resample_data(temperature_ts, temperature, const.TEMP_FS)
    # eda_ts, eda = resample_data(eda_ts, eda, const.EDA_FS)
    # preprocessor = HPYSleepPreprocessor(phase="window")
    # features_extractor = HPYSleepFeatureExtractor(preprocessor=preprocessor, order=const.SLEEP_WINDOW_ORDER,
    #                                               phase="all")

    # because I want the ts to match the labels I needed to create ts my self and couldn't use preprocessor

    # snippet_t0s = np.arange(labels[0, 0], labels[-1, 0] + const.SW_SNIPPET_STRIDE, const.SW_SNIPPET_STRIDE)
    window_ranges = window_ranger(
        acceleration_ts[0],
        acceleration_ts[-1],
        20,
        20,
    )
    labels = get_labels(labels,window_ranges)

    fig, axs = plt.subplots(2, 1, figsize = (15, 9))
    acc_mu = np.ones(labels.shape[0])
    acc_var = np.ones(labels.shape[0])
    # acc_x = np.ones(labels.shape[0])
    # acc_y = np.ones(labels.shape[0])
    # acc_z = np.ones(labels.shape[0])
    # acc_ang = np.ones(labels.shape[0])

    for i, window in enumerate(window_ranges):
        print("On window: ", i)
        start = window[0]
        end = window[1]
        ss_slice = acc_df[(acc_df.ts >= start) & (acc_df.ts <= end)].ss.values
        acc_mu[i] = np.nanmean(ss_slice)
        acc_var[i] = np.nanvar(ss_slice)
        # acc_x[i] = np.nanvar(np.diff(acc_df[(acc_df.ts >= start) & (acc_df.ts <= end)].x.values))
        # acc_y[i] = np.nanvar(np.diff(acc_df[(acc_df.ts >= start) & (acc_df.ts <= end)].y.values))
        # acc_z[i] = np.nanvar(np.diff(acc_df[(acc_df.ts >= start) & (acc_df.ts <= end)].z.values))
        # acc_ang[i] = np.nanmean(acc_df[(acc_df.ts >= start) & (acc_df.ts <= end)].ang.values)

    df = pd.DataFrame({"ts": labels[:, 0],
                       "acc_mean": acc_mu,
                       "acc_var": acc_var,
                       # "acc_x": acc_x,
                       # "acc_y": acc_y,
                       # "acc_z": acc_z,
                       # "acc_ang": acc_ang,
                       "label": labels[:, 1]})

    # fig, axs = plt.subplots(7, 1, figsize = (15, 9), sharex = True)
    # axs[0].plot(df.acc_mean)
    # axs[1].plot(df.acc_var)
    # axs[2].plot(df.acc_x)
    # axs[3].plot(df.acc_y)
    # axs[4].plot(df.acc_z)
    # axs[5].plot(df.acc_ang)
    # axs[6].plot(df.label)
    # plt.tight_layout()
    # plt.show()
    df.to_parquet(f"/Users/lselig/Desktop/labeled_sleep_3numtaps/{user_id}_{date_str}.parquet", index = False)
    a = 1
