"""
CONFIDENTIAL
__________________
2022 Happy Health Incorporated
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
import seaborn as sns
import matplotlib.pyplot as plt
import glob, os, random
from pathlib import Path

np.random.seed( 0 )
sns.set_style( "darkgrid" )
from scipy import signal
from scipy.integrate import simps
pd.options.mode.chained_assignment = None  # default='warn'
from happyalgos.utils.filtering import window_ranger, window_maker
from happyalgos.activity.activity_counter import ActivityCounter
import tsfresh
import scipy
from sklearn.preprocessing import minmax_scale

import heapq
from alive_progress import alive_bar


# df = pd.read_parquet("/Users/lselig/Desktop/activity_validation_hr_activity_model_dev_v2/HR_ACTIVITY_20210608_1005_512_1/features_labels.parquet")
# print(list(df))
def extract_features(act_obj, acc_df):
    counts = act_obj.get_count( acc_slice.x.values, acc_slice.y.values, acc_slice.z.values, acc_slice.etime.values )
    print(counts)
    mag = np.sqrt(acc_df.x ** 2 + acc_df.y ** 2 + acc_df.z ** 2)
    f, pxx = signal.welch(
        mag,
        fs = 26,
        window = "hamming",
        nperseg = 256,
        noverlap = None,
        nfft = None,
        scaling = "density",
        detrend = "constant",
        average = "mean",
    )
    # plt.plot(pxx)
    # plt.show()

    max_pxx = np.nanmax( pxx )
    energy = tsfresh.feature_extraction.feature_calculators.abs_energy( mag )
    entropy = tsfresh.feature_extraction.feature_calculators.approximate_entropy( mag, m = 40, r = 0.8 )
    cid_ce = tsfresh.feature_extraction.feature_calculators.cid_ce( mag, normalize = True )
    peaks1 = tsfresh.feature_extraction.feature_calculators.number_peaks( mag, n = 1 )
    peaks2 = tsfresh.feature_extraction.feature_calculators.number_peaks( mag, n = 2 )
    peaks3 = tsfresh.feature_extraction.feature_calculators.number_peaks( mag, n = 3 )
    crossings1 = tsfresh.feature_extraction.feature_calculators.number_crossing_m( mag, m = 1 )
    mean = np.nanmean( mag )
    std = np.nanstd( mag )
    median = np.nanmedian( mag )

    # plt.hist(mag, bins = 30)
    # plt.show()

    features = {"counts"    : counts,
                "energy"    : energy,
                "entropy"   : entropy,
                "cid_ce"    : cid_ce,
                "peaks1"    : peaks1,
                "peaks2"    : peaks2,
                "peaks3"    : peaks3,
                "crossings1": crossings1,
                "mean"      : mean,
                "std"       : std,
                "median"    : median,
                "max_pxx"   : max_pxx}
    return features
get_data_hr_activity = True
if(get_data_hr_activity):
    for trial in glob.glob("/Volumes/GoogleDrive/Shared drives/DSCI-ANALYSIS/HR_ACTIVITY/*/"):
        trial_name = trial.split("/")[-2]
        # if("932764" not in trial_name):
        #     continue
        if ("202105" in trial or "694957" in trial or "510162" in trial or "1085_656" in trial or "1200_782" in trial):
            continue

        if("2377" in trial or '653218' in trial or "694967" in trial or "539786" in trial or "845613" in trial or "492477" in trial or "305432" in trial):
            continue
        trial_name = trial.split("/")[-2]

        stopwatch = pd.read_csv(glob.glob(trial + "StopWatch.csv")[0])
        eda_df = pd.read_csv(glob.glob(trial + "P6_*_EDA.csv")[0])
        acc_df = pd.read_csv(glob.glob(trial + "P6_*_ACC.csv")[0])

        eda_df["conductance"] = eda_df["conductance"] * 1e6
        eda_df["eda_fs_cur_na"] = [1] * eda_df.shape[0]
        eda_df["flags"] = [1] * eda_df.shape[0]
        eda_df["drive_voltage"] = [1] * eda_df.shape[0]


        # mid_activity = {,
        mid_activity =  {
                        "Elliptical low": ["Elliptical low", "Elliptical", "Low eliptical", "Ellip"],
                        "Bike low": ["Bike low", "Stationary bike", "Bike"],
                        # "Yoga low": ["Low yoga", "Yoga low", "Yoga/stretching (low)", "Yoga / stretching (low)"],
                        # "Yoga mod": ["Mod yoga", "Yoga mod", "Yoga/stretching (moderate)", "Yoga / stretching (moderate)"],
                        "Trans": ["Trans", "Transition"],
                        "Tread low": ["Low treadmill", "Low tread", "Tread low", "Baseline treadmill (low)", "Treadmill (low)", "Treadmill low"],
                        "Tread mod": ["Med treadmill", "Mod tread", "Tread mod", "Baseline treadmill (moderate)", "Treadmill (moderate)", "Treadmill moderate"],
                        "Outside low": ["Outside low", "Slow walk", "Outdoor walk (low)", "Outdoor low"],
                        "Outside mod": ["Outside mod", "Moderate walk", "Outdoor walk (moderate)", "Outdoor mod", "Outdoor (moderate)"],
                        "Recovery": ["Recovery", "Treadmill (recovery)", "Treadmill ( recovery)", "Tread recovery", "Treadmill recovery"],
                        "Walk natural": ["Walk natural"],
                        "Walk texting": ["Walk texting"],
                        "Walk pockets": ["Walk pockets"]
                       }

        # high_activity = {"Yoga high": ["High yoga", "Yoga high", "Yoga/stretching (high)", "Yoga / stretching (high)"],
        high_activity = {
                         "Tread high": ["High tread", "Tread high", "Baseline treadmill (high)", "Treadmill (high)", "Treadmill high"],
                         "Outside high": ["Outside high", "High walk/jog", "Jogging", "Outdoor jog (high)", "Outdoor high"]}

        fakes = {"Laundry": ["Laundry", "Standing laundry", "Standing (laundry)", "Folding", "Fold poncho"],
                 "Typing": ["Typing", "Sitting (typing)"],
                 "Rest": ["Rest"],
                 }
        for k in range(stopwatch.shape[0]):
            label = stopwatch.iloc[k].label

            for key in mid_activity:
                for value in mid_activity[key]:
                    if(label == value):
                        stopwatch.loc[k, 'label'] = key
                        # stopwatch.iloc[k].label = key

            for key in high_activity:
                for value in high_activity[key]:
                    if(label == value):
                        stopwatch.loc[k, 'label'] = key
                        # stopwatch.iloc[k].label = key

            for key in fakes:
                for value in fakes[key]:
                    if(label == value):
                        stopwatch.loc[k, 'label'] = key

        print(trial_name)
        try:
            idx_outside_low = stopwatch[stopwatch.label == "Outside low"].index.values[0]
            stopwatch.loc[idx_outside_low - 1, 'label'] = "Outside low"
            # if("653218" in trial_name):
            #     acc_df["etime"] = acc_df["etime"] + 3600 * -5
            idx_outside_high = stopwatch[stopwatch.label == "Outside high"].index.values[0]
            stopwatch = stopwatch[:idx_outside_high + 2]
        except:
            pass
        acc_df = acc_df[acc_df.etime <= stopwatch.iloc[-1].etime]
        # plt.plot(acc_df.etime, acc_df.x)
        # plt.axvline(stopwatch.iloc[0].etime)
        # plt.axvline(stopwatch.iloc[-1].etime)
        # plt.show()
        mid_activity = list(mid_activity.keys())
        high_activity = list(high_activity.keys())
        fakes = list(fakes.keys())
        acc_x = signal.decimate(acc_df.to_numpy()[:, 1], q = 4)
        acc_y = signal.decimate(acc_df.to_numpy()[:, 2], q = 4)
        acc_z = signal.decimate(acc_df.to_numpy()[:, 3], q = 4)

        start = acc_df.iloc[0].etime
        end = acc_df.iloc[-1].etime

        new_ts = np.linspace(start, end, num = len(acc_x))
        acc_26 = pd.DataFrame({"etime": new_ts,
                                "x": acc_x,
                                "y": acc_y,
                                "z": acc_z})
        if("514655" in trial_name):
            acc_26["etime"] = acc_26["etime"] + 60

        WINDOW, STRIDE = 20.0, 10.0
        window_ranges = window_ranger(
            acc_26.iloc[0].etime,
            acc_26.iloc[-1].etime,
            STRIDE,
            WINDOW,
        )

        stopwatch["my_dt"] = pd.to_datetime(stopwatch.etime, unit = "s")
        stopwatch = stopwatch.set_index( 'my_dt' ).resample( '1S' ).ffill().reset_index()
        stopwatch["etime"] = [x.timestamp() for x in stopwatch.my_dt]

        act_obj = ActivityCounter( window = WINDOW, stride = STRIDE )
        trial_features = {}
        # with alive_bar(len(window_ranges)) as bar:
        for i, window in enumerate(window_ranges):
            start = window[0]
            end = window[1]
            acc_slice = acc_26[acc_26.etime.between(start, end)]
            stopwatch_slice = stopwatch[stopwatch.etime.between( start, end )]
            stopwatch_slice = list(stopwatch_slice.label.values)
            # print( i, stopwatch_slice)
            uniques = list( set( stopwatch_slice ) )
            if (len( uniques ) != 1):
                print( f"Skipping: {uniques}" )
                continue
            curr_label = uniques[0]
            if (curr_label in mid_activity):
                label = "active"
            elif (curr_label in high_activity):
                label = "active"
            else:
                label = "inactive"

            if ( curr_label == "Trans" or curr_label == "Tread low" or curr_label == "Elliptical low" or curr_label == "Recovery" or curr_label == "Outside low"):
                multiclass_label = "low"
            elif (curr_label == "Tread mod" or curr_label == "Outside mod"):
                multiclass_label = "mod"
            elif (curr_label == "Tread high" or curr_label == "Outside high"):
                multiclass_label = "high"
            else:
                multiclass_label = "inactive"

            ts_anchor = np.nanmedian(acc_slice.etime.values)
            counts = act_obj.get_count( acc_slice.x.values, acc_slice.y.values, acc_slice.z.values, acc_slice.etime.values )
            mag = np.sqrt( acc_slice.x.values ** 2 + acc_slice.y.values ** 2 + acc_slice.z.values ** 2 )
            f, pxx = signal.welch(
                mag,
                fs = 26,
                window = "hamming",
                nperseg = 255,
                noverlap = None,
                nfft = None,
                scaling = "density",
                # scaling = "spectrum",
                detrend = "constant",
                average = "mean",
            )

            max_pxx = np.nanmax( pxx )
            energy = tsfresh.feature_extraction.feature_calculators.abs_energy( mag )
            # entropy = tsfresh.feature_extraction.feature_calculators.approximate_entropy( mag, m = 40, r = 0.8 )
            cid_ce = tsfresh.feature_extraction.feature_calculators.cid_ce( mag, normalize = True )
            peaks1 = tsfresh.feature_extraction.feature_calculators.number_peaks( mag, n = 1 )
            peaks2 = tsfresh.feature_extraction.feature_calculators.number_peaks( mag, n = 2 )
            peaks3 = tsfresh.feature_extraction.feature_calculators.number_peaks( mag, n = 3 )
            crossings1 = tsfresh.feature_extraction.feature_calculators.number_crossing_m( mag, m = 1 )
            mean = np.nanmean( mag )
            std = np.nanstd( mag )
            median = np.nanmedian( mag )
            skew = scipy.stats.skew(mag)
            kurtosis = scipy.stats.kurtosis( mag )
            cvar_time = (std ** 2) / mean

            pxx_list = list(pxx)
            top_3 = [pxx_list.index(x) for x in sorted(pxx_list, reverse=True)[:3]]
            peak_height = np.nanquantile(pxx, 0.95)
            peaks_pxx_idx = signal.find_peaks( pxx, height = peak_height)[0]

            pxx_scaled = minmax_scale(pxx)
            demographics = pd.read_csv("/Users/lselig/Desktop/activity_validation_hr_activity_model_dev/HR_ACTIVITY - DEMOGRAPHICS.csv")
            age = demographics[demographics.study_id == int(trial_name.split("_")[-3])][["age at trial"]].values[0][0]
            gender = demographics[demographics.study_id == int(trial_name.split("_")[-3])][["gender"]].values[0][0]
            height_cm = demographics[demographics.study_id == int(trial_name.split("_")[-3])][["height_cm"]].values[0][0]
            weight_kg = demographics[demographics.study_id == int(trial_name.split("_")[-3])][["weight_kg"]].values[0][0]

            peaks_pxx_idx = signal.find_peaks( pxx )[0]
            sorted_idx_pxx_peaks = np.flip( np.argsort( pxx[peaks_pxx_idx] ) )
            ordered_f_peaks = f[peaks_pxx_idx][sorted_idx_pxx_peaks]
            ordered_pxx_peaks = pxx[sorted_idx_pxx_peaks]
            pxx_sorted = np.flip( np.sort( pxx[peaks_pxx_idx] ) )

            peak_loc_names = ["f_biggest_peak", "f_second_biggest_peak", "f_third_biggest_peak"]
            peak_val_names = ["pxx_biggest_peak", "pxx_second_biggest_peak", "pxx_third_biggest_peak"]
            peak_val_vals = []
            peak_loc_vals = []
            for m in range( 3 ):
                # in case feature extractor doesn't find 3 peaks.
                # try:
                peak_loc_vals.append( ordered_f_peaks[m] )
                peak_val_vals.append( pxx_sorted[m] )

            peak_loc_dict = dict(zip(peak_loc_names, peak_loc_vals))
            peak_val_dict = dict(zip(peak_val_names, peak_val_vals))

            window_features = {"counts"                          : counts, "energy": energy, "cid_ce": cid_ce,
                               "peaks1"                          : peaks1, "peaks2": peaks2, "peaks3": peaks3,
                               "crossings1"                      : crossings1, "mean_time": mean, "std_time": std,
                               "median_time"                     : median, "max_pxx": max_pxx,
                               "log_pxx"                         : np.log( max_pxx ),
                               "var_pxx_peaks"                   : np.var( f[peaks_pxx_idx] ),
                               "sum_diff_pxx_peaks"              : np.sum( np.abs( np.diff( f[peaks_pxx_idx] ) ) ),
                               "peaks_dotproduct"                : (np.sum(
                                   f[peaks_pxx_idx] * pxx_scaled[peaks_pxx_idx] )) / len( peaks_pxx_idx ),
                               "sum_f_peaks"                     : np.sum( f[peaks_pxx_idx] ), "skewness_time": skew,
                               "kurtosis_time"                   : kurtosis, "skewness_freq": scipy.stats.skew( pxx ),
                               "kurotis_freq"                    : scipy.stats.kurtosis( pxx ),
                               "mean_freq"                       : np.mean( pxx ), "std_freq": np.std( pxx ),
                               "age"                             : age, "gender": gender, "height_cm": height_cm,
                               "weight_kg"                       : weight_kg, "true_binary": label,
                               "true_multi"                      : multiclass_label, "etime": ts_anchor,
                               "cvar_time": cvar_time,
                               "cvar_freq": np.var(pxx) / np.mean(pxx)}
            window_features.update(peak_loc_dict)
            window_features.update(peak_val_dict)
            a = 1


            # if(curr_label == "Tread low" or curr_label == "Outside low" or curr_label == "Tread mod" or curr_label == "Outside mod" or curr_label == "Recovery"):
            #     # peaks_pxx_idx = signal.find_peaks( pxx, height = peak_height )[0]
            #     peaks_pxx_idx = signal.find_peaks( pxx)[0]
            #     sorted_idx_pxx_peaks = np.flip(np.argsort(pxx[peaks_pxx_idx]))
            #     ordered_f_peaks = f[peaks_pxx_idx][sorted_idx_pxx_peaks]
            #     ordered_pxx_peaks = pxx[sorted_idx_pxx_peaks]
            #     pxx_sorted = np.flip(np.sort(pxx[peaks_pxx_idx]))
            #
            #     peak_loc_names = ["f_biggest_peak", "f_second_biggest_peak", "f_third_biggest_peak"]
            #     peak_val_names = ["pxx_biggest_peak", "pxx_second_biggest_peak", "pxx_third_biggest_peak"]
            #     peak_val_vals = []
            #     peak_loc_vals = []
            #     for m in range(3):
            #         # in case feature extractor doesn't find 3 peaks.
            #         # try:
            #         peak_loc_vals.append(ordered_f_peaks[m])
            #         peak_val_vals.append(pxx_sorted[m])
            #         # except:
            #         #     peak_loc_vals.append( f[sorted_idx_pxx_peaks[m - 1]] )
            #
            #
            #     # sort peaks by amplitude
            #     # hz_biggest_peak
            #     # hz_second_biggest_peak
            #     # hz_third_biggest_peak
            #     a = 1
            #     print(peak_loc_vals)
            #     print(peak_val_vals)
            #     plt.plot( f, pxx )
            #     plt.title( f"Welch's Periodgram\nWindow event: {curr_label}\nf biggest peak: {peak_loc_vals[0]}\nf 2nd biggest peak: {peak_loc_vals[1]}\nf 3rd peak: {peak_loc_vals[2]}")
            #     plt.xlabel( "Hz" )
            #     plt.ylabel( "Density" )
            #     plt.scatter( f[peaks_pxx_idx], pxx[peaks_pxx_idx], color = 'green', marker = 'x' )
            #     plt.show()

            freq_res = f[1] - f[0]
            bands_low = np.linspace(0, 13, num = 52)
            bands_025hz_higher = np.linspace(0, 13, num = 52) + 0.25
            bands_05hz_higher = np.linspace(0, 13, num = 52) + 0.5
            bands_1hz_higher = np.linspace(0, 13, num = 52) + 1
            higher = [bands_025hz_higher, bands_05hz_higher, bands_1hz_higher]
            # higher = [bands_1hz_higher]
            aucs = {}
            for j in range(len(bands_low)):
                for h in higher:
                    low, high = bands_low[j], h[j]
                    idx_delta = np.logical_and( f >= low, f <= high )
                    # print( f )
                    # print( idx_delta )
                    auc = np.trapz( pxx[idx_delta], dx = freq_res )
                    # print(j, h, auc)
                    # auc = np.trapz( pxx_scaled[idx_delta], dx = freq_res )
                    aucs["auc_" + str(np.round(low, 3)) + "_" + str(np.round(high, 3)) + "_hz"] = auc

            window_features.update( aucs )
            if not os.path.isdir( f"/Users/lselig/Desktop/activity_validation_hr_activity_model_dev_v2/{trial_name}" ):
                os.makedirs( f"/Users/lselig/Desktop/activity_validation_hr_activity_model_dev_v2/{trial_name}" )

            acc_df.to_parquet(
                f"/Users/lselig/Desktop/activity_validation_hr_activity_model_dev_v2/{trial_name}/acc_26.parquet",
                index = False )
            eda_df.to_parquet(
                f"/Users/lselig/Desktop/activity_validation_hr_activity_model_dev_v2/{trial_name}/eda.parquet",
                index = False )
            stopwatch.to_parquet(
                f"/Users/lselig/Desktop/activity_validation_hr_activity_model_dev_v2/{trial_name}/stopwatch.parquet",
                index = False )

            for f in window_features:
                if(f not in trial_features):
                    trial_features[f] = [window_features[f]]
                else:
                    trial_features[f] = trial_features[f] + [window_features[f]]

        trial_features = pd.DataFrame(trial_features)
        trial_features.to_parquet(
            f"/Users/lselig/Desktop/activity_validation_hr_activity_model_dev_v2/{trial_name}/features_labels.parquet",
            index = False )

        fig, axs = plt.subplots( 4, 1, figsize = (15, 9), sharex = True )
        # axs[0].plot(pd.to_datetime(acc_df.etime, unit = 's'), acc_df.x, label = "x")
        # axs[0].plot(pd.to_datetime(acc_df.etime, unit = "s"), acc_df.y, label = "y")
        # axs[0].plot(pd.to_datetime(acc_df.etime, unit = "s"), acc_df.z, label = "z")
        # axs[0].set_ylabel("xyz")
        # axs[0].legend()
        # axs[1].plot(pd.to_datetime(acc_df.etime, unit = "s"), np.sqrt(acc_df.x ** 2 + acc_df.y ** 2 + acc_df.z ** 2), label = "100hz")
        axs[0].plot( pd.to_datetime( acc_26.etime, unit = "s" ),
                     np.sqrt( acc_26.x ** 2 + acc_26.y ** 2 + acc_26.z ** 2 ), label = "26hz", color = "green")
        axs[0].set_ylabel( "Mag (gS)" )

        seen = []
        for j in range( stopwatch.shape[0] - 1 ):

            label = stopwatch.iloc[j].label
            if(label not in seen):
                seen.append(label)
            else:
                continue
            if (label in mid_activity):
                color = "orange"
            elif (label in high_activity):
                color = 'red'
            elif (label in fakes):
                color = "green"
            else:
                continue

            next_label_idx = j + 1
            for k in range(j + 1, stopwatch.shape[0] - 1):
                if(stopwatch.iloc[k].label != label):
                    next_label_idx = k
                    break
            for m in range(0, 4):
                axs[m].axvspan(
                    pd.to_datetime( stopwatch.iloc[j].etime, unit = "s" ),
                    pd.to_datetime( stopwatch.iloc[next_label_idx].etime, unit = "s" ),
                    color = color,
                    alpha = 0.4
                    # label = stopwatch_df.iloc[i].label,
                )

            mid_x = (stopwatch.iloc[j].etime + stopwatch.iloc[next_label_idx].etime) / 2
            # if j == 0:
            lower, upper = axs[0].get_ylim()
            mid_y = (lower + upper) / 2

            axs[0].text(
                pd.to_datetime( mid_x, unit = "s" ),
                mid_y,
                s = label.replace( " ", "\n" ),
                bbox = {
                    "facecolor": "white",
                    "alpha"    : 0,
                    "edgecolor": "black",
                    "pad"      : 1,
                },
                ha = "center",
                va = "center",
            )

        # axs[1].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features.pred, color = "red", alpha = 0.8,
        #              label = "Pred", marker = ".", drawstyle = "steps-pre" )
        axs[1].set_ylabel( "Class" )
        axs[1].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features.true_binary, color = "black", alpha = 0.8,
                     label = "True", marker = ".", drawstyle = "steps-pre" )
        axs[1].legend()
        axs[1].invert_yaxis()

        axs[2].set_ylabel( "Class" )
        axs[2].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features.true_multi, color = "black", alpha = 0.8,
                     label = "True", marker = ".", drawstyle = "steps-pre" )
        axs[2].legend()
        axs[2].invert_yaxis()

        axs[3].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features.counts, marker = "." )
        axs[3].set_ylabel( "Counts" )
        # axs[3].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features[["auc_1.0_2.6_hz"]], marker = "." )
        # axs[3].set_ylabel( "1.0 - 2.6" )
        # axs[4].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features[["auc_1.3_1.8_hz"]], marker = "." )
        # axs[4].set_ylabel( "1.3 - 1.8" )
        # axs[5].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features[["auc_1.6_2.6_hz"]], marker = "." )
        # axs[5].set_ylabel( "1.6 - 2.6" )
        # axs[6].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features[["auc_2.2_4.2_hz"]], marker = "." )
        # axs[6].set_ylabel( "2.2 - 4.2" )
        # axs[7].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features[["sum_f_peaks"]], marker = "." )
        # axs[7].set_ylabel( "Sum f peaks" )
        # axs[8].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features[["peaks_dotproduct"]], marker = "." , drawstyle = "steps-pre")
        # axs[8].set_ylabel( "peaks dot" )
        # axs[9].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features[["sum_diff_pxx_peaks"]], marker = "." )
        # axs[9].set_ylabel( "Sum diff pxx peaks" )
        # axs[10].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features[["var_pxx_peaks"]], marker = "." )
        # axs[10].set_ylabel( "Var pxx peaks" )
        # axs[7].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features.peaks2, marker = "." )
        # axs[7].set_ylabel( "Peaks2" )
        # axs[8].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features.peaks3, marker = "." )
        # axs[8].set_ylabel( "Peaks3" )
        # axs[9].plot(pd.to_datetime(pred_acc_df.etime, unit = "s"), pred_acc_df.energy, marker = ".")
        # axs[9].set_ylabel("Energy")
        axs[3].axhline( 16, color = "C3", alpha = 0.7, ls = "--", label = "Low thresh" )
        axs[3].axhline( 28, color = "C4", alpha = 0.7, ls = "--", label = "High thresh" )
        axs[3].legend()
        # hr_df = pd.read_parquet('/Users/lselig/Desktop/walking/1717/1717/20221024_124229-20221031_122847/h10_9F8E632B_hr.parquet')
        # ibi_df = pd.read_parquet('/Users/lselig/Desktop/walking/1717/1717/20221024_124229-20221031_122847/h10_9F8E632B_ibi.parquet')
        # ecg_df = pd.read_parquet('/Users/lselig/Desktop/walking/1717/1717/20221024_124229-20221031_122847/h10_9F8E632B_ecg.parquet')

        # axs[4].plot(pd.to_datetime(hr_df.ts, unit = "s"), hr_df.hr)
        # axs[4].set_ylabel("HR (BPM)")
        # axs[5].plot(pd.to_datetime(ecg_df.ts, unit = "s"), ecg_df.ecgSample)
        # axs[5].set_ylabel("ECG")
        axs[0].legend()
        fig.suptitle(trial_name)
        # fig.suptitle( "600104 Walking" )
        # plt.tight_layout()
        # plt.show()
        plt.savefig(f"/Users/lselig/Desktop/activity_validation_hr_activity_model_dev_v2/{trial_name}/view.png", dpi = 300)
        plt.close()


get_data_io = True
if(get_data_io):
    for trial in glob.glob("/Volumes/GoogleDrive/Shared drives/DSCI-ANALYSIS/MOVE_IT_STRESS/*/"):
        trial_name = trial.split("/")[-2]
        # if("932764" not in trial_name):
        #     continue
        # if ("202105" in trial or "694957" in trial or "510162" in trial or "1085_656" in trial or "1200_782" in trial):
        #     continue
        #
        # if("2377" in trial or '653218' in trial or "694967" in trial or "539786" in trial or "845613" in trial or "492477" in trial or "305432" in trial):
        #     continue
        trial_name = trial.split("/")[-2]

        stopwatch = pd.read_parquet(glob.glob(trial + "stopwatch.parquet")[0])
        stopwatch["label"] = stopwatch.lap_name

        # eda_df = pd.read_csv(glob.glob(trial + "P6_*_EDA.csv")[0])
        try:
            acc_df = pd.read_parquet(glob.glob(trial + "v1a_acc_df.parquet")[0])
            eda_df = pd.read_parquet(glob.glob(trial + "v1a_eda_df.parquet")[0])
        except:
            continue

        # eda_df["conductance"] = eda_df["conductance"] * 1e6
        # eda_df["eda_fs_cur_na"] = [1] * eda_df.shape[0]
        # eda_df["flags"] = [1] * eda_df.shape[0]
        # eda_df["drive_voltage"] = [1] * eda_df.shape[0]


        # mid_activity = {,
        # mid_activity =  {
        #                 "Elliptical low": ["Elliptical low", "Elliptical", "Low eliptical", "Ellip"],
        #                 "Bike low": ["Bike low", "Stationary bike", "Bike"],
        #                 # "Yoga low": ["Low yoga", "Yoga low", "Yoga/stretching (low)", "Yoga / stretching (low)"],
        #                 # "Yoga mod": ["Mod yoga", "Yoga mod", "Yoga/stretching (moderate)", "Yoga / stretching (moderate)"],
        #                 "Trans": ["Trans", "Transition"],
        #                 "Tread low": ["Low treadmill", "Low tread", "Tread low", "Baseline treadmill (low)", "Treadmill (low)", "Treadmill low"],
        #                 "Tread mod": ["Med treadmill", "Mod tread", "Tread mod", "Baseline treadmill (moderate)", "Treadmill (moderate)", "Treadmill moderate"],
        #                 "Outside low": ["Outside low", "Slow walk", "Outdoor walk (low)", "Outdoor low"],
        #                 "Outside mod": ["Outside mod", "Moderate walk", "Outdoor walk (moderate)", "Outdoor mod", "Outdoor (moderate)"],
        #                 "Recovery": ["Recovery", "Treadmill (recovery)", "Treadmill ( recovery)", "Tread recovery", "Treadmill recovery"],
        #                 "Walk natural": ["Walk natural"],
        #                 "Walk texting": ["Walk texting"],
        #                 "Walk pockets": ["Walk pockets"]
        #                }
        #
        # # high_activity = {"Yoga high": ["High yoga", "Yoga high", "Yoga/stretching (high)", "Yoga / stretching (high)"],
        # high_activity = {
        #                  "Tread high": ["High tread", "Tread high", "Baseline treadmill (high)", "Treadmill (high)", "Treadmill high"],
        #                  "Outside high": ["Outside high", "High walk/jog", "Jogging", "Outdoor jog (high)", "Outdoor high"]}
        #
        # fakes = {"Laundry": ["Laundry", "Standing laundry", "Standing (laundry)", "Folding", "Fold poncho"],
        #          "Typing": ["Typing", "Sitting (typing)"],
        #          "Rest": ["Rest"],
        #          }
        # for k in range(stopwatch.shape[0]):
        #     label = stopwatch.iloc[k].label
        #
        #     for key in mid_activity:
        #         for value in mid_activity[key]:
        #             if(label == value):
        #                 stopwatch.loc[k, 'label'] = key
        #                 # stopwatch.iloc[k].label = key
        #
        #     for key in high_activity:
        #         for value in high_activity[key]:
        #             if(label == value):
        #                 stopwatch.loc[k, 'label'] = key
        #                 # stopwatch.iloc[k].label = key
        #
        #     for key in fakes:
        #         for value in fakes[key]:
        #             if(label == value):
        #                 stopwatch.loc[k, 'label'] = key

        # print(trial_name)
        # try:
        #     idx_outside_low = stopwatch[stopwatch.label == "Outside low"].index.values[0]
        #     stopwatch.loc[idx_outside_low - 1, 'label'] = "Outside low"
        #     # if("653218" in trial_name):
        #     #     acc_df["etime"] = acc_df["etime"] + 3600 * -5
        #     idx_outside_high = stopwatch[stopwatch.label == "Outside high"].index.values[0]
        #     stopwatch = stopwatch[:idx_outside_high + 2]
        # except:
        #     pass
        acc_df = acc_df[(acc_df.etime >= stopwatch.iloc[0].etime) & (acc_df.etime <= stopwatch.iloc[-1].etime)]
        sos = signal.butter( 4, 10.0, fs = 26.0, output = "sos" )
        # sos =  signal.cheby1(N = 8, rp = 0.05, Wn = 0.8 / q, output='sos')
        x = signal.sosfilt( sos, acc_df.x.values )
        y = signal.sosfilt( sos, acc_df.y.values )
        z = signal.sosfilt( sos, acc_df.z.values )

        acc_26 = pd.DataFrame({"etime": acc_df.etime.values,
                               "x": x,
                               "y": y,
                               "z": z})

        print(acc_df.shape[0] / (stopwatch.iloc[-1].etime - stopwatch.iloc[0].etime))
        # plt.plot(acc_df.etime, acc_df.x)
        # plt.axvline(stopwatch.iloc[0].etime, color = "red")
        # plt.axvline(stopwatch.iloc[-1].etime, color = "red")
        # plt.show()
        # continue
        # mid_activity = list(mid_activity.keys())
        # high_activity = list(high_activity.keys())
        # fakes = list(fakes.keys())
        # acc_x = signal.decimate(acc_df.to_numpy()[:, 1], q = 4)
        # acc_y = signal.decimate(acc_df.to_numpy()[:, 2], q = 4)
        # acc_z = signal.decimate(acc_df.to_numpy()[:, 3], q = 4)
        #
        # start = acc_df.iloc[0].etime
        # end = acc_df.iloc[-1].etime
        #
        # new_ts = np.linspace(start, end, num = len(acc_x))
        # acc_26 = pd.DataFrame({"etime": new_ts,
        #                         "x": acc_x,
        #                         "y": acc_y,
        #                         "z": acc_z})
        # if("514655" in trial_name):
        #     acc_26["etime"] = acc_26["etime"] + 60

        WINDOW, STRIDE = 20.0, 10.0
        window_ranges = window_ranger(
            acc_26.iloc[0].etime,
            acc_26.iloc[-1].etime,
            STRIDE,
            WINDOW,
        )

        stopwatch["my_dt"] = pd.to_datetime(stopwatch.etime, unit = "s")
        stopwatch = stopwatch.set_index( 'my_dt' ).resample( '1S' ).ffill().reset_index()
        stopwatch["etime"] = [x.timestamp() for x in stopwatch.my_dt]

        act_obj = ActivityCounter( window = WINDOW, stride = STRIDE )
        trial_features = {}
        # with alive_bar(len(window_ranges)) as bar:
        for i, window in enumerate(window_ranges):
            start = window[0]
            end = window[1]
            acc_slice = acc_26[acc_26.etime.between(start, end)]
            stopwatch_slice = stopwatch[stopwatch.etime.between( start, end )]
            stopwatch_slice = list(stopwatch_slice.label.values)
            # print( i, stopwatch_slice)
            uniques = list( set( stopwatch_slice ) )
            if (len( uniques ) != 1):
                print( f"Skipping: {uniques}" )
                continue
            curr_label = uniques[0]
            print(curr_label)
            if (curr_label.lower() == "jumping jack" or curr_label.lower() == "jumping jacks" or curr_label.lower() == "taps"):
                label = "active"
            else:
                label = "inactive"

            # if ( curr_label == "Trans" or curr_label == "Tread low" or curr_label == "Elliptical low" or curr_label == "Recovery" or curr_label == "Outside low"):
            #     multiclass_label = "low"
            # elif (curr_label == "Tread mod" or curr_label == "Outside mod"):
            #     multiclass_label = "mod"
            # elif (curr_label == "Tread high" or curr_label == "Outside high"):
            #     multiclass_label = "high"
            # else:
            #     multiclass_label = "inactive"

            ts_anchor = np.nanmedian(acc_slice.etime.values)
            counts = act_obj.get_count( acc_slice.x.values, acc_slice.y.values, acc_slice.z.values, acc_slice.etime.values )
            mag = np.sqrt( acc_slice.x.values ** 2 + acc_slice.y.values ** 2 + acc_slice.z.values ** 2 )
            f, pxx = signal.welch(
                mag,
                fs = 26,
                window = "hamming",
                nperseg = 255,
                noverlap = None,
                nfft = None,
                scaling = "density",
                # scaling = "spectrum",
                detrend = "constant",
                average = "mean",
            )

            max_pxx = np.nanmax( pxx )
            energy = tsfresh.feature_extraction.feature_calculators.abs_energy( mag )
            # entropy = tsfresh.feature_extraction.feature_calculators.approximate_entropy( mag, m = 40, r = 0.8 )
            cid_ce = tsfresh.feature_extraction.feature_calculators.cid_ce( mag, normalize = True )
            peaks1 = tsfresh.feature_extraction.feature_calculators.number_peaks( mag, n = 1 )
            peaks2 = tsfresh.feature_extraction.feature_calculators.number_peaks( mag, n = 2 )
            peaks3 = tsfresh.feature_extraction.feature_calculators.number_peaks( mag, n = 3 )
            crossings1 = tsfresh.feature_extraction.feature_calculators.number_crossing_m( mag, m = 1 )
            mean = np.nanmean( mag )
            std = np.nanstd( mag )
            median = np.nanmedian( mag )
            skew = scipy.stats.skew(mag)
            kurtosis = scipy.stats.kurtosis( mag )
            cvar_time = (std ** 2) / mean

            pxx_list = list(pxx)
            top_3 = [pxx_list.index(x) for x in sorted(pxx_list, reverse=True)[:3]]
            peak_height = np.nanquantile(pxx, 0.95)
            peaks_pxx_idx = signal.find_peaks( pxx, height = peak_height)[0]

            pxx_scaled = minmax_scale(pxx)
            # demographics = pd.read_csv("/Users/lselig/Desktop/activity_validation_hr_activity_model_dev/HR_ACTIVITY - DEMOGRAPHICS.csv")
            # age = demographics[demographics.study_id == int(trial_name.split("_")[-3])][["age at trial"]].values[0][0]
            # gender = demographics[demographics.study_id == int(trial_name.split("_")[-3])][["gender"]].values[0][0]
            # height_cm = demographics[demographics.study_id == int(trial_name.split("_")[-3])][["height_cm"]].values[0][0]
            # weight_kg = demographics[demographics.study_id == int(trial_name.split("_")[-3])][["weight_kg"]].values[0][0]

            age = 40
            gender = "M"
            height_cm = 0
            weight_kg = 0

            peaks_pxx_idx = signal.find_peaks( pxx )[0]
            sorted_idx_pxx_peaks = np.flip( np.argsort( pxx[peaks_pxx_idx] ) )
            ordered_f_peaks = f[peaks_pxx_idx][sorted_idx_pxx_peaks]
            ordered_pxx_peaks = pxx[sorted_idx_pxx_peaks]
            pxx_sorted = np.flip( np.sort( pxx[peaks_pxx_idx] ) )

            peak_loc_names = ["f_biggest_peak", "f_second_biggest_peak", "f_third_biggest_peak"]
            peak_val_names = ["pxx_biggest_peak", "pxx_second_biggest_peak", "pxx_third_biggest_peak"]
            peak_val_vals = []
            peak_loc_vals = []
            for m in range( 3 ):
                # in case feature extractor doesn't find 3 peaks.
                try:
                    peak_loc_vals.append( ordered_f_peaks[m] )
                    peak_val_vals.append( pxx_sorted[m] )
                except:
                    peak_loc_vals.append( ordered_f_peaks[m -1] )
                    peak_val_vals.append( pxx_sorted[m-1] )



            peak_loc_dict = dict(zip(peak_loc_names, peak_loc_vals))
            peak_val_dict = dict(zip(peak_val_names, peak_val_vals))
            multiclass_label = "inactive"

            window_features = {"counts"                          : counts, "energy": energy, "cid_ce": cid_ce,
                               "peaks1"                          : peaks1, "peaks2": peaks2, "peaks3": peaks3,
                               "crossings1"                      : crossings1, "mean_time": mean, "std_time": std,
                               "median_time"                     : median, "max_pxx": max_pxx,
                               "log_pxx"                         : np.log( max_pxx ),
                               "var_pxx_peaks"                   : np.var( f[peaks_pxx_idx] ),
                               "sum_diff_pxx_peaks"              : np.sum( np.abs( np.diff( f[peaks_pxx_idx] ) ) ),
                               "peaks_dotproduct"                : (np.sum(
                                   f[peaks_pxx_idx] * pxx_scaled[peaks_pxx_idx] )) / len( peaks_pxx_idx ),
                               "sum_f_peaks"                     : np.sum( f[peaks_pxx_idx] ), "skewness_time": skew,
                               "kurtosis_time"                   : kurtosis, "skewness_freq": scipy.stats.skew( pxx ),
                               "kurotis_freq"                    : scipy.stats.kurtosis( pxx ),
                               "mean_freq"                       : np.mean( pxx ), "std_freq": np.std( pxx ),
                               "age"                             : age, "gender": gender, "height_cm": height_cm,
                               "weight_kg"                       : weight_kg, "true_binary": label,
                               "true_multi"                      : multiclass_label, "etime": ts_anchor,
                               "cvar_time": cvar_time,
                               "cvar_freq": np.var(pxx) / np.mean(pxx)}
            window_features.update(peak_loc_dict)
            window_features.update(peak_val_dict)
            a = 1


            # if(curr_label == "Tread low" or curr_label == "Outside low" or curr_label == "Tread mod" or curr_label == "Outside mod" or curr_label == "Recovery"):
            #     # peaks_pxx_idx = signal.find_peaks( pxx, height = peak_height )[0]
            #     peaks_pxx_idx = signal.find_peaks( pxx)[0]
            #     sorted_idx_pxx_peaks = np.flip(np.argsort(pxx[peaks_pxx_idx]))
            #     ordered_f_peaks = f[peaks_pxx_idx][sorted_idx_pxx_peaks]
            #     ordered_pxx_peaks = pxx[sorted_idx_pxx_peaks]
            #     pxx_sorted = np.flip(np.sort(pxx[peaks_pxx_idx]))
            #
            #     peak_loc_names = ["f_biggest_peak", "f_second_biggest_peak", "f_third_biggest_peak"]
            #     peak_val_names = ["pxx_biggest_peak", "pxx_second_biggest_peak", "pxx_third_biggest_peak"]
            #     peak_val_vals = []
            #     peak_loc_vals = []
            #     for m in range(3):
            #         # in case feature extractor doesn't find 3 peaks.
            #         # try:
            #         peak_loc_vals.append(ordered_f_peaks[m])
            #         peak_val_vals.append(pxx_sorted[m])
            #         # except:
            #         #     peak_loc_vals.append( f[sorted_idx_pxx_peaks[m - 1]] )
            #
            #
            #     # sort peaks by amplitude
            #     # hz_biggest_peak
            #     # hz_second_biggest_peak
            #     # hz_third_biggest_peak
            #     a = 1
            #     print(peak_loc_vals)
            #     print(peak_val_vals)
            # plt.plot( f, pxx )
            # plt.title( f"Welch's Periodgram\nWindow event: {curr_label}\nf biggest peak: {peak_loc_vals[0]}\nf 2nd biggest peak: {peak_loc_vals[1]}\nf 3rd peak: {peak_loc_vals[2]}")
            # plt.xlabel( "Hz" )
            # plt.ylabel( "Density" )
            # plt.scatter( f[peaks_pxx_idx], pxx[peaks_pxx_idx], color = 'green', marker = 'x' )
            # plt.show()

            freq_res = f[1] - f[0]
            bands_low = np.linspace(0, 13, num = 52)
            bands_025hz_higher = np.linspace(0, 13, num = 52) + 0.25
            bands_05hz_higher = np.linspace(0, 13, num = 52) + 0.5
            bands_1hz_higher = np.linspace(0, 13, num = 52) + 1
            higher = [bands_025hz_higher, bands_05hz_higher, bands_1hz_higher]
            # higher = [bands_1hz_higher]
            aucs = {}
            for j in range(len(bands_low)):
                for h in higher:
                    low, high = bands_low[j], h[j]
                    idx_delta = np.logical_and( f >= low, f <= high )
                    # print( f )
                    # print( idx_delta )
                    auc = np.trapz( pxx[idx_delta], dx = freq_res )
                    # print(j, h, auc)
                    # auc = np.trapz( pxx_scaled[idx_delta], dx = freq_res )
                    aucs["auc_" + str(np.round(low, 3)) + "_" + str(np.round(high, 3)) + "_hz"] = auc

            window_features.update( aucs )
            if not os.path.isdir( f"/Users/lselig/Desktop/activity_validation_hr_activity_model_dev_v2/{trial_name}" ):
                os.makedirs( f"/Users/lselig/Desktop/activity_validation_hr_activity_model_dev_v2/{trial_name}" )

            acc_df.to_parquet(
                f"/Users/lselig/Desktop/activity_validation_hr_activity_model_dev_v2/{trial_name}/acc_26.parquet",
                index = False )
            # eda_df.to_parquet(
            #     f"/Users/lselig/Desktop/activity_validation_hr_activity_model_dev_v2/{trial_name}/eda.parquet",
            #     index = False )
            stopwatch.to_parquet(
                f"/Users/lselig/Desktop/activity_validation_hr_activity_model_dev_v2/{trial_name}/stopwatch.parquet",
                index = False )

            for f in window_features:
                if(f not in trial_features):
                    trial_features[f] = [window_features[f]]
                else:
                    trial_features[f] = trial_features[f] + [window_features[f]]

        trial_features = pd.DataFrame(trial_features)
        trial_features.to_parquet(
            f"/Users/lselig/Desktop/activity_validation_hr_activity_model_dev_v2/{trial_name}/features_labels.parquet",
            index = False )

        fig, axs = plt.subplots( 5, 1, figsize = (15, 9), sharex = True )
        # axs[0].plot(pd.to_datetime(acc_df.etime, unit = 's'), acc_df.x, label = "x")
        # axs[0].plot(pd.to_datetime(acc_df.etime, unit = "s"), acc_df.y, label = "y")
        # axs[0].plot(pd.to_datetime(acc_df.etime, unit = "s"), acc_df.z, label = "z")
        # axs[0].set_ylabel("xyz")
        # axs[0].legend()
        # axs[1].plot(pd.to_datetime(acc_df.etime, unit = "s"), np.sqrt(acc_df.x ** 2 + acc_df.y ** 2 + acc_df.z ** 2), label = "100hz")
        axs[4].plot( pd.to_datetime( acc_26.etime, unit = "s" ),
                     np.sqrt( acc_26.x ** 2 + acc_26.y ** 2 + acc_26.z ** 2 ), label = "26hz", color = "green")
        axs[4].set_ylabel( "Mag (gS)" )
        axs[0].plot(pd.to_datetime(eda_df.etime, unit = "s"), eda_df.conductance)
        axs[0].set_ylabel("EDA")

        seen = []
        for j in range( stopwatch.shape[0] - 1 ):

            label = stopwatch.iloc[j].label
            if(label not in seen):
                seen.append(label)
            else:
                continue
            color = "C0"
            # if (label in mid_activity):
            #     color = "orange"
            # elif (label in high_activity):
            #     color = 'red'
            # elif (label in fakes):
            #     color = "green"
            # else:
            #     continue

            next_label_idx = j + 1
            for k in range(j + 1, stopwatch.shape[0] - 1):
                if(stopwatch.iloc[k].label != label):
                    next_label_idx = k
                    break
            for m in range(0, 4):
                axs[m].axvspan(
                    pd.to_datetime( stopwatch.iloc[j].etime, unit = "s" ),
                    pd.to_datetime( stopwatch.iloc[next_label_idx].etime, unit = "s" ),
                    color = color,
                    alpha = 0.4
                    # label = stopwatch_df.iloc[i].label,
                )

            mid_x = (stopwatch.iloc[j].etime + stopwatch.iloc[next_label_idx].etime) / 2
            # if j == 0:
            lower, upper = axs[0].get_ylim()
            mid_y = (lower + upper) / 2

            print(label)
            if(pd.isna(label)):
                label = ""
            axs[0].text(
                pd.to_datetime( mid_x, unit = "s" ),
                mid_y,
                s = label.replace( " ", "\n" ),
                bbox = {
                    "facecolor": "white",
                    "alpha"    : 0,
                    "edgecolor": "black",
                    "pad"      : 1,
                },
                ha = "center",
                va = "center",
            )

        # axs[1].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features.pred, color = "red", alpha = 0.8,
        #              label = "Pred", marker = ".", drawstyle = "steps-pre" )
        axs[1].set_ylabel( "Class" )
        axs[1].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features.true_binary, color = "black", alpha = 0.8,
                     label = "True", marker = ".", drawstyle = "steps-pre" )
        axs[1].legend()
        axs[1].invert_yaxis()

        axs[2].set_ylabel( "Class" )
        axs[2].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features.true_multi, color = "black", alpha = 0.8,
                     label = "True", marker = ".", drawstyle = "steps-pre" )
        axs[2].legend()
        axs[2].invert_yaxis()

        axs[3].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features.counts, marker = "." )
        axs[3].set_ylabel( "Counts" )
        # axs[3].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features[["auc_1.0_2.6_hz"]], marker = "." )
        # axs[3].set_ylabel( "1.0 - 2.6" )
        # axs[4].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features[["auc_1.3_1.8_hz"]], marker = "." )
        # axs[4].set_ylabel( "1.3 - 1.8" )
        # axs[5].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features[["auc_1.6_2.6_hz"]], marker = "." )
        # axs[5].set_ylabel( "1.6 - 2.6" )
        # axs[6].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features[["auc_2.2_4.2_hz"]], marker = "." )
        # axs[6].set_ylabel( "2.2 - 4.2" )
        # axs[7].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features[["sum_f_peaks"]], marker = "." )
        # axs[7].set_ylabel( "Sum f peaks" )
        # axs[8].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features[["peaks_dotproduct"]], marker = "." , drawstyle = "steps-pre")
        # axs[8].set_ylabel( "peaks dot" )
        # axs[9].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features[["sum_diff_pxx_peaks"]], marker = "." )
        # axs[9].set_ylabel( "Sum diff pxx peaks" )
        # axs[10].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features[["var_pxx_peaks"]], marker = "." )
        # axs[10].set_ylabel( "Var pxx peaks" )
        # axs[7].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features.peaks2, marker = "." )
        # axs[7].set_ylabel( "Peaks2" )
        # axs[8].plot( pd.to_datetime( trial_features.etime, unit = "s" ), trial_features.peaks3, marker = "." )
        # axs[8].set_ylabel( "Peaks3" )
        # axs[9].plot(pd.to_datetime(pred_acc_df.etime, unit = "s"), pred_acc_df.energy, marker = ".")
        # axs[9].set_ylabel("Energy")
        axs[3].axhline( 16, color = "C3", alpha = 0.7, ls = "--", label = "Low thresh" )
        axs[3].axhline( 28, color = "C4", alpha = 0.7, ls = "--", label = "High thresh" )
        axs[3].legend()
        # hr_df = pd.read_parquet('/Users/lselig/Desktop/walking/1717/1717/20221024_124229-20221031_122847/h10_9F8E632B_hr.parquet')
        # ibi_df = pd.read_parquet('/Users/lselig/Desktop/walking/1717/1717/20221024_124229-20221031_122847/h10_9F8E632B_ibi.parquet')
        # ecg_df = pd.read_parquet('/Users/lselig/Desktop/walking/1717/1717/20221024_124229-20221031_122847/h10_9F8E632B_ecg.parquet')

        # axs[4].plot(pd.to_datetime(hr_df.ts, unit = "s"), hr_df.hr)
        # axs[4].set_ylabel("HR (BPM)")
        # axs[5].plot(pd.to_datetime(ecg_df.ts, unit = "s"), ecg_df.ecgSample)
        # axs[5].set_ylabel("ECG")
        axs[0].legend()
        fig.suptitle(trial_name)
        # fig.suptitle( "600104 Walking" )
        # plt.tight_layout()
        # plt.show()
        plt.savefig(f"/Users/lselig/Desktop/activity_validation_hr_activity_model_dev_v2/{trial_name}/view.png", dpi = 300)
        plt.close()
