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
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

all_features = []
MODE = "multi"
if(MODE == "binary"):
    target_names = ["Inactive", "Active"]

if(MODE == "multi"):
    target_names = ["Inactive", "Low", "Mod", "High"]

for trial in glob.glob("/Users/lselig/Desktop/activity_validation_hr_activity_model_dev/*/"):
    # if("HR_ACTIVITY_20210630_282723_827_1" in trial):
    #     continue
    features = pd.read_parquet(trial + "/features_labels.parquet")
    print(features)
    all_features.append(features)

all_features = pd.concat(all_features)
y_df = all_features.pop(f"true_{MODE}")
if(MODE == "multi"):
    y_df = y_df.replace("high", 3)
    y_df = y_df.replace("mod", 2)
    y_df = y_df.replace("low", 1)
    y_df = y_df.replace("inactive", 0)
else:
    y_df = y_df.replace("active", 1)
    y_df = y_df.replace("inactive", 0)



X_df = all_features.drop(columns = ["etime", "true_multi" if MODE == "binary" else "true_binary"])
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size = 0.2, random_state = 42)
a = 1

# clf = tree.DecisionTreeClassifier(random_state = 4)
clf = RandomForestClassifier(random_state = 4)
clf = clf.fit(X_train, y_train)
importances = clf.feature_importances_
y_pred = clf.predict(X_test)
plt.plot(importances)
plt.show()


class_report = classification_report(y_test,
                                     y_pred,
                                     target_names = target_names,
                                     output_dict=True,
                                     digits=4)
print(class_report)


for trial in glob.glob("/Users/lselig/Desktop/activity_validation_hr_activity_model_dev/*/"):
    test_trial = trial.split("/")[-2]
    # test_trial = "HR_ACTIVITY_20210630_282723_827_1"
    df = pd.read_parquet(f"/Users/lselig/Desktop/activity_validation_hr_activity_model_dev/{test_trial}/features_labels.parquet")
    acc_26 = pd.read_parquet(f"/Users/lselig/Desktop/activity_validation_hr_activity_model_dev/{test_trial}/acc_26.parquet")
    stopwatch = pd.read_parquet(f"/Users/lselig/Desktop/activity_validation_hr_activity_model_dev/{test_trial}/stopwatch.parquet")

    y_df = df.pop(f"true_{MODE}")
    if (MODE == "multi"):
        y_df = y_df.replace( "high", 3 )
        y_df = y_df.replace( "mod", 2 )
        y_df = y_df.replace( "low", 1 )
        y_df = y_df.replace( "inactive", 0 )
    else:
        y_df = y_df.replace( "active", 1 )
        y_df = y_df.replace( "inactive", 0 )
    timestamps = df.pop("etime")
    X_df = df.drop(columns = ["true_multi" if MODE == "binary" else "true_binary"])
    y_pred = clf.predict(X_df)


    naxis = 3
    fig, axs = plt.subplots(naxis, 1, figsize = (15, 9), sharex = True )
    # axs[0].plot(pd.to_datetime(acc_df.etime, unit = 's'), acc_df.x, label = "x")
    # axs[0].plot(pd.to_datetime(acc_df.etime, unit = "s"), acc_df.y, label = "y")
    # axs[0].plot(pd.to_datetime(acc_df.etime, unit = "s"), acc_df.z, label = "z")
    # axs[0].set_ylabel("xyz")
    # axs[0].legend()
    # axs[1].plot(pd.to_datetime(acc_df.etime, unit = "s"), np.sqrt(acc_df.x ** 2 + acc_df.y ** 2 + acc_df.z ** 2), label = "100hz")
    axs[0].plot( pd.to_datetime( acc_26.etime, unit = "s" ),
                 np.sqrt( acc_26.x ** 2 + acc_26.y ** 2 + acc_26.z ** 2 ), label = "26hz", color = "green")
    axs[0].set_ylabel( "Mag (gS)" )
    mid_activity =  {
                            "Elliptical low": ["Elliptical low", "Elliptical", "Low eliptical", "Ellip"],
                            # "Yoga low": ["Low yoga", "Yoga low", "Yoga/stretching (low)", "Yoga / stretching (low)"],
                            # "Yoga mod": ["Mod yoga", "Yoga mod", "Yoga/stretching (moderate)", "Yoga / stretching (moderate)"],
                            "Tread low": ["Low treadmill", "Low tread", "Tread low", "Baseline treadmill (low)", "Treadmill (low)", "Treadmill low"],
                            "Tread mod": ["Med treadmill", "Mod tread", "Tread mod", "Baseline treadmill (moderate)", "Treadmill (moderate)", "Treadmill moderate"],
                            "Outside low": ["Outside low", "Slow walk", "Outdoor walk (low)", "Outdoor low"],
                            "Outside mod": ["Outside mod", "Moderate walk", "Outdoor walk (moderate)", "Outdoor mod", "Outdoor (moderate)"],
                            "Recovery": ["Recovery", "Treadmill (recovery)", "Treadmill ( recovery)", "Tread recovery"],
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
                 "Rest": ["Rest"]}
    mid_activity = list( mid_activity.keys() )
    high_activity = list( high_activity.keys() )
    fakes = list( fakes.keys() )

    for j in range( stopwatch.shape[0] - 1 ):

        label = stopwatch.iloc[j].label
        if (label in mid_activity):
            color = "orange"
        elif (label in high_activity):
            color = 'red'
        elif (label in fakes):
            color = "green"
        else:
            continue
        for m in range(0, naxis):
            axs[m].axvspan(
                pd.to_datetime( stopwatch.iloc[j].etime, unit = "s" ),
                pd.to_datetime( stopwatch.iloc[j + 1].etime, unit = "s" ),
                color = color,
                alpha = 0.4
                # label = stopwatch_df.iloc[i].label,
            )

        mid_x = (stopwatch.iloc[j].etime + stopwatch.iloc[j + 1].etime) / 2
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
    axs[1].plot( pd.to_datetime( timestamps, unit = "s" ), y_df, color = "black", alpha = 0.8,
                 label = "True", marker = ".", drawstyle = "steps-pre" )

    # axs[2].set_ylabel( "Class" )
    axs[1].plot( pd.to_datetime( timestamps, unit = "s" ), y_pred, color = "red", alpha = 0.8,
                 label = "Pred", marker = ".", drawstyle = "steps-pre" )
    axs[1].legend()
    axs[1].invert_yaxis()
    # axs[2].legend()
    # axs[2].invert_yaxis()

    axs[2].plot( pd.to_datetime( timestamps, unit = "s" ), X_df.counts, marker = "." )
    axs[2].set_ylabel( "Counts" )
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
    # axs[3].axhline( 16, color = "C3", alpha = 0.7, ls = "--", label = "Low thresh" )
    # axs[3].axhline( 28, color = "C4", alpha = 0.7, ls = "--", label = "High thresh" )
    # axs[3].legend()
    # hr_df = pd.read_parquet('/Users/lselig/Desktop/walking/1717/1717/20221024_124229-20221031_122847/h10_9F8E632B_hr.parquet')
    # ibi_df = pd.read_parquet('/Users/lselig/Desktop/walking/1717/1717/20221024_124229-20221031_122847/h10_9F8E632B_ibi.parquet')
    # ecg_df = pd.read_parquet('/Users/lselig/Desktop/walking/1717/1717/20221024_124229-20221031_122847/h10_9F8E632B_ecg.parquet')

    # axs[4].plot(pd.to_datetime(hr_df.ts, unit = "s"), hr_df.hr)
    # axs[4].set_ylabel("HR (BPM)")
    # axs[5].plot(pd.to_datetime(ecg_df.ts, unit = "s"), ecg_df.ecgSample)
    # axs[5].set_ylabel("ECG")
    axs[0].legend()
    # fig.suptitle( "600104 Walking" )
    # plt.tight_layout()
    # plt.show()
    # plt.savefig(f"/Users/lselig/Desktop/activity_validation_hr_activity_model_dev/{trial_name}/view.png", dpi = 300)
    plt.show()
    plt.close()
