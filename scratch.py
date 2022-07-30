import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob, pickle, csv, re, timeit
import csv, sqlite3, os, traceback
import seaborn as sns
sns.set_style("darkgrid")
np.random.seed(42)

df = pd.read_parquet("/Users/lselig/Desktop/staging_data/SAVER_SCREEN_Harlie/P9/e91505000125/e91505000125_eda.parquet")
# df = df[df.etime.between(1658941200, 1658941200 + 3600)]
# plt.plot(pd.to_datetime(df.etime, unit = "s"), df.conductance)
# plt.show()
fig, axs = plt.subplots(2, 1, figsize = (15, 9))
df["resistance"] = 1e6 / df.conductance
df["etime"] = df["etime"] * 1e9
df = df.sort_values(by = ["etime"])
df = df.drop_duplicates(subset=['etime'])
axs[0].plot(pd.to_datetime(df.etime * 1e-9, unit = "s"), df.conductance)
df.to_parquet("/Users/lselig/Desktop/tmp_c_val/harlie_staging/eda-data.parquet", index = False)
df = pd.read_parquet("/Users/lselig/Desktop/staging_data/SAVER_SCREEN_Harlie/P9/e91505000125/e91505000125_acc.parquet")
# df = df[df.etime.between(1658941200, 1658941200 + 3600)]
df["etime"] = df["etime"] * 1e9
df = df.sort_values(by = ["etime"])
df = df.drop_duplicates(subset=['etime'])
axs[1].plot(pd.to_datetime(df.etime * 1e-9, unit = "s"), df.mag)
df.to_parquet("/Users/lselig/Desktop/tmp_c_val/harlie_staging/accel-data.parquet", index = False)
plt.show()


# df = pd.read_parquet("/Users/lselig/Desktop/staging_data/SAVER_SCREEN_Chris/P9/eb211600004a/eb211600004a_eda.parquet")
# df["resistance"] = 1e6 / df.conductance
# df["etime"] = df["etime"] * 1e9
# df = pd.to_parquet("/Users/lselig/Desktop/tmp_c_val/chris_staging/eda-data.parquet", index = False)



#
# df = pd.read_parquet("/Users/lselig/Desktop/staging_data/SAVER_SCREEN_Harlie/P9/e91505000125/e91505000125_eda.parquet")
# plt.plot(df.etime, df.conductance)
# # df = pd.read_parquet("/Users/lselig/Desktop/tmp_c_val/3789_1654772301/eda-data.parquet")
# df = pd.read_parquet("/Users/lselig/Desktop/move_it_stress_val/MOVE_IT_3789/v0_3789/cps_svr_minmax_60/runner_eda/eda_1654772901.parquet")
# start = 1654773196
# end = 1654773445
# fig, axs = plt.subplots(2, 1, figsize = (15, 9))
# df = df[(df.etime <= start) | (df.etime >= end)]
# df["resistance"] = 1e6 / df.conductance
# df["etime"] = df["etime"] * 1e9
# df.to_parquet("/Users/lselig/Desktop/tmp_c_val/3789_1654772301/eda-data.parquet", index = False)
# axs[0].plot(pd.to_datetime(df.etime * 1e-9, unit = "s"), df.conductance)
#
# df = pd.read_parquet("/Users/lselig/Desktop/move_it_stress_val/MOVE_IT_3789/v0_3789/cps_svr_minmax_60/runner_acc/acc_1654772901.parquet")
# start = 1654773196
# end = 1654773445
# df = df[(df.etime <= start) | (df.etime >= end)]
# df["etime"] = df["etime"] * 1e9
# df.to_parquet("/Users/lselig/Desktop/tmp_c_val/3789_1654772301/accel-data.parquet", index = False)
# axs[1].plot(pd.to_datetime(df.etime * 1e-9, unit = "s"), df.x)
# plt.show()

# df = pd.read_parquet("/Users/lselig/Desktop/tmp_c_val/3789_1654857976/accel-data.parquet")
# df = df[1200:]
# df.to_parquet("/Users/lselig/Desktop/tmp_c_val/3789_1654857976/accel-data.parquet")
#
# df1 = pd.read_parquet("/Volumes/GoogleDrive/Shared drives/DSCI-ANALYSIS/SAVER_STRESS/SAVER_3789_C1_S1/EDA_00AE.parquet")
# df2 = pd.read_parquet("/Volumes/GoogleDrive/Shared drives/DSCI-ANALYSIS/SAVER_STRESS/SAVER_3789_C1_S1/EDA_0040.parquet")
# print(df1.conductance.values, df2.conductance.values)
#
#
# def zoom_factory(ax, max_xlim, max_ylim, base_scale=2.0):
#     def zoom_fun(event):
#         # get the current x and y limits
#         cur_xlim = ax.get_xlim()
#         cur_ylim = ax.get_ylim()
#         xdata = event.xdata  # get event x location
#         ydata = event.ydata  # get event y location
#         if event.button == "up":
#             # deal with zoom in
#             scale_factor = 1 / base_scale
#             x_scale = scale_factor / 2
#         elif event.button == "down":
#             # deal with zoom out
#             scale_factor = base_scale
#             x_scale = scale_factor * 2
#         else:
#             # deal with something that should never happen
#             scale_factor = 1
#             print(event.button)
#         # set new limits
#         new_width = (cur_xlim[1] - cur_xlim[0]) * x_scale
#         new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
#
#         relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
#         rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
#
#         if xdata - new_width * (1 - relx) > max_xlim[0]:
#             x_min = xdata - new_width * (1 - relx)
#         else:
#             x_min = max_xlim[0]
#         if xdata + new_width * (relx) < max_xlim[1]:
#             x_max = xdata + new_width * (relx)
#         else:
#             x_max = max_xlim[1]
#         if ydata - new_height * (1 - rely) > max_ylim[0]:
#             y_min = ydata - new_height * (1 - rely)
#         else:
#             y_min = max_ylim[0]
#         if ydata + new_height * (rely) < max_ylim[1]:
#             y_max = ydata + new_height * (rely)
#         else:
#             y_max = max_ylim[1]
#         ax.set_xlim([x_min, x_max])
#         ax.set_ylim([y_min, y_max])
#         ax.figure.canvas.draw()
#
#     fig = ax.get_figure()  # get the figure of interest
#     # attach the call back
#     fig.canvas.mpl_connect("scroll_event", zoom_fun)
#
#     # return the function
#     return zoom_fun
#
#
# annotated_csv = "/Users/lselig/Desktop/annotations.csv"
# try:
#     annotated_df = pd.read_csv(annotated_csv)
# except:
#     annotated_df = pd.DataFrame(
#         {
#             "marked_ts": [],
#         }
#     )
#
# figure_mosaic = """
#                 AAAAA
#                 """
#
# fig, axs = plt.subplot_mosaic( figure_mosaic, figsize = (15, 9) )
# df_csv =  pd.read_csv("/Users/lselig/Desktop/ex_eda.csv")
# df = pd.read_parquet("/Users/lselig/Desktop/happy_data/data/pq/ea148700010f/0B09C3CF-C70D-46C4-BE89-E0BD26481621/0B09C3CF-C70D-46C4-BE89-E0BD26481621-HH_010f-20220721_082821/eda.parquet")
# axs["A"].plot(df.etime, df.conductance, label = "parquet")
# max_xlim = axs["A"].get_xlim()  # get current x_limits to set max zoom out
# max_ylim = axs["A"].get_ylim()  # get current y_limits to set max zoom out
# f = zoom_factory( axs["A"], max_xlim, max_ylim, base_scale = 1.1 )
# clicks = plt.ginput(-1, timeout=0)
#
# clicks = list(np.vstack(clicks)[:, 0])
# for i, click in enumerate(clicks):
#     event = clicks[i]
#     annotated_df = annotated_df.append(
#         pd.DataFrame(
#             {
#                 "marked_ts": [event * 24 * 60 * 60],
#             }
#         )
#     )
#     annotated_df = annotated_df.reset_index(drop=True)
#     annotated_df.to_csv(annotated_csv, index=False)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# start = timeit.default_timer()
# n = 10000000
# x = np.random.normal(0, 1, size = n)
# u = np.ones(n)
# for i, val in enumerate(x):
#     u[i] = np.sqrt(x[i] * 4 / 3 + 23 - 10) ** 2
#
# end = timeit.default_timer()
# plt.hist(u)
# plt.title(str(end - start))
# plt.show()
#
# # print(end - start)
# # df = pd.read_parquet("/Users/lselig/Desktop/pq_tmp/HH_0059/D300144A-64FC-44EB-BCFC-1AC11630F066-HH_0059-20220722_172055/eda.parquet")
# # plt.plot(df.etime, df.conductance)
# # plt.show()
# #
# #
# # plt.legend()
# # plt.show()
# #
# # df = pd.read_parquet("/Users/lselig/Desktop/tmp/eda.parquet")
# # print(df["device_name"].unique())
# #
# # df_0170 = df[df.device_name == "HH_0170"]
# # df_0170 = df_0170.sort_values(by = ["ts"])
# # plt.plot(df_0170.ts * 1e-9, 1e6 / df_0170.resistance, label = "HH_0170")
# #
# # df_010f = df[df.device_name == "HH_010f"]
# # df_010f = df_010f.sort_values(by = ["ts"])
# #
# # plt.plot(df_010f.ts * 1e-9, 1e6 / df_010f.resistance, label = "HH_010f")
# # plt.show()
# # plt.legend()
# # plt.show()
#
#
#     # ring2 = "ea148700010f"
#     # ring1 = "ea2116000170"
# # ring1 = "eb1532000059"
# # ring2 = "eb211600005b"
#
# eda_data = [pd.read_parquet(x) for x in glob.glob(f"/Users/lselig/Desktop/pq_tmp/HH_005b/*/eda.parquet")]
# acc_data = [pd.read_parquet(x) for x in glob.glob(f"/Users/lselig/Desktop/pq_tmp/HH_005b/*/acc.parquet")]
# # ppg_data = [pd.read_parquet(x) for x in glob.glob(f"/Users/lselig/Desktop/pq_tmp/HH_0059/*/ppg.parquet")]
# # skintemp_data = [pd.read_parquet(x) for x in glob.glob(f"/Users/lselig/Desktop/pq_tmp/HH_0059/*/skintemp.parquet")]
# # ambtemp_data = [pd.read_parquet(x) for x in glob.glob(f"/Users/lselig/Desktop/pq_tmp/HH_0059/*/ambtemp.parquet")]
#
# eda = pd.concat(eda_data)
# acc = pd.concat(acc_data)
# # ppg = pd.concat(ppg_data)
# # skintemp = pd.concat(skintemp_data)
# # ambtemp = pd.concat(ambtemp_data)
#
# eda = eda.sort_values(by=["etime"])
# acc = acc.sort_values(by=["etime"])
# # ppg = ppg.sort_values(by=["etime"])
# # skintemp = skintemp.sort_values(by=["etime"])
# # ambtemp = ambtemp.sort_values(by=["etime"])
#
# eda["etime"] = eda.etime + 3600 * -5
# acc["etime"] = acc.etime + 3600 * -5
#
#
# eda_data2 = [pd.read_parquet(x) for x in glob.glob(f"/Users/lselig/Desktop/pq_tmp/HH_0059/*/eda.parquet")]
# acc_data2 = [pd.read_parquet(x) for x in glob.glob(f"/Users/lselig/Desktop/pq_tmp/HH_0059/*/acc.parquet")]
# eda2 = pd.concat(eda_data2)
# acc2 = pd.concat(acc_data2)
# eda2 = eda2.sort_values(by=["etime"])
# acc2 = acc2.sort_values(by=["etime"])
# eda2["etime"] = eda2.etime + 3600 * -5
# acc2["etime"] = acc2.etime + 3600 * -5
#
#
# fig, axs = plt.subplots(3, 1, figsize = (15, 9), sharex = True)
# axs[0].plot(pd.to_datetime(eda.etime, unit = "s"), eda.conductance, color = "red", label = "Factory")
# axs[1].plot(pd.to_datetime(acc.etime, unit = "s"), acc.mag, color = "red", label = "Factory")
# axs[2].plot(pd.to_datetime(eda.etime, unit = "s"), eda.drive_voltage, color = "red", label = "Factory")
#
# axs[0].plot(pd.to_datetime(eda2.etime, unit = "s"), eda2.conductance, color = "black", label = "Du (hand modified)")
# axs[1].plot(pd.to_datetime(acc2.etime, unit = "s"), acc2.mag, color = "black", label = "Du (hand modified)")
# axs[2].plot(pd.to_datetime(eda2.etime, unit = "s"), eda2.drive_voltage, color = "black", label = "Du (hand modified)")
# axs[0].legend()
# axs[1].legend()
# axs[2].legend()
# factory_avg = np.nanmedian(eda.conductance)
# du_avg = np.nanmedian(eda2.conductance)
# fig.suptitle(f"Lucas weekend wear\n"
#              f"Factory ring median EDA (µS): {np.round(factory_avg, 3)}\n"
#              f"Du ring median EDA (µS): {np.round(du_avg, 3)}")
# # axs[2].plot(pd.to_datetime(ppg.etime, unit = "s"), ppg.green, color = "red")
# # axs[2].plot(pd.to_datetime(ppg.etime, unit = "s"), ppg.ir, color = "black")
# # axs[2].plot(pd.to_datetime(ppg.etime, unit = "s"), ppg.red, color = "purple")
# # axs[3].plot(pd.to_datetime(ambtemp.etime, unit = "s"), ambtemp.temperature, color = "red")
# # axs[3].plot(pd.to_datetime(skintemp.etime, unit = "s"), skintemp.temperature, color = "black")
#
# axs[0].set_ylabel("EDA (µS)")
# axs[1].set_ylabel("Acc (gS)")
# axs[2].set_ylabel("Drive Voltage (V)")
#
#
# # axs[0].plot(pd.to_datetime(eda2.etime, unit = "s"), eda2.conductance, label = "factory ring", color = "black")
# # axs[1].plot(pd.to_datetime(acc2.etime, unit = "s"), acc2.mag, label = "factory ring", color = "black")
# # axs[2].plot(pd.to_datetime(eda2.etime, unit = "s"), eda2.drive_voltage / 1e12, label = "factory ring", color = "black")
# # axs[0].set_ylabel("EDA (µS)")
# # axs[1].set_ylabel("Acc (gS)")
# # axs[2].set_ylabel("Drive Voltage (V)")
# #
# # axs[0].legend()
# # axs[1].legend()
# # axs[2].legend()
# axs[2].set_xlabel("Time (CST)")
# plt.show()
#
#
# df = pd.read_parquet("/Users/lselig/Desktop/labeled_sleep/0B09C3CF-C70D-46C4-BE89-E0BD26481621_04042022.parquet")
#
# dir = glob.glob("/Users/lselig/Desktop/labeled_sleep_3numtaps/*")
# seen = []
# for f in dir:
#     id = f.split("/")[-1].split(".parquet")[0].split("-")[0]
#     if(id not in seen):
#         seen.append(id)
#     else:
#         pass
# print(len(seen))
# print(np.nanmin(df.acc_x), np.nanmax(df.acc_x))
# print(np.nanmin(df.acc_y), np.nanmax(df.acc_y))
# print(np.nanmin(df.acc_z), np.nanmax(df.acc_z))
#
# df = pd.read_parquet("/Users/lselig/Desktop/tmp_c_val/3789_1654857976/eda-data.parquet")
# plt.plot(df.etime, df.conductance)
# plt.show()
# nt3 = pd.read_parquet("/Users/lselig/Desktop/labeled_sleep_3numtaps/FDCD0BFE-165F-4774-B457-E444C6EE6EC9_03152022.parquet")
# nt20 = pd.read_parquet("/Users/lselig/Desktop/labeled_sleep/FDCD0BFE-165F-4774-B457-E444C6EE6EC9_03152022.parquet")
# fig, axs = plt.subplots(2, 1, figsize = (15, 9), sharex = True)
# axs[0].plot(nt3.ts, nt3.acc_mean, label = "nt3")
# axs[0].plot(nt20.ts, nt20.acc_mean, label = "nt20")
# axs[1].plot(nt3.ts, nt3.acc_var, label = "nt3")
# axs[1].plot(nt20.ts, nt20.acc_var, label = "nt20")
# axs[0].legend()
# axs[1].legend()
# plt.show()
#
#
#
# df = pd.read_parquet("/Volumes/GoogleDrive/Shared drives/DSCI-DATA/SAVER/SAVER_216694/v1a_1/eda.parquet")
# plt.plot(1e6 / df.resistance.values)
# plt.show()
# def ttest():
#
#     # df = pd.read_parquet("/Volumes/GoogleDrive/Shared drives/Partners - Studies Data sets/for_conrad_tmp/1393_1654772301/eda.parquet")
#     df = pd.read_parquet("/Volumes/GoogleDrive/Shared drives/DSCI-ANALYSIS/HAPPY_PADS/Matt/Default/eda.parquet")
#     control = df.conductance.values
#     # a = -10
#     treatment = pd.read_parquet("/Volumes/GoogleDrive/Shared drives/DSCI-ANALYSIS/HAPPY_PADS/Matt/V1a/eda.parquet")
#     # treatment = control + skewnorm.rvs(a, scale = 0.03, loc = 0.00002, size=len(control)) # theoretical V1a with generally increased baseline EDA
#     # treatment = savgol_filter(treatment, 19, 2)
#     from scipy import stats
#     # result = stats.ttest_ind(control, treatment)
#     # print(result)
#     plt.plot(control, label = f"V0 - baseline: {np.nanmean(control)}")
#     plt.plot(treatment.conductance.values, label = f"V1a - baseline: {np.nanmean(treatment.conductance.values)}")
#     plt.legend()
#     plt.show()
#
#     return result
#
# print (ttest())
#
#
# def my_mean(x):
#     return np.sum(x) / len(x)
#
# def my_var(x):
#     ret = []
#     mu = my_mean(x)
#     for i in range(len(x)):
#         ret.append((x[i] - mu)**2)
#     return my_mean(ret)
#
#
# def my_covariance(x, y):
#     #COV(X, Y) = E(X)E(Y) - E(XY)
#     x = np.array(x)
#     y = np.array(y)
#     return my_mean(x*y) - my_mean(x)*my_mean(y)
#
# def correlation(x, y):
#     if(len(x) != len(y)):
#         return None
#
#     std_x = np.sqrt(my_var(x))
#     std_y = np.sqrt(my_var(y))
#     if(std_x == 0 or std_y == 0):
#         return None
#     return my_covariance(x, y) / (std_x * std_y)
#
#
# # lambda_ = 4
# # alpha_ = 3
# # size = 40
# # observed = lambda_ * np.random.weibull(alpha_, size=size)
# # observed_median = np.nanmedian(observed)
# # print("observed_median:", observed_median)
# #
# # replicates = 1000
# #
# # r_medians = []
# # for r in range(replicates):
# #     r_samp = lambda_ * np.random.weibull(alpha_, size=size)
# #     r_samp_median = np.nanmedian(r_samp)
# #     print(r_samp_median)
# #     r_medians.append(r_samp_median)
# #
# # z_alpha_over_2 = 1.96
# # se_weibull_median = my_var(r_medians) / size
# # lower = observed_median - z_alpha_over_2*se_weibull_median
# # upper = observed_median + z_alpha_over_2*se_weibull_median
# # plt.hist(r_medians, bins = 40)
# # plt.show()
# #
# # print(f"95% CI for weibull median: ({lower}, {upper})")
# # print("observed_median:", observed_median, "in CI:", observed_median >= lower and observed_median <= upper)
# # print("Theoretical median:", lambda_ * (np.log(2)**(1/alpha_)))
# #
# # x = [1, 2, 3, 4, 5]
# # y = [2, 5, 2, 5, 1]
# # print(correlation(x = x, y = y))
# #
# # print(np.corrcoef(x, y)[0, 1])
#
# # df = pd.read_parquet("/Volumes/GoogleDrive/Shared drives/DSCI-DATA/SAVER/SAVER_1479/v1a_1/skin_temp.parquet")
# # plt.plot(df.ts, df.temperature)
# # start1 = pd.read_csv("/Volumes/GoogleDrive/Shared drives/DSCI-DATA/SAVER/SAVER_1479/stopwatch/saver_c1_1479_s1.csv").iloc[0].start_timestamp
# # start2 = pd.read_csv("/Volumes/GoogleDrive/Shared drives/DSCI-DATA/SAVER/SAVER_1479/stopwatch/saver_c1_1479_s3.csv").iloc[0].start_timestamp
# # start3 = pd.read_csv("/Volumes/GoogleDrive/Shared drives/DSCI-DATA/SAVER/SAVER_1479/stopwatch/saver_c1_group_s2.csv").iloc[0].start_timestamp
# # plt.axvline(start1, color = "red")
# # plt.axvline(start2, color = "green")
# # plt.axvline(start3, color = "blue")
# # plt.show()
# #
# #
# # df = pd.read_parquet("/Volumes/GoogleDrive/Shared drives/DSCI-DATA/SAVER/SAVER_8334/polar/polar_6C2DF226_hr.parquet")
# # plt.plot(pd.to_datetime(df.ts, unit = "s"), df.hr)
# # plt.show()
# # # df = p
# # # d.read
# # df = pd.read_parquet("/Volumes/GoogleDrive/Shared drives/DSCI-DATA/SAVER/BLE_PQS/1300/Decr iPhone|Polar H10 25EDF025|20220713_124231/h10_25EDF025_ibi.parquet")
# # print(df.head())
# # df = pd.read_parquet("/Volumes/GoogleDrive/Shared drives/DSCI-DATA/SAVER/SAVER_8334/v1a_1/skin_temp.parquet")
# # plt.plot(pd.to_datetime(np.array(df.ts * 1e-9), unit = "s"), df.temperature)
# # plt.show()
# def get_ibi(df):
#     max_length = 10
#     a_list=[]
#     for ibi in df.rr.values:
#         if (isinstance(ibi, str) and ibi != ""):
#             #print(ibi)
#             this_row = ibi.split(',')
#             this_row= [float(item) for item in this_row]
#             while len(this_row) < max_length:
#                 this_row.append(None)
#             a_list.append(this_row)
#         else:
#             a_list.append(np.repeat(np.nan, max_length))
#     RR = np.array(a_list)
#     RR = RR.astype(np.float)
#     RR = np.delete(RR,np.where(np.isnan(RR).all(axis=0)),axis=1)
#     RR= RR/1000
#     etime = np.array(df.ts)
#     n,m = RR.shape
#     RRStruct=[]
#     tStruct=[]
#     for i in range(n-1,0,-1):
#         rr_row = 0
#         for j in range(m):
#             if not np.isnan(RR[i,j]):
#                 RRStruct.append(RR[i,j])
#                 rr_row = rr_row + RR[i,j]
#                 tStruct.append(etime[i] - rr_row)
#                 slack = tStruct[-1] - etime[i-1]
#                 if slack<0:
#                     tStruct[-1] = etime[i-1]
#     rr_row = 0
#
#     # print("-----------")
#     # print("TSTRUCT:", len(tStruct))
#     # print("-----------")
#     etime_1 = tStruct[-1]
#     for j in range(m):
#         if not np.isnan(RR[0,j]):
#             RRStruct.append(RR[0, j])
#             rr_row = rr_row + RR[0,j]
#             tStruct.append(etime_1 - rr_row)
#
#     H10_IBI = pd.DataFrame({'etime':tStruct, 'IBI':RRStruct})
#     H10_IBI=H10_IBI.sort_values(by=['etime'])
#
#     return H10_IBI
#
# df_in = pd.read_parquet("/Users/lselig/Desktop/ecg_compare_fast/1234/polar_6C2D6A23_hr.parquet")
# df_in = df_in[(df_in.ts >= 1657401300) & (df_in.ts <= 1657408500)]
# df = get_ibi(df_in)
# # df = df[df.]
# # plt.hist2d()
# import seaborn as sns
# import heartpy as hp
# sns.set_style("darkgrid")
# max_hr = np.nanmax(df_in.hr.values)
# print(max_hr)
# ecg_raw = pd.read_parquet("/Users/lselig/Desktop/ecg_compare_fast/1234/polar_6C2D6A23_ecg.parquet")
# ecg_raw = ecg_raw[(ecg_raw.ts >= 1657401300) & (ecg_raw.ts <= 1657408500)]
# plt.plot(pd.to_datetime(ecg_raw.ts, unit = "s"), ecg_raw.ecgSample)
# plt.show()
# working_data, measures = hp.process_segmentwise(np.array(ecg_raw.ecgSample), sample_rate=130,
#                                                 segment_width=4, mode="full", segment_overlap = 0.95,
#                                                 replace_outliers=False, calc_freq=True, bpmmax=max_hr)
#
#
# my_ts = np.zeros(len(measures["segment_indices"]))
# for i_, pair in enumerate(measures["segment_indices"]):
#     start = ecg_raw.iloc[pair[1]].ts
#     my_ts[i_] = start
#
# measures["ts"] = my_ts
# from scipy import interpolate
# hpy_ibi = pd.DataFrame({"ts": my_ts, "ibi": measures["ibi"]})
# f = interpolate.interp1d(my_ts, measures["ibi"], fill_value = [np.nanmedian(measures["ibi"])], bounds_error = False)
# ibi_resampled = f(df.etime.values)
# ibi_resampled[np.where(np.abs(np.diff(ibi_resampled)) > 100)[0] + 1] = np.nan
# # plt.plot(np.diff(ibi_resampled))
#
# ecg_features = pd.read_parquet("/Users/lselig/Desktop/ecg_compare_fast/1234_polar/ecg_features.parquet")
# plt.plot(pd.to_datetime(df.etime, unit = "s"), ibi_resampled / 1000, label = "H10 IBI from H10 ECG", color = "Red")
# plt.plot(pd.to_datetime(df.etime, unit = "s"), df.IBI, label = "H10 IBI from H10 RR intervals", color = "Black")
# plt.plot(pd.to_datetime(my_ts, unit = "s"), np.array(measures["ibi"])/1000, label = "H10 IBI raw heartpy", color = "Green")
#
# plt.ylabel("IBI (S)")
# plt.legend()
# plt.show()
#
# non_nan_idx = np.where(~np.isnan(ibi_resampled))
# my_ibi = ibi_resampled[non_nan_idx] / 1000
# h10_ibi = df.IBI.values[non_nan_idx]
# # plt.gca().set_aspect('equal')
# # plt.scatter(my_ibi /1000, h10_ibi)
# mini = min(min(my_ibi), min(h10_ibi))
# maxi = max(max(my_ibi), max(h10_ibi))
# mini = 0.4
# maxi = 1.2
# plt.hist2d(my_ibi, h10_ibi, bins=[np.linspace(mini, maxi, 100), np.linspace(mini, maxi, 100)])
# print(mini, maxi)
# plt.xlim(mini, maxi)
# plt.ylim(mini, maxi)
# plt.xlabel("IBI (S) from H10 ECG")
# plt.ylabel("IBI (S) from H10 RR")
# plt.title(f"r: {np.round(np.corrcoef(my_ibi, h10_ibi)[0, 1], 3)}")
# plt.tight_layout()
# plt.show()
#
# # Assumptions of errors for linear regression
#
# # Independent (no correlation for true v pred)
# # Constant variance
# # Normally distributed
# df = pd.read_parquet("/Users/lselig/Desktop/move_it_stress_val/MOVE_IT_8334/v0_8334/cps_default/feature/features_1654772901.parquet")
# # print(df.head())
# # df = df.sample(frac=1).reset_index(drop=True)
# # print(df.head())
# y_true = np.array(df.y_true)
# y_pred = np.array(df.y_pred)
#
# y_true = y_true[np.where(~np.isnan(y_true))]
# y_pred = y_pred[np.where(~np.isnan(y_pred))]
#
# residuals = y_true - y_pred
# # np.random.shuffle(residuals)
# print("Durbin Watson test stat for auto correlation in residuals:", durbin_watson(residuals))
# # plt.plot(residuals)
# # plt.show()
# r = np.corrcoef(y_pred, residuals)[0, 1]
# plt.scatter(y_pred, residuals)
# plt.title(np.round(r, 2))
# plt.xlabel("True")
# plt.ylabel("Residuals")
# plt.show()
#
# # r = np.corrcoef(y_true, y_pred)[0, 1]
# plt.hist(residuals, bins = 40)
# # plt.title(np.round(r, 2))
# plt.xlabel("Residuals")
# plt.ylabel("Count")
# plt.show()
#
#
# # r = np.corrcoef(y_true, y_pred)[0, 1]
# plt.hist2d(y_pred, residuals, bins = 40)
# # plt.title(np.round(r, 2))
# plt.ylabel("Residuals")
# plt.xlabel("Predicted")
# plt.show()
#
#
#
# def p_seven_games(wr, monte_carlo_n):
#     outcomes = []
#     # monte_carlo_n = 100000
#     for i in range(monte_carlo_n):
#         t1_wins = 0
#         t2_wins = 0
#         series_game = 0
#         while(True):
#             if(t1_wins == 4 or t2_wins == 4 or series_game == 7):
#                 break
#             t1_win = np.random.uniform(0, 1)
#             if(t1_win <= wr):
#                 t1_wins += 1
#             else:
#                 t2_wins += 1
#             series_game += 1
#
#         outcomes.append(series_game)
#
#     game_7s = np.where(np.array(outcomes) == 7)[0].size
#     p_goes_to_game_7 = game_7s / np.array(outcomes).size
#     print(p_goes_to_game_7)
#     plt.hist(outcomes)
#     plt.show()
#
# # p_seven_games(wr = 0.5, monte_carlo_n = 100000)
#
# def uniform_ev(monte_carlo_n):
#     # Which has a higher EV
#     # a: Random uniform(1, 100) ** 2
#     # b: Random uniform(1, 100) * Random uniform (1, 100)
#     # as_ = []
#     # bs_ = []
#     as_ = []
#     bs_ = []
#
#
#     tic = timeit.default_timer()
#     for i in range(monte_carlo_n):
#         as_.extend([np.random.uniform(1, 100) ** 2])
#         bs_.extend([np.random.uniform(1, 100) * np.random.uniform(1, 100)])
#         # as_[i] = a
#         # bs_.append(b)
#
#     toc = timeit.default_timer()
#     print(f"RUNTIME: {toc - tic}")
#     print(f"EV(a) = {np.nanmean(as_)}")
#     print(f"EV(b) = {np.nanmean(bs_)}")
#
# # print(uniform_ev(1000000))
#
# from scipy.stats import skewnorm
#
# def plot_mean_med_skewnorm(skew, size):
#     # skew = 10
#     data = skewnorm.rvs(skew, size = 10000)
#     plt.hist(data)
#     plt.axvline(np.nanmean(data), label = "Mean", color = "C0")
#     plt.axvline(np.nanmedian(data), label = "Median", color = "C1")
#     plt.legend()
#     plt.show()
#
# # plot_mean_med_skewnorm(10, 10000)
#
# def manifest(monte_carlo_n):
#     success = 0
#     import string
#     for i in range(monte_carlo_n):
#         population = list(string.ascii_lowercase)[:14]
#         np.random.shuffle(population)
#         # population = np.random.shuffle(range(14))
#         options = []
#         for j in range(3):
#             draw = np.random.randint(0, len(population))
#             options.append(population.pop(draw))
#
#         if('a' in options or 'b' in options):
#             success += 1
#
#     print(success, monte_carlo_n, success / monte_carlo_n)
#
# print(manifest(1000000))
# # bashCommand = "sqlite3 my_
# # corrupted.sqlite '.recover' | sqlite3 new.sql"
# # process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, cwd = "/Users/lselig/Desktop")
# # output, error = process.communicate()
# #
# #
# # con = sqlite3.connect('/Volumes/GoogleDrive/Shared drives/DSCI-DATA/MOVE_IT/MOVE_IT_1234_3.1/Movesense/Lucas’s iPhone|Movesense 200830000205,Movesense 210730000511,Movesense 200830000309|20220629_192530.sqlite')
# # with open('/Users/lselig/Desktop/Lucas’s iPhone|Movesense 200830000205,Movesense 210730000511,Movesense 200830000309|20220629_192530_dump.sql', 'w') as f:
# #     for line in con.iterdump():
# #         print(line)
# #         # try:
# #         f.write(f'{line}\n')
# #         # except Exception:
# #         #     continue
# #
# x = [1, 2, 3, 4]
# x = np.array(x)
# print(x)
# x = x.reshape(1, -1)
# print(x)
# print(x.size, x.shape, len(x))
# df = pd.read_parquet("/Volumes/GoogleDrive/Shared drives/DSCI-DATA/MOVE_IT/MOVE_IT_1864_3.1/v0_1864/eda.parquet")
# plt.plot(pd.to_datetime(df.ts * 1e-9, unit="s"), 1e6 / df.resistance)
# plt.show()
# connection = sqlite3.connect("/Volumes/GoogleDrive/Shared drives/DSCI-DATA/MOVE_IT/MOVE_IT_470499_3.1/Movesense/Atar’s iPhone|Movesense 210730000540,Movesense 210730000550|20220630_085710_recovered.sqlite")
# cursor = connection.cursor()
# sql_file = open('/Users/lselig/Desktop/Atar’s iPhone|Movesense 210730000540,Movesense 210730000550|20220630_085710.sqlite.sql')
# sql_as_string = sql_file.read()
# cursor.executescript(sql_as_string)
# # os.remove('Lucas’s iPhone|Movesense 200830000205,Movesense 210730000511,Movesense 200830000309|20220629_192530_dump.sql')
#
# a = 1
# def add_nans(data, gap_size):
#     big_gap_ind = np.hstack((np.where(np.diff(data[:, 0]) > gap_size)[0], len(data)))
#     ts = data[:, 0]
#
#     vals = np.array(data[:, 1])
#     new_ts, new_vals = [], []
#     new_ts.append(ts[: big_gap_ind[0]])
#     new_vals.append(vals[: big_gap_ind[0]])
#     for i in range(len(big_gap_ind) - 1):
#         new_ts.append(
#             ts[big_gap_ind[i]] + (ts[big_gap_ind[i] + 1] - ts[big_gap_ind[i]]) / 2
#         )
#         new_vals.append([np.nan])
#         new_ts.append(ts[big_gap_ind[i] + 1 : big_gap_ind[i + 1]])
#         new_vals.append(vals[big_gap_ind[i] + 1 : big_gap_ind[i + 1]])
#
#     new_ts = np.hstack(new_ts)
#     new_vals = np.hstack(new_vals)
#     return new_ts, new_vals
#
#
#
#
# def get_apple_watch_hr(filename, outname):
#     # file_name = 'export.xml'
#     pattern = '^.*IdentifierHeartRate".*startDate="(.{19}).*value="([0-9]*).*$'
#
#     with open(outname, 'w') as f:
#         writer = csv.writer(f)
#         writer.writerow(['dt', 'bpm'])
#
#         with open(filename, 'r') as f2:
#             for line in f2:
#                 search = re.search(pattern, line)
#                 if search is not None:
#                     writer.writerow([search.group(1), search.group(2)])
# filename = "/Users/lselig/Downloads/apple_health_export/export.xml"
# outname = '/Users/lselig/Downloads/apple_health_export/hr.csv'
#
# v0 = pd.read_parquet("/Volumes/GoogleDrive/Shared drives/DSCI-DATA/MOVE_IT/MOVE_IT_470499_3.1/v0_470499/eda.parquet")
# v1a = pd.read_parquet("/Volumes/GoogleDrive/Shared drives/DSCI-DATA/MOVE_IT/MOVE_IT_470499_3.1/v1a_470499/eda.parquet")
#
# plt.plot(v0.ts * 1e-9, 1e6 / v0.resistance, label = "V0")
# plt.plot(v1a.ts * 1e-9, 1e6 / v1a.resistance, label = "V1")
# plt.legend()
# plt.show()
#
# hr_csv = pd.read_csv(outname)[-2000:]
#
# plt.plot(pd.to_datetime([x.timestamp() for x in hr_csv.dt], unit = "s"), hr_csv.bpm)
# plt.xlabel("Time")
# plt.ylabel("HR (BPM)")
# plt.title("Lucas apple watch HR (BPM)")
# plt.show()
#
#
# df = pd.read_parquet("/Volumes/GoogleDrive/Shared drives/Partners - Studies Data sets/Happy_App_Sleep_Window_Nans/0B09C3CF-C70D-46C4-BE89-E0BD26481621/ea148700010f/02252022/acc.parquet")
# print(df.head())
#
# eda_df = pd.read_parquet("/Volumes/GoogleDrive/Shared drives/Partners - Studies Data sets/Happy_App_Sleep_Window_Nans/0B09C3CF-C70D-46C4-BE89-E0BD26481621/ea148700010f/02252022/eda.parquet")
# print(eda_df.head())
# plt.plot(df.etime, df.x)
# plt.show()
# import traceback
#
# try:
#     1/0
# except Exception:
#     traceback.print_exc()
#
# sns.set_style("darkgrid")
# df = pd.read_parquet("/Users/lselig/Desktop/move_it_stress_val/MOVE_IT_8334/v0_8334/cps_ridge_fw_flags/scores/stress_scores_1654858576.parquet")
# df = df[["ts", "sqi_gated_ss", "acr", "sqi", "any_fw_flag_set"]]
# df["stress_score"] = df["sqi_gated_ss"]
# # print(list(df))
# # df = df.drop('sqi_gated_ss', 1)
# # df.to_csv("/Users/lselig/Desktop/demo.csv", index = False)
# # print(df.shape)
# df_stress = add_nans(np.array(df[["ts", "sqi_gated_ss"]]), 8)
# df_stress = pd.DataFrame({"ts": df_stress[0], "sqi_gated_ss": df_stress[1]})
#
# fig, axs = plt.subplots(3, 1, figsize = (12, 4), sharex = True)
# # axs[0].plot(df.ts, df.stress_score)
# axs[0].plot(pd.to_datetime(df_stress.ts, unit = "s"), df_stress.sqi_gated_ss)
# axs[0].set_ylim(0, 100)
# axs[1].plot(pd.to_datetime(df.ts, unit = "s"), df.any_fw_flag_set)
# axs[2].plot(pd.to_datetime(df.ts, unit = "s"), df.acr)
# plt.show()
# print(list(df))
# with open(
#     "/Volumes/GoogleDrive/Shared drives/DSCI-DATA/MOVE_IT/converted_movesense_list.pickle",
#     "rb",
# ) as f:
#     x = pickle.load(f)
#
# df = pd.read_parquet(
#     "/Volumes/GoogleDrive/Shared drives/DSCI-ANALYSIS/MOVE_IT/MOVE_IT_1300/v0_eda_df.parquet"
# )
# sw = pd.read_csv(
#     "/Volumes/GoogleDrive/Shared drives/DSCI-DATA/MOVE_IT/stopwatch/clean/cohort_1/move_it_c1_s1_stopwatch_6.9.22_1.csv"
# )
# plt.plot(df.etime, df.conductance)
# plt.axvline(sw.iloc[0].start_timestamp * 1e-9, color="red", lw=3)
# plt.show()
#
# default = pd.read_parquet(
#     "/Volumes/GoogleDrive/Shared drives/DSCI-ANALYSIS/HAPPY_PADS_PIPELINE/Lucas_3/Default/long/eda.parquet"
# )
# plated = pd.read_parquet(
#     "/Volumes/GoogleDrive/Shared drives/DSCI-ANALYSIS/HAPPY_PADS_PIPELINE/Lucas_3/Plated/long/eda.parquet"
# )
# v1a = pd.read_parquet(
#     "/Volumes/GoogleDrive/Shared drives/DSCI-ANALYSIS/HAPPY_PADS_PIPELINE/Lucas_3/V1a/long/eda.parquet"
# )
#
# mini = np.nanmin(
#     [np.nanmin(default.etime), np.nanmin(plated.etime), np.nanmin(v1a.etime)]
# )
# maxi = np.nanmax(
#     [np.nanmax(default.etime), np.nanmax(plated.etime), np.nanmax(v1a.etime)]
# )
#
# default = default[(default.etime >= mini) & (default.etime <= maxi)]
# plated = plated[(plated.etime >= mini) & (plated.etime <= maxi)]
# v1a = v1a[(v1a.etime >= mini) & (v1a.etime <= maxi)]
#
# plt.title("D300")
# plt.plot(pd.to_datetime(default.etime, unit="s"), default.conductance, label="Default")
# plt.plot(pd.to_datetime(plated.etime, unit="s"), plated.conductance, label="Plated")
# plt.plot(pd.to_datetime(v1a.etime, unit="s"), v1a.conductance, label="V1a")
# plt.legend()
# plt.ylabel("EDA µS")
# plt.show()
#
#
# files = glob.glob(
#     "/Users/lselig/research-tools/research_tools/nators/research/data/hpy2/cb1315000210/F20*.HPY2"
# )
# print(files)
# data = ""
#
# for file in files:
#     with open(file, "rb") as fp:
#         new = fp.read()
#         # print(new)
#         if len(data) == 0:
#             data = new
#         else:
#             data += new
#
# with open("/Users/lselig/Desktop/lucas_plated.HPY2", "wb") as fp:
#     fp.write(data)
# #
# # prs_lmw = pd.read_parquet("/Users/lselig/Desktop/field_preproc/D300144A-64FC-44EB-BCFC-1AC11630F066/eb14960000d9/cps_final_no_gaps_svr_peak_debug/feature/features_0.parquet")
# # runner_lmw = pd.read_parquet("/Users/lselig/Desktop/field_preproc/D300144A-64FC-44EB-BCFC-1AC11630F066/eb14960000d9/cps_final_no_gaps_svr_peak_debug/scores/stress_scores_0.parquet")
# #
# #
# #
# # prs_nolmw = pd.read_parquet("/Users/lselig/Desktop/field_preproc/D300144A-64FC-44EB-BCFC-1AC11630F066/eb14960000d9/cps_final_no_gaps_svr_peak_debug_no_lmw/feature/features_0.parquet")
# # runner_nolmw = pd.read_parquet("/Users/lselig/Desktop/field_preproc/D300144A-64FC-44EB-BCFC-1AC11630F066/eb14960000d9/cps_final_no_gaps_svr_peak_debug_no_lmw/scores/stress_scores_0.parquet")
# # plt.plot(prs_lmw.ts, prs_lmw.peak_count, label = "LMW ON")
# # plt.plot(prs_nolmw.ts, prs_nolmw.peak_count, label = "LMW OFF")
# # plt.legend()
# # plt.show()
# #
# # fig, axs = plt.subplots(2, 2, figsize = (15, 9))
# # axs = axs.flatten()
# #
# # axs[0].boxplot(prs_lmw.peak_count.values[~np.isnan(prs_lmw.peak_count.values)])
# # axs[0].set_title("prs, lmw = on")
# #
# # axs[1].boxplot(runner_lmw.peaks_unscaled.values[~np.isnan(runner_lmw.peaks_unscaled.values)])
# # axs[1].set_title("runner, lmw = on")
# #
# # axs[2].boxplot(prs_nolmw.peak_count.values[~np.isnan(prs_nolmw.peak_count.values)])
# # axs[2].set_title("prs, lmw = off")
# #
# # axs[3].boxplot(runner_nolmw.peaks_unscaled.values[~np.isnan(runner_nolmw.peaks_unscaled.values)])
# # axs[3].set_title("runner, lmw = off")
# # fig.suptitle(f"D300144A-64FC-44EB-BCFC-1AC11630F066/eb14960000d9\nn_windows = {prs_lmw.shape[0]}")
# # plt.tight_layout()
# # plt.show()
# #
# #
# # # axs[0].boxplot(prs_lmw.peak_count)
# # # axs[0].set_title("prs, lmw = on")
# #
# #
# # def scale_vec(x):
# #     return (x - np.nanmean(x)) / np.nanstd(x)
# #
# # df = pd.read_parquet("/Volumes/GoogleDrive/Shared drives/DSCI-DATA/MOVE_IT/MOVE_IT_1393/v1a_1393/skin_temp.parquet")
# # df = df.sort_values(by = "ts")
# # temp = df.temperature.to_numpy()[:-1]
# # ts = df.ts.to_numpy() * 1e-9
# #
# # diff_ts = np.diff(ts)
# # bad_idx = np.argwhere(diff_ts <= 1)
# # temp[bad_idx] =  np.nan
# # # plt.plot(diff_ts)
# # # plt.show()
# # plt.plot(pd.to_datetime(ts[:-1], unit="s"), temp)
# #
# # # plt.plot(df.temperature)
# # plt.xlabel("time")
# # plt.ylabel("skin temp (C)")
# # plt.title("MOVE_IT_1393/v1a_1393/")
# # plt.show()
# # # plt.plot(df.device_name)
# # # plt.show()
# # # df1 = df[df["device_name"] == "HH_04be"]
# # # df2 = df[df["device_name"] == "HH_0033"]
# # # df1 = df
# # # df2 = df
# # # fig, axs = plt.subplots(2, 1)
# # # axs[0].plot(df1.temperature)
# # # axs[1].plot(df2.temperature)
# # plt.show()
# # # plt.plot(df.ts, )
# # # df = pd.read_parquet(
# # #     "/Volumes/GoogleDrive/Shared drives/DSCI-DATA/MOVE_IT/MOVE_IT_8334/prod_2A6B6A66-72B9-485A-8ACD-8A06AD8586D0/ppg.parquet"
# # # )
# # print(df.head())
# # ts = df.ts.to_numpy() * 1e-9
# #
# # plt.plot(pd.to_datetime(ts, unit="s"), df.temperature.to_numpy())
# # plt.xlabel("Time")
# # plt.ylabel("PPG Green")
# # plt.title("prod_090A3D5F-C318-423A-BE62-988B8834994F")
# # plt.show()
# # feat = pd.read_parquet(
# #     "/Users/lselig/Desktop/slope_signal_feature_engineering/2_2_2_2_2_2_2_2/INSIDE_OUT_20210614_539786_607_1/features.parquet"
# # )
# # targets = pd.read_parquet(
# #     "/Users/lselig/Desktop/IO_features/INSIDE_OUT_20210614_539786_607_1/features.parquet"
# # )
# #
# #
# # df = pd.read_parquet(
# #     "/Users/lselig/Desktop/IO_features/INSIDE_OUT_20210609_1005_512_1/features.parquet"
# # )
# # plt.plot(df.ts, df.y_true)
# # plt.show()
# # df = pd.read_parquet(
# #     "/Users/lselig/Desktop/INSIDE_OUT_PEAKS/INSIDE_OUT_20210616_653218_741_1/acps/regression_targets.parquet"
# # )
# # df = pd.read_parquet(
# #     "/Users/lselig/Desktop/field_preproc/D300144A-64FC-44EB-BCFC-1AC11630F066/eb14960000d9/cps_final_no_gaps_svr_v4/feature/features_0.parquet"
# # )
# # plt.plot(df.ts, df.stress_score)
# # plt.show()
# # seed = np.random.seed(1)
# # df = pd.read_parquet(
# #     "/Users/lselig/Desktop/field_preproc/2A6B6A66-72B9-485A-8ACD-8A06AD8586D0/d8137100012c/cps_final_no_gaps_svr/feature/features_0.parquet"
# # )
# # x = df.y_pred
# # y = df.y_true
# # y = y[~np.isnan(df.y_true.values)]
# # x = x[~np.isnan(df.y_true.values)]
# #
# #
# # def scale_vec(x, min_, max_):
# #     # scale vector to (0, 100)
# #     return (x - np.nanmean(x)) / np.nanstd(x)
# #
# #
# # plt.hist2d(x, y, bins=50)
# # plt.show()
# #
# # # x = np.random.poisson(lam = 20, size = 50)
# # # y = np.random.poisson(lam = 10, size = 50)
# # # y = x ** 3
# # # y = x + z
# # corr = np.corrcoef(x, y)
# # r2score = r2_score(y, x)
# # mse = mean_squared_error(x, y)
# # print(r2score)
# # print(corr)
# #
# # x = scale_vec(x, min_=np.min(x), max_=np.max(x))
# # y = scale_vec(y, min_=np.min(y), max_=np.max(y))
# # corr = np.corrcoef(x, y)
# # r2score = r2_score(y, x)
# # mse = mean_squared_error(x, y)
# # print(r2score)
# # print(corr)
# #
# #
# # # print(mse)
# # # plt.plot(x)
# # # plt.plot(y, alpha = 0.7)
# # # plt.show()
# #
# # # x =  scale_vec(x, min_ = np.min(x), max_ = np.max(x))
# # # y =  scale_vec(y, min_ = np.min(y), max_ = np.max(y))
# # reg = LinearRegression().fit(x.values.reshape(-1, 1), y)
# # corr = np.corrcoef(y, x)
# # print("linreg score", reg.score(x.values.reshape(-1, 1), y), (corr ** 2)[0, 1])
# # r2score = r2_score(y, x)
# # custom_r2_score = 1 - np.sum((x - y) ** 2) / np.sum((y - np.nanmean(y)) ** 2)
# # print(r2score, custom_r2_score)
# # print(corr)
# # # print(mse)
# # # plt.plot(x, "k-")
# # # plt.plot(y, "r-")
# # # plt.show()
