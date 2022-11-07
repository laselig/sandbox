from happyalgos.utils.signal.preprocessor import filter
from happyalgos.utils.buffer.timebased import HPYTimeBasedBuffer, DatagramFileSource
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import signal
import sys
from enum import Enum
import pandas as pd
from happyalgos.activity.activity_counter import ActivityCounter
import seaborn as sns
import glob, os
sns.set_style("darkgrid")
WINDOW, STRIDE = 60.0, 50.0


# data_files = ["718B6E00-3248-4983-ACBD-F62DB14BF542-HH_00a5-20220831_145907.HPY2"]
# folder = Path("/Users/behnam_molavi/Downloads/Preprocessing Datasets/")

import pandas as pd
from collections import defaultdict
import sqlite3
import struct
import numpy as np

FRAMING_BYTE = 0x7E

LAST_BATT = {}


def get_sample_type(input_dict):
    if input_dict == None:
        return "none"
    elif "heartRate" in input_dict:
        return "heartRate"
    elif "ecg" in input_dict:
        return "ecg"
    elif "acc" in input_dict:
        return "acc"


ECG_NORMAL_FRAME_TICKS_DELT = 560697041
ECG_SAMPLES_PER_FRAME = 73
ACC_NORMAL_FRAME_TICKS_DELT = 1409533952
ACC_SAMPLES_PER_FRAME = 36
ACC_INT_TO_GS = 1000.0

# HR_TICKS_VAL = 1024
HR_TICKS_VAL = 1000

# Common idxs
ID_COL = 0
TS_COL = 1
DEV_ID_COL = 2

# device_data idx
CHAR_UUID_COL = 3
CHAR_VAL_COL = 4

# ecg idx
DEV_TS_COL = 3
ECG_CSV_COL = 4

# accel idx
ACC_DEV_TS_COL = 3
ACC_CSV_X_COL = 4
ACC_CSV_Y_COL = 5
ACC_CSV_Z_COL = 6

# heart_rate idx
HR_BPM_COL = 3
HR_RR_COL = 4

NS_PER_S = 1e9

MOVESENSE_CHAR_UUID = "6B200002-FF4E-4979-8186-FB7BA486FCD7"
BATT_CHAR_UUID = "2A19"

debug_first_pass = True


def process_closeout_ecg(dev_id, output_data_list, column_output):
    if len(output_data_list) > 2:
        first_samp = output_data_list[0]
        last_samp = output_data_list[-1]
        first_dev_ts = first_samp["Timestamp"]
        last_dev_ts = last_samp["Timestamp"]
        first_host_ts = first_samp["host_ts"]
        last_host_ts = last_samp["host_ts"]
        # do simple translation
        host_time_diff = last_host_ts - first_host_ts
        ticks_multiplier = host_time_diff / (last_dev_ts - first_dev_ts)
        nominal_slope = ECG_NORMAL_FRAME_TICKS_DELT / ECG_SAMPLES_PER_FRAME
        ticks_slope = ticks_multiplier * nominal_slope
        if host_time_diff < 30 * 60 * NS_PER_S:
            ticks_slope = nominal_slope
        # samples_start_ts = first_host_ts + (ticks_slope * sample_i)

        running_emb_count = 0
        for sample_dict in output_data_list:
            cur_ts = sample_dict["host_ts"]
            dev_ts = sample_dict["Timestamp"]
            cur_ts_start = first_host_ts + (ticks_slope * running_emb_count)

            out_array = [[cur_ts_start + (ticks_slope * i), val] for i, val in enumerate(sample_dict["Samples"])]
            running_emb_count += len(sample_dict["Samples"])
            column_output[dev_id].extend(out_array)
    else:
        pass


def process_closeout_acc(dev_id, output_data_list, column_output):
    if len(output_data_list) > 2:
        first_samp = output_data_list[0]
        last_samp = output_data_list[-1]
        first_dev_ts = first_samp["Timestamp"]
        last_dev_ts = last_samp["Timestamp"]
        first_host_ts = first_samp["host_ts"]
        last_host_ts = last_samp["host_ts"]
        # do simple translation
        host_time_diff = last_host_ts - first_host_ts
        ticks_multiplier = host_time_diff / (last_dev_ts - first_dev_ts)
        nominal_slope = ACC_NORMAL_FRAME_TICKS_DELT / ACC_SAMPLES_PER_FRAME
        ticks_slope = ticks_multiplier * nominal_slope
        if host_time_diff < 30 * 60 * NS_PER_S:
            ticks_slope = nominal_slope
        # samples_start_ts = first_host_ts + (ticks_slope * sample_i)

        running_emb_count = 0
        for sample_dict in output_data_list:
            cur_ts = sample_dict["host_ts"]
            dev_ts = sample_dict["Timestamp"]
            cur_ts_start = first_host_ts + (ticks_slope * running_emb_count)

            zip_items = list(zip(sample_dict["x"], sample_dict["y"], sample_dict["z"]))
            out_array = [[cur_ts_start + (ticks_slope * i), *val] for i, val in enumerate(zip_items)]
            running_emb_count += len(sample_dict["x"])
            column_output[dev_id].extend(out_array)
    else:
        pass


def process_closeout_hr(dev_id, output_data_list, column_output_hr):
    # TODO DECR - I am just filtering out all small segments because they aren't going have good data anyway
    if len(output_data_list) > 200:
        # if len(output_data_list) > 2 :
        first_samp = output_data_list[0]
        last_samp = output_data_list[-1]
        first_dev_ts = first_samp["Timestamp"]
        last_dev_ts = last_samp["Timestamp"]
        first_host_ts = first_samp["host_ts"]
        last_host_ts = last_samp["host_ts"]
        # do simple translation
        host_time_diff = last_host_ts - first_host_ts
        if last_dev_ts - first_dev_ts == 0:
            return
        ticks_multiplier = host_time_diff / (last_dev_ts - first_dev_ts)
        nominal_slope = 1
        ticks_slope = ticks_multiplier * nominal_slope
        # If we are smaller than a large number of minutes worth of data
        # don't try to correct the timing.
        # You will probably add more error.
        if host_time_diff < 30 * 60 * NS_PER_S:
            ticks_slope = nominal_slope
        # samples_start_ts = first_host_ts + (ticks_slope * sample_i)

        for sample_dict in output_data_list:
            cur_ts = sample_dict["host_ts"]
            dev_ts = sample_dict["Timestamp"]
            hr_bpm = sample_dict["hr"]
            cur_ts_start = first_host_ts + (ticks_slope * (dev_ts - first_dev_ts))

            running_rr_count = 0
            out_array = []
            for i, val in enumerate(sample_dict["Samples"]):
                out_array.append([cur_ts_start + (ticks_slope * running_rr_count), val, hr_bpm])
                running_rr_count += (val / HR_TICKS_VAL) * NS_PER_S
            column_output_hr[dev_id].extend(out_array)
    else:
        pass


def interpret(data):
    """
    data is a list of integers corresponding to readings from the BLE HR monitor
    """

    byte0 = data[0]
    res = {}
    res["hrv_uint8"] = (byte0 & 1) == 0
    sensor_contact = (byte0 >> 1) & 3
    if sensor_contact == 2:
        res["sensor_contact"] = "No contact detected"
    elif sensor_contact == 3:
        res["sensor_contact"] = "Contact detected"
    else:
        res["sensor_contact"] = "Sensor contact not supported"
    res["ee_status"] = ((byte0 >> 3) & 1) == 1
    res["rr_interval"] = ((byte0 >> 4) & 1) == 1

    if res["hrv_uint8"]:
        res["hr"] = data[1]
        i = 2
    else:
        res["hr"] = (data[2] << 8) | data[1]
        i = 3

    if res["ee_status"]:
        res["ee"] = (data[i + 1] << 8) | data[i]
        i += 2

    if res["rr_interval"]:
        res["rr"] = []
        while i < len(data):
            # Note: Need to divide the value by 1024 to get in seconds
            res["rr"].append((data[i + 1] << 8) | data[i])
            i += 2

    return res


def get_ibi_direct(df):
    # df = pd.read_parquet(in_path)
    max_length = 10
    a_list = []
    for ibi in df.rr.values:
        if isinstance(ibi, str) and ibi != "":
            this_row = ibi.split(",")
            this_row = [float(item) for item in this_row]
            while len(this_row) < max_length:
                this_row.append(None)
            a_list.append(this_row)
        else:
            a_list.append(np.repeat(np.nan, max_length))
    RR = np.array(a_list)
    RR = RR.astype(np.float)
    RR = np.delete(RR, np.where(np.isnan(RR).all(axis=0)), axis=1)
    RR = RR / 1000
    etime = np.array(df.ts)
    n, m = RR.shape
    RRStruct = []
    tStruct = []
    for i in range(n - 1, 0, -1):
        rr_row = 0
        for j in range(m):
            if not np.isnan(RR[i, j]):
                RRStruct.append(RR[i, j])
                rr_row = rr_row + RR[i, j]
                tStruct.append(etime[i] - rr_row)
                slack = tStruct[-1] - etime[i - 1]
                if slack < 0:
                    tStruct[-1] = etime[i - 1]
    rr_row = 0

    etime_1 = tStruct[-1]
    for j in range(m):
        if not np.isnan(RR[0, j]):
            RRStruct.append(RR[0, j])
            rr_row = rr_row + RR[0, j]
            tStruct.append(etime_1 - rr_row)

    H10_IBI = pd.DataFrame({"ts": tStruct, "ibi": RRStruct})
    H10_IBI = H10_IBI.sort_values(by=["ts"])

    return H10_IBI


def extract(sqlite_path, out_dfs_path, study_id, is_binary):
    # description: reads a sqlite at sqlite_path containing H10 data and parses each binary row. Saves output as a df at out_dfs_path

    # sqlite_path = str to sqlite location
    # out_dfs_path = Path object, base location for saving output df
    # do_quick_plot = bool, whether or not to show a quick plot of hr for sanity check
    # is_binary = bool, whether or not we have to parse binary blobs
    # returns: number of rows failed to convert

    in_f_name = sqlite_path.split("/")[-1].split(".sqlite")[0]
    out_dfs_path = out_dfs_path / study_id / in_f_name
    # out_dfs_path = Path(savedir_str)
    if not out_dfs_path.exists():
        out_dfs_path.mkdir(parents=True, exist_ok=True)

    # Handle HR/RR
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    parsed_rows, parsed_rows_ecg = [], []
    n_failed_rows = 0
    n_success_rows = 0
    BLOB_IDX = 4
    TS_IDX = 1

    # find device_id
    h10_device_ids = []
    h10_device_names = []
    for row in cur.execute("SELECT * FROM device"):
        if row[3].startswith("Polar H10"):
            h10_device_ids.append(row[0])
            h10_device_names.append(row[3])

    if len(h10_device_ids) == 0:
        return
    h10_device_ids = np.hstack(h10_device_ids)
    h10_device_names = np.hstack(h10_device_names)

    if is_binary:

        for i in range(len(h10_device_ids)):
            h10_device_id = h10_device_ids[i]
            h10_device_name = h10_device_names[i]
            h10_device_name_id = h10_device_name.split(" ")[-1]
            for row in cur.execute("SELECT * FROM device_data WHERE device_id = '%s'" % h10_device_id):
                data = row[BLOB_IDX]
                try:
                    res = interpret(data)
                    res["ts"] = row[TS_IDX] * 1e-9
                    parsed_rows.append(res)
                    n_success_rows += 1
                except:
                    n_failed_rows += 1
                    continue
            if len(parsed_rows) > 0:
                out_hr_df = pd.DataFrame(parsed_rows)
                out_ecg_df = pd.DataFrame(parsed_rows_ecg)

                out_hr_df.to_parquet(out_dfs_path / f"h10_{h10_device_name_id}_hr.parquet", index=False)
                out_ecg_df.to_parquet(
                    out_dfs_path / f"h10_{h10_device_name_id}ecg_out.parquet",
                    index=False,
                )

    else:
        for i in range(len(h10_device_ids)):
            h10_device_id = h10_device_ids[i]
            h10_device_name = h10_device_names[i]
            h10_device_name_id = h10_device_name.split(" ")[-1]
            failed = False
            try:
                rows = cur.execute(
                    f"SELECT timestamp, heart_rate_bpm, heart_rate_rr_interval "
                    f"FROM heart_rate "
                    f"WHERE device_id = {h10_device_id}"
                )
                for row in rows:
                    row = {"ts": row[0] * 1e-9, "hr": row[1], "rr": row[2]}

                    parsed_rows.append(row)

                # for row in cur.execute("SELECT timestamp, device_timestamp, ecg FROM ecg"):
                #     row = {"ts": row[0] * 1e-9,
                #            "ecg": row[2]}
                #     parsed_rows_ecg.append(row)

                out_hr_df = pd.DataFrame(parsed_rows)
                # plt.plot(np.diff(out_hr_df.ts))
                # plt.title(sqlite_path)
                # plt.show()

                # out_ecg_df = pd.DataFrame(parsed_rows_ecg)
                ibi_df = get_ibi_direct(out_hr_df)
                out_hr_df = out_hr_df[["ts", "hr"]]
                out_hr_df.to_parquet(out_dfs_path / f"h10_{h10_device_name_id}_hr.parquet", index=False)
                ibi_df.to_parquet(out_dfs_path / f"h10_{h10_device_name_id}_ibi.parquet", index=False)
                a = 1

            except Exception:
                print("Failed extracting H10 HR/RR: ", sqlite_path.split("/")[-1])
                failed = True
                continue

    # Handle ECG/ACC/HR/IBI
    # if not is_binary and not failed:
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()

    dev_dict = defaultdict(list)

    # we create a list of records that become our final output files
    final_output_list = []
    output_dict = {}

    dev_id_dict = {}

    column_output_acc = {}

    column_output_ecg = {}

    column_output_hr = {}

    # Get the BLE device names
    for row in cur.execute("SELECT * FROM device"):
        dev_id = row[0]
        dev_name = row[3]
        dev_id_dict[dev_id] = dev_name
        LAST_BATT[dev_id] = []
        column_output_ecg[dev_id] = []
        column_output_hr[dev_id] = []
        column_output_acc[dev_id] = []

    for dev_id in dev_id_dict.keys():
        # start populating output_dict
        if not dev_id in output_dict:
            output_dict[dev_id] = defaultdict(list)

    for row in cur.execute("SELECT * FROM device_data"):
        # for row in cur.execute('SELECT * FROM "main"."device_data" WHERE "characteristic_uuid" LIKE \'%6B200002-FF4E-4979-8186-FB7BA486FCD7%\' ESCAPE \'\\\' '):

        dev_id = row[DEV_ID_COL]
        cur_ts = row[TS_COL]

        if BATT_CHAR_UUID == row[CHAR_UUID_COL]:
            LAST_BATT[dev_id].append((cur_ts, struct.unpack("B", row[CHAR_VAL_COL])[0]))

    #########################
    # ECG first pass
    last_ts = 0
    last_dev_ts = 0

    for row in cur.execute("SELECT * FROM ecg"):
        # for row in cur.execute('SELECT * FROM "main"."device_data" WHERE "characteristic_uuid" LIKE \'%6B200002-FF4E-4979-8186-FB7BA486FCD7%\' ESCAPE \'\\\' '):

        dev_id = row[DEV_ID_COL]
        cur_ts = row[TS_COL]
        dev_ts = row[DEV_TS_COL]
        ecg_csv = row[ECG_CSV_COL]

        # Create a new file for every set of samples with a large gap in between
        #
        # If the data is more than 60 seconds in the future
        # then start a new file
        if (dev_ts - last_dev_ts) > (2 * ECG_NORMAL_FRAME_TICKS_DELT) or (cur_ts - last_ts) > 60 * NS_PER_S:
            process_closeout_ecg(dev_id, output_dict[dev_id]["ecg"], column_output_ecg)
            final_output_list.append(output_dict)
            output_dict = {}
            for dev_id in dev_id_dict.keys():
                # start populating output_dict
                if not dev_id in output_dict:
                    output_dict[dev_id] = defaultdict(list)
        last_ts = cur_ts
        last_dev_ts = dev_ts

        # start populating output_dict
        if not dev_id in output_dict:
            output_dict[dev_id] = defaultdict(list)

        out = {}
        out["host_ts"] = cur_ts
        out["Timestamp"] = dev_ts
        ecg_ints = [int(val) for val in ecg_csv.split(",")]
        out["Samples"] = list(ecg_ints)
        output_dict[dev_id]["ecg"].append(out)

    # TODO DECR - this is adding too much data for us to process
    # so I am removing it even though it is wrong.
    #
    # We have to take the final dict and add it to the running list
    for dev_id in dev_id_dict.keys():
        if dev_id in output_dict and "ecg" in output_dict[dev_id]:
            process_closeout_ecg(dev_id, output_dict[dev_id]["ecg"], column_output_ecg)

    final_output_list.append(output_dict)
    output_dict = {}
    for dev_id in dev_id_dict.keys():
        # start populating output_dict
        if not dev_id in output_dict:
            output_dict[dev_id] = defaultdict(list)

    #############################
    # ACC first pass
    last_ts = 0
    last_dev_ts = 0

    have_acc_table = True
    try:
        tmp = cur.execute("SELECT * FROM acceleration")
    except:
        have_acc_table = False
    if have_acc_table:
        for row in cur.execute("SELECT * FROM acceleration"):
            # for row in cur.execute('SELECT * FROM "main"."device_data" WHERE "characteristic_uuid" LIKE \'%6B200002-FF4E-4979-8186-FB7BA486FCD7%\' ESCAPE \'\\\' '):

            dev_id = row[DEV_ID_COL]
            cur_ts = row[TS_COL]
            dev_ts = row[ACC_DEV_TS_COL]
            acc_x_csv = row[ACC_CSV_X_COL]
            acc_y_csv = row[ACC_CSV_Y_COL]
            acc_z_csv = row[ACC_CSV_Z_COL]

            # Create a new file for every set of samples with a large gap in between
            #
            # If the data is more than 60 seconds in the future
            # then start a new file
            if (dev_ts - last_dev_ts) > (2 * ACC_NORMAL_FRAME_TICKS_DELT) or (cur_ts - last_ts) > 60 * NS_PER_S:
                process_closeout_acc(dev_id, output_dict[dev_id]["acc"], column_output_acc)
                final_output_list.append(output_dict)
                output_dict = {}
                for dev_id in dev_id_dict.keys():
                    # start populating output_dict
                    if not dev_id in output_dict:
                        output_dict[dev_id] = defaultdict(list)
            last_ts = cur_ts
            last_dev_ts = dev_ts

            # start populating output_dict
            if not dev_id in output_dict:
                output_dict[dev_id] = defaultdict(list)

            out = {}
            out["host_ts"] = cur_ts
            out["Timestamp"] = dev_ts

            acc_x_ints = [int(val) / ACC_INT_TO_GS for val in acc_x_csv.split(",")]
            out["x"] = list(acc_x_ints)

            acc_y_ints = [int(val) / ACC_INT_TO_GS for val in acc_y_csv.split(",")]
            out["y"] = list(acc_y_ints)

            acc_z_ints = [int(val) / ACC_INT_TO_GS for val in acc_z_csv.split(",")]
            out["z"] = list(acc_z_ints)

            output_dict[dev_id]["acc"].append(out)

        # We have to take the final dict and add it to the running list
        for dev_id in dev_id_dict.keys():
            if dev_id in output_dict and "acc" in output_dict[dev_id]:
                process_closeout_acc(dev_id, output_dict[dev_id]["acc"], column_output_acc)

        final_output_list.append(output_dict)
        output_dict = {}
        for dev_id in dev_id_dict.keys():
            # start populating output_dict
            if not dev_id in output_dict:
                output_dict[dev_id] = defaultdict(list)

    else:
        print("No H10 acceleration table, old BLE app version?", sqlite_path)
    #######################
    ## HR first pass
    start_cur_ts = 0
    running_rr_sum = 0
    for row in cur.execute("SELECT * FROM heart_rate"):
        # for row in cur.execute('SELECT * FROM "main"."device_data" WHERE "characteristic_uuid" LIKE \'%6B200002-FF4E-4979-8186-FB7BA486FCD7%\' ESCAPE \'\\\' '):

        dev_id = row[DEV_ID_COL]
        cur_ts = row[TS_COL]
        hr_bpm = row[HR_BPM_COL]
        rr_csv = row[HR_RR_COL]

        # manage the odd rr interval state machine
        if running_rr_sum == 0:
            start_cur_ts = cur_ts
            last_ts = cur_ts
            last_dev_ts = dev_ts
        dev_ts = start_cur_ts + (running_rr_sum / HR_TICKS_VAL * 1e9)

        if (dev_ts - last_dev_ts) > (2 * NS_PER_S) or (cur_ts - last_ts) > 60 * NS_PER_S:
            process_closeout_hr(dev_id, output_dict[dev_id]["hr"], column_output_hr)
            final_output_list.append(output_dict)
            output_dict = {}
            for dev_id in dev_id_dict.keys():
                # start populating output_dict
                if not dev_id in output_dict:
                    output_dict[dev_id] = defaultdict(list)
            running_rr_sum = 0
        last_ts = cur_ts
        last_dev_ts = dev_ts

        # start populating output_dict
        if not dev_id in output_dict:
            output_dict[dev_id] = defaultdict(list)

        out = {}
        out["host_ts"] = cur_ts
        out["Timestamp"] = dev_ts
        out["hr"] = hr_bpm
        rr_ints = []
        if len(rr_csv) > 0:
            rr_ints = [int(val[:-2]) for val in rr_csv.split(",")]
        for val in rr_ints:
            running_rr_sum += val
        out["Samples"] = list(rr_ints)
        output_dict[dev_id]["hr"].append(out)

    # We have to take the final dict and add it to the running list
    for dev_id in dev_id_dict.keys():
        if dev_id in output_dict and "hr" in output_dict[dev_id]:
            process_closeout_hr(dev_id, output_dict[dev_id]["hr"], column_output_hr)

    final_output_list.append(output_dict)
    output_dict = {}
    for dev_id in dev_id_dict.keys():
        # start populating output_dict
        if not dev_id in output_dict:
            output_dict[dev_id] = defaultdict(list)

    ##################################################

    # HR - second pass
    for dev_id, data_list in column_output_hr.items():
        if len(data_list) > 2:
            np_data_list = np.array(data_list)
            f_name = str(dev_id_dict[dev_id]) + "__hr__" + str(np_data_list[0][0] * 1e-9)
            np_data_list_t = np_data_list.T
            hr_time = np_data_list_t[0]
            hr_rr = np_data_list_t[1]
            hr_bpm = np_data_list_t[1]

            ibi_df = pd.DataFrame({"ts": hr_time * 1e-9, "ibi": hr_rr / 1000})
            dev_name_id = str(dev_id_dict[dev_id]).split(" ")[-1]

            bpm_df = pd.DataFrame({"ts": hr_time * 1e-9, "hr": hr_bpm})
            dev_name_id = str(dev_id_dict[dev_id]).split(" ")[-1]

    #######################
    ## ACC - second pass
    if have_acc_table:
        for dev_id, data_list in column_output_acc.items():
            if len(data_list) > 2:
                np_data_list = np.array(data_list)
                np_data_list_t = np_data_list.T
                acc_time = np_data_list_t[0]
                acc_x_signal = np_data_list_t[1]
                acc_y_signal = np_data_list_t[2]
                acc_z_signal = np_data_list_t[3]
                if acc_x_signal.shape[0] > 2:
                    pass
                    # plt.figure()
                    # plt.plot(acc_time*1e-9, acc_x_signal)
                    # plt.plot(acc_time*1e-9, acc_y_signal)
                    # plt.plot(acc_time*1e-9, acc_z_signal)
                    # plt.show()
                acc_df = pd.DataFrame(
                    {
                        "ts": acc_time * 1e-9,
                        "x": acc_x_signal,
                        "y": acc_y_signal,
                        "z": acc_z_signal,
                    }
                )
                dev_name_id = str(dev_id_dict[dev_id]).split(" ")[-1]
                acc_df.to_parquet(out_dfs_path / f"h10_{dev_name_id}_acc.parquet", index=False)

    #######################
    ## ECG - second pass
    for dev_id, data_list in column_output_ecg.items():
        if len(data_list) > 2:
            np_data_list = np.array(data_list)
            np_data_list_t = np_data_list.T
            ecg_time = np_data_list_t[0]
            ecg_signal = np_data_list_t[1]
            ecg_signal = np.nan_to_num(ecg_signal)
            if ecg_signal.shape[0] > 2:
                pass
                # plt.figure()
                # plt.plot(ecg_time*1e-9, ecg_signal)
            ecg_df = pd.DataFrame({"ts": ecg_time * 1e-9, "ecgSample": ecg_signal})
            dev_name_id = str(dev_id_dict[dev_id]).split(" ")[-1]
            ecg_df.to_parquet(out_dfs_path / f"h10_{dev_name_id}_ecg.parquet", index=False)

    return 0


# example how to use
# extract(sqlite_path = "/Users/lselig/Downloads/Dustin’s iPhone (2)_Movesense 210730000570,Polar H10 2F0FBA25_20220722_134155.sqlite",
#         out_dfs_path = Path("/Users/lselig/Desktop"),
#         study_id = "1",
#         is_binary = False
#         )

# ecg_df = pd.read_parquet("/Users/lselig/Desktop/1/Dustin’s iPhone (2)_Movesense 210730000570,Polar H10 2F0FBA25_20220722_134155/h10_2F0FBA25_ecg.parquet")
# hr_df = pd.read_parquet("/Users/lselig/Desktop/1/Dustin’s iPhone (2)_Movesense 210730000570,Polar H10 2F0FBA25_20220722_134155/h10_2F0FBA25_hr.parquet")
# acc_df = pd.read_parquet("/Users/lselig/Desktop/1/Dustin’s iPhone (2)_Movesense 210730000570,Polar H10 2F0FBA25_20220722_134155/h10_2F0FBA25_acc.parquet")
#
#
# fig, axs = plt.subplots(4, 1, figsize = (15, 9), sharex = True)
# axs[0].plot(ecg_df.ts * 1e-9, ecg_df.ecgSample)
# axs[1].plot(hr_df.ts, hr_df.hr)
# axs[2].plot(acc_df.ts * 1e-9, acc_df.x)
# axs[2].plot(acc_df.ts * 1e-9, acc_df.y)
# axs[2].plot(acc_df.ts * 1e-9, acc_df.z)
# axs[3].plot(acc_df.ts * 1e-9, np.sqrt(acc_df.x ** 2 + acc_df.y**2 + acc_df.z**2))
#
#
#
# plt.show()


h10_data = extract("/Users/lselig/Desktop/walking/1717/20221024_124229-20221031_122847.sqlite",
                   Path("/Users/lselig/Desktop/walking/1717"),
                   "1717",
                   is_binary = False)
get_data = True
if(get_data):
    for trial in glob.glob("/Volumes/GoogleDrive/Shared drives/DSCI-ANALYSIS/HR_ACTIVITY/*/"):
        pred_tss = []
        pred_classes = []
        pred_counts = []
        features = {}
        trial_name = trial.split("/")[-2]
        if("HR_ACTIVITY_20210623_492477_784_1" not in trial_name):
            continue
        eda_df = pd.read_csv(glob.glob(trial + "P6_*_EDA.csv")[0])
        # eda_df["conductance"] = eda_df["conductance"] * 1e6
        # eda_df["eda_fs_cur_na"] = [1] * eda_df.shape[0]
        # eda_df["flags"] = [1] * eda_df.shape[0]
        # eda_df["drive_voltage"] = [1] * eda_df.shape[0]

        # acc_df = pd.read_parquet("/Volumes/GoogleDrive/Shared drives/Partners - Studies Data sets/light_tmp/light_tmp_1234/P9/eb153200004e/eb153200004e_acc.parquet")
        # eda_df = pd.read_parquet("/Volumes/GoogleDrive/Shared drives/Partners - Studies Data sets/light_tmp/light_tmp_1234/P9/eb153200004e/eb153200004e_eda.parquet")
        # stopwatch = pd.read_csv("/Users/lselig/Desktop/lucas_activity.csv")
        stopwatch = pd.read_csv("/Users/lselig/Desktop/walking/600104/stopwatch.csv")
        stopwatch["etime"] = stopwatch.start_timestamp * 1e-9
        stopwatch["label"] = stopwatch["notes"]
        stopwatch = stopwatch.sort_values(by = "start_timestamp")
        print(stopwatch)
        # eda_df = eda_df[eda_df.etime >= 1666987128202944000 * 1e-9 - 600]
        # acc_df = acc_df[acc_df.etime >= 1666987128202944000 * 1e-9 - 600]

        # acc_df = pd.read_csv(glob.glob(trial + "P6_*_ACC.csv")[0])
        # print(acc_df.shape[0] / (acc_df.iloc[-1].etime - acc_df.iloc[0].etime))

        # stopwatch = pd.read_csv(glob.glob(trial + "StopWatch.csv")[0])

        mid_activity = {"Bike low": ["Bike low", "Stationary bike"],
                        "Elliptical low": ["Elliptical low", "Elliptical", "Low eliptical", "Ellip"],
                        "Yoga low": ["Low yoga", "Yoga low", "Yoga/stretching (low)", "Yoga / stretching (low)"],
                        "Yoga mod": ["Mod yoga", "Yoga mod", "Yoga/stretching (moderate)", "Yoga / stretching (moderate)"],
                        "Tread low": ["Low treadmill", "Low tread", "Tread low", "Baseline treadmill (low)", "Treadmill (low)", "Treadmill low"],
                        "Tread mod": ["Med treadmill", "Mod tread", "Tread mod", "Baseline treadmill (moderate)", "Treadmill (moderate)", "Treadmill moderate"],
                        "Outside low": ["Outside low", "Slow walk"],
                        "Outside mod": ["Outside mod", "Moderate walk"],
                        "Recovery": ["Recovery", "Treadmill (recovery)", "Treadmill ( recovery)"],
                        "Walk natural": ["Walk natural"],
                        "Walk texting": ["Walk texting"],
                        "Walk pockets": ["Walk pockets"]
                       }

        high_activity = {"Yoga high": ["High yoga", "Yoga high", "Yoga/stretching (high)", "Yoga / stretching (high)"],
                         "Tread high": ["High tread", "Tread high", "Baseline treadmill (high)", "Treadmill (high)", "Treadmill high"],
                         "Outside high": ["Outside high", "High walk/jog", "Jogging"]}

        fakes = {"Laundry": ["Laundry", "Standing laundry", "Standing (laundry)", "Folding"],
                 "Typing": ["Typing", "Sitting (typing)"],
                 "Rest": ["Rest"]}


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


        mid_activity = list(mid_activity.keys())
        high_activity = list(high_activity.keys())
        fakes = list(fakes.keys())

        # acc_x = signal.decimate(acc_df.to_numpy()[:, 1], q = 4)
        # acc_y = signal.decimate(acc_df.to_numpy()[:, 2], q = 4)
        # acc_z = signal.decimate(acc_df.to_numpy()[:, 3], q = 4)
        # start = acc_df.iloc[0].etime
        # end = acc_df.iloc[-1].etime
        #
        # new_ts = np.linspace(start, end, num = len(acc_x))
        # acc_new = pd.DataFrame({"etime": new_ts,
        #                         "x": acc_x,
        #                         "y": acc_y,
        #                         "z": acc_z})

        # acc_new = acc_df


        # print(acc_new.shape[0] / (acc_new.iloc[-1].etime - acc_new.iloc[0].etime))
        buf = HPYTimeBasedBuffer(window=WINDOW, stride=STRIDE, min_window=STRIDE - 3, min_samples=1, debug = True)
        # ds = DatagramFileSource.from_dfs(eda_df = eda_df, acc_df = acc_new)
        # ds = DatagramFileSource("/Users/lselig/research-tools/research_tools/.cache/HPY2_FILES/eb153200004e/D6ED73FA-4CE6-4A5D-B9D4-7D0B78287AE6-HH_004e-20221028_145820.HPY2",
        #                         stream = False)
        ds = DatagramFileSource("/Users/lselig/Desktop/walking/600104/B05924C9-8B90-43E7-B74C-841F08491D3F-HH_0075-20221031_133025.HPY2",
                                stream = False)
        acc_df = pd.DataFrame( {"etime": ds.accel_t,
                                 "x"    : ds.accel_x,
                                 "y"    : ds.accel_y,
                                 "z"    : ds.accel_z} )

        plt.plot(pd.to_datetime(ds.accel_t, unit = "s"), np.sqrt(ds.accel_x**2 + ds.accel_y**2 + ds.accel_z**2))


        for p in range(stopwatch.shape[0]):
            plt.axvline(pd.to_datetime(stopwatch.iloc[p].start_timestamp * 1e-9, unit = "s"), color = "red")
        plt.show()
        ring_counter = ActivityCounter( window = WINDOW, stride = STRIDE )
        counter = 0
        for win in buf( ds ):
            # print(counter, win)
            counter += 1
            if win is None:
                print("None")
                continue
            pred_class, act_features = ring_counter.get_activity(
                win["accel_x"], win["accel_y"], win["accel_z"], win["accel_t"]
            )
            ts_anchor = np.median(win["accel_t"])
            print(pd.to_datetime(ts_anchor, unit = "s"))
            pred_tss.append(ts_anchor)
            if(pred_class.name == "low"):
                pred_classes.append("active")
            elif(pred_class.name == "high"):
                pred_classes.append("active")
            else:
                pred_classes.append("rest")

            # pred_counts.append(act_count)
            for f in act_features:
                if(f not in features):
                    features[f] = [act_features[f]]
                else:
                    features[f] = features[f] + [act_features[f]]

        a = 1
        features["etime"] = pred_tss
        features["pred"] = pred_classes
        pred_acc_df = pd.DataFrame(features)
        acc_new = acc_df[acc_df.etime.between(pred_acc_df.iloc[0].etime, pred_acc_df.iloc[-1].etime)]
        eda_df = eda_df[eda_df.etime.between(pred_acc_df.iloc[0].etime, pred_acc_df.iloc[-1].etime)]
        # acc_new = acc_new[acc_new.etime.between(pred_acc_df.iloc[0].etime, pred_acc_df.iloc[-1].etime)]
        stopwatch = stopwatch[stopwatch.etime.between(pred_acc_df.iloc[0].etime, pred_acc_df.iloc[-1].etime)]

        labels = []
        print(stopwatch)
        if(stopwatch.shape[0] == 0):
            continue
        for ts in pred_acc_df.etime.values:
            print(ts)
            insert_idx = np.searchsorted(stopwatch.etime.values, ts)
            if(insert_idx == 0):
                curr_label = stopwatch.iloc[0].label
            else:
                curr_label = stopwatch.iloc[insert_idx - 1].label
            if(curr_label in mid_activity):
                # labels.append('low')
                labels.append('active')
            elif(curr_label in high_activity):
                # labels.append('high')
                labels.append('active')
            else:
                labels.append('rest')

        diff = pred_acc_df.shape[0] - len(labels)
        labels = ["rest"] * diff + labels

        fig, axs = plt.subplots(9, 1, figsize = (15, 9), sharex = True)
        # axs[0].plot(pd.to_datetime(acc_df.etime, unit = 's'), acc_df.x, label = "x")
        # axs[0].plot(pd.to_datetime(acc_df.etime, unit = "s"), acc_df.y, label = "y")
        # axs[0].plot(pd.to_datetime(acc_df.etime, unit = "s"), acc_df.z, label = "z")
        # axs[0].set_ylabel("xyz")
        # axs[0].legend()
        # axs[1].plot(pd.to_datetime(acc_df.etime, unit = "s"), np.sqrt(acc_df.x ** 2 + acc_df.y ** 2 + acc_df.z ** 2), label = "100hz")
        axs[0].plot(pd.to_datetime(acc_new.etime, unit = "s"), np.sqrt(acc_new.x ** 2 + acc_new.y ** 2 + acc_new.z ** 2), label = "26hz", marker = ".")
        axs[0].set_ylabel("Mag (gS)")

        for j in range( stopwatch.shape[0] - 1 ):

            label = stopwatch.iloc[j].label
            if(label in mid_activity):
                color = "orange"
            elif(label in high_activity):
                color = 'red'
            elif(label in fakes):
                color = "green"
            else:
                continue
            for m in [1]:
                axs[m].axvspan(
                    pd.to_datetime( stopwatch.iloc[j].etime, unit = "s" ),
                    pd.to_datetime( stopwatch.iloc[j + 1].etime, unit = "s" ),
                    color = color,
                    alpha = 0.4
                    # label = stopwatch_df.iloc[i].label,
                )

            mid_x = (stopwatch.iloc[j].etime + stopwatch.iloc[j + 1].etime) / 2
            # if j == 0:
            lower, upper = axs[1].get_ylim()
            mid_y = (lower + upper) / 2

            axs[0].text(
                pd.to_datetime( mid_x, unit = "s" ),
                mid_y,
                s = label.replace(" ", "\n"),
                bbox = {
                    "facecolor": "white",
                    "alpha"    : 0,
                    "edgecolor": "black",
                    "pad"      : 1,
                },
                ha = "center",
                va = "center",
            )

        axs[1].plot(pd.to_datetime(pred_acc_df.etime, unit = "s"), pred_acc_df.pred, color = "red", alpha = 0.8, label = "Pred", marker = ".", drawstyle = "steps-pre")
        axs[1].set_ylabel("Class")
        axs[1].plot(pd.to_datetime(pred_acc_df.etime, unit = "s"), labels, color = "black", alpha = 0.8, label = "True", marker = ".", drawstyle = "steps-pre")
        axs[1].legend()
        axs[1].invert_yaxis()
        axs[2].plot(pd.to_datetime(pred_acc_df.etime, unit = "s"), pred_acc_df.counts, marker = ".")
        axs[2].set_ylabel("Counts")
        axs[3].plot(pd.to_datetime(pred_acc_df.etime, unit = "s"), pred_acc_df.crossings1, marker = ".")
        axs[3].set_ylabel("Crossings1")
        axs[4].plot(pd.to_datetime(pred_acc_df.etime, unit = "s"), pred_acc_df.max_pxx, marker = ".")
        axs[4].set_ylabel("Max Power")
        axs[5].plot(pd.to_datetime(pred_acc_df.etime, unit = "s"), pred_acc_df[["std"]], marker = ".")
        axs[5].set_ylabel("Std")
        axs[6].plot(pd.to_datetime(pred_acc_df.etime, unit = "s"), pred_acc_df.peaks1, marker = ".")
        axs[6].set_ylabel("Peaks1")
        axs[7].plot(pd.to_datetime(pred_acc_df.etime, unit = "s"), pred_acc_df.peaks2, marker = ".")
        axs[7].set_ylabel("Peaks2")
        axs[8].plot(pd.to_datetime(pred_acc_df.etime, unit = "s"), pred_acc_df.peaks3, marker = ".")
        axs[8].set_ylabel("Peaks3")
        # axs[9].plot(pd.to_datetime(pred_acc_df.etime, unit = "s"), pred_acc_df.energy, marker = ".")
        # axs[9].set_ylabel("Energy")
        axs[2].axhline(16, color = "C3", alpha = 0.7, ls = "--", label = "Low thresh")
        axs[2].axhline(28, color = "C4", alpha = 0.7, ls = "--", label = "High thresh")
        axs[2].legend()
        # hr_df = pd.read_parquet('/Users/lselig/Desktop/walking/1717/1717/20221024_124229-20221031_122847/h10_9F8E632B_hr.parquet')
        # ibi_df = pd.read_parquet('/Users/lselig/Desktop/walking/1717/1717/20221024_124229-20221031_122847/h10_9F8E632B_ibi.parquet')
        # ecg_df = pd.read_parquet('/Users/lselig/Desktop/walking/1717/1717/20221024_124229-20221031_122847/h10_9F8E632B_ecg.parquet')

        # axs[4].plot(pd.to_datetime(hr_df.ts, unit = "s"), hr_df.hr)
        # axs[4].set_ylabel("HR (BPM)")
        # axs[5].plot(pd.to_datetime(ecg_df.ts, unit = "s"), ecg_df.ecgSample)
        # axs[5].set_ylabel("ECG")
        axs[0].legend()
        # fig.suptitle(trial_name)
        fig.suptitle("600104 Walking")
        plt.tight_layout()

        print((acc_new.iloc[-1].etime - pred_acc_df.iloc[-1].etime) / 60)

        if not os.path.isdir(f"/Users/lselig/Desktop/activity_validation_hr_activity_2_class/{trial_name}"):
            os.makedirs(f"/Users/lselig/Desktop/activity_validation_hr_activity_2_class/{trial_name}")

        acc_df.to_parquet(f"/Users/lselig/Desktop/activity_validation_hr_activity_2_class/{trial_name}/acc_100.parquet", index = False)
        acc_new.to_parquet(f"/Users/lselig/Desktop/activity_validation_hr_activity_2_class/{trial_name}/acc_26.parquet", index = False)
        eda_df.to_parquet(f"/Users/lselig/Desktop/activity_validation_hr_activity_2_class/{trial_name}/eda.parquet", index = False)
        stopwatch.to_parquet(f"/Users/lselig/Desktop/activity_validation_hr_activity_2_class/{trial_name}/stopwatch.parquet", index = False)
        pred_acc_df.to_parquet(f"/Users/lselig/Desktop/activity_validation_hr_activity_2_class/{trial_name}/activity_algo_output.parquet", index = False)

        true_labels = pd.DataFrame({"true": labels,
                                    "etime": pred_acc_df.etime.values})
        true_labels.to_parquet(f"/Users/lselig/Desktop/activity_validation_hr_activity_2_class/{trial_name}/true_labels.parquet", index = False)
        # plt.savefig(f"/Users/lselig/Desktop/activity_validation_hr_activity_2_class/{trial_name}/{trial_name}_view.png", dpi = 300)
        plt.show()


# ANALYSIS
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, confusion_matrix

full_true = []
full_pred = []
n = 0
for trial in glob.glob("/Users/lselig/Desktop/activity_validation_hr_activity/*/"):
    if("202105" in trial or "694957" in trial or "510162" in trial or "1085_656" in trial or "1200_782" in trial):
        continue
    n += 1
    true = list(pd.read_parquet(trial + "/true_labels.parquet").true.values)
    pred = list(pd.read_parquet(trial + "/activity_algo_output.parquet").pred.values)
    for i, ele in enumerate(true):
        if(ele == "rest"):
            true[i] = 0
        if(ele == "low"):
            true[i] = 1
        if(ele == "high"):
            true[i] = 2

    for i, ele in enumerate(pred):
        if(ele == "rest"):
            pred[i] = 0
        if(ele == "low"):
            pred[i] = 1
        if(ele == "high"):
            pred[i] = 2


    full_true += list(true)
    full_pred += list(pred)
    print(n)

target_names = ["rest", "low", "high"]
# target_names = ["rest", "active"]
print(classification_report(full_true, full_pred, target_names=target_names))
x = classification_report(full_true, full_pred, target_names=target_names)
conf_mat = confusion_matrix(full_true, full_pred, normalize=None)
conf_mat_norm = confusion_matrix(full_true, full_pred)
# conf_mat = conf_mat.T
from matplotlib import cm
from sklearn.metrics import accuracy_score
# plot confusion matrix
acc = np.round(accuracy_score(full_true, full_pred), 3)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat,
                               display_labels=["rest", "low", "high"])
disp.plot()
plt.title(f"Overall Accuracy: {acc}")
plt.gca().invert_yaxis()
plt.show()
# plt.savefig(f'/Users/lselig/Desktop/2class_hr_activity.png', dpi = 200)
