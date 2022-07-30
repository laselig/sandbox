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
import requests, json
np.random.seed( 0 )
sns.set_style( "darkgrid" )

user_id = "A0348940-D1DB-4B3E-AEA0-7B9BD7B025AC"
ring_serial = "ea1487000130"

# user_id = "697B2605-ECE3-4C1C-A294-A3C2DE7CCEB0"
# ring_serial = "eb211600004a"
url = "https://apollo-api-staging.happy.dev/v1/history/cstressprs/v3"
headers = {
        "X-HAPPY-MP-SUB": user_id,
        "Content-Type": "application/json",
        "accept": "application/json",
    }

payload = {
  "start": "2022-01-01T17:33:14.937Z",
  "end": "2022-07-28T17:33:14.937Z",
  "ring_id": ring_serial
}

json_data = json.dumps(payload)
resp = requests.post(url, data=json_data, headers=headers)
print(resp.json())
