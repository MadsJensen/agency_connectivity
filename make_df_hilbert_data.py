import numpy as np
import pandas as pd
import scipy.io as sio

from my_settings import *

data = sio.loadmat("/home/mje/Projects/agency_connectivity/Data/data_all.mat")[
    "data_all"]

column_keys = ["subject", "trial", "condition", "shift"]
result_df = pd.DataFrame(columns=column_keys)

for k, subject in enumerate(subjects):

    p8_invol_shift = data[k, 3] - np.mean(data[k, 0])
    p8_vol_shift = data[k, 2] - np.mean(data[k, 0])
    p8_vol_bs_shift = data[k, 1] - np.mean(data[k, 0])

    for j in range(89):
        row = pd.DataFrame([{"trial": int(j),
                             "subject": subject,
                             "condition": "vol_bs",
                             "shift": p8_vol_bs_shift[j + 1][0]}])

        result_df = result_df.append(row, ignore_index=True)

    for j in range(89):
        row = pd.DataFrame([{"trial": int(j),
                             "subject": subject,
                             "condition": "vol",
                             "shift": p8_vol_shift[j + 1][0]}])

        result_df = result_df.append(row, ignore_index=True)

    for j in range(89):
        row = pd.DataFrame([{"trial": int(j),
                             "subject": subject,
                             "condition": "invol",
                             "shift": p8_invol_shift[j][0]}])

        result_df = result_df.append(row, ignore_index=True)
