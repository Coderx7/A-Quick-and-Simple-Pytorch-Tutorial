#! /home/hossein/anaconda3/bin/python3
#%%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import re
import glob

for log_file in glob.glob('*.txt'):
    with open(log_file, 'r') as file: 
        lines = file.readlines()
        # 2022-12-07 05:30:05,228 -     train: [    INFO] - Test (EMA): [ 195/195]  Time-Batch: 0.010 (0.101)  Loss:  2.3223 (1.2892)  Acc@1: 36.2500 (70.9100)  Acc@5: 83.7500 (89.6360)
        regegx = r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2},\d{3}).*\(EMA\):.*\[ 195/195\].*Loss:.*\((\d{1,2}.\d{4})\).*\((\d{2}.\d{4})\).*\((\d{2}.\d{4})\).*"
        results = list(re.findall(regegx, line,flags=re.IGNORECASE) for line in lines)
        results = list(tuple(*result) for result in results if result)
        # print(*results,sep='\n')
        df = pd.DataFrame(results)
        df = df.astype({2:'float',3:'float',4:'float'})
        # print(df)
        df.plot()
        plt.show()
# %%
