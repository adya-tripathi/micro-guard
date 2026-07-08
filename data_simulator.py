import pandas as pd
import time
 
 
def stream_rows(path, feature_cols, delay=0.05, loop=True):
    
    df = pd.read_csv(path)
    while True:
        for _, row in df.iterrows():
            x = row[feature_cols].values
            yield x, row
            time.sleep(delay)
        if not loop:
            break
 
 
def load_dataset(path):
    return pd.read_csv(path) 