import pandas as ps
import numpy as np
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def process(self, df):
        scaler = StandardScaler()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.mean())
        return scaler.fit_transform(df.drop(columns = "stop_avg_delay_time"))
