import numpy as np
import pandas as pd 

df=pd.read_csv("driver-pulse-hackathon\\backend\\driver_pulse_hackathon_data\\processed_outputs\\flagged_moments.csv")
print(df["flag_type"].value_counts())

