from joblib import load
import numpy as np
import pandas as pd

model = load('model.pkl')
proccesor = load('preprocessor.pkl')

df = pd.read_csv('Book1.csv')

enc = proccesor.transform(df)
pred = model.predict(enc)

print(pred)