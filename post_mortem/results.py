import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

df = pd.read_csv('submit_results.csv',
                 delimiter=';',
                 dtype={'model': str,
                        'CV': float,
                        'AV': float,
                        'public': float,
                        'private': float,
                        'date': str},
                 parse_dates=['date']).reset_index()
print(df.info())

sns.lineplot(data=df[['CV', 'public', 'private', 'AV']], linewidth=2)
plt.gca().invert_xaxis()
plt.show()

