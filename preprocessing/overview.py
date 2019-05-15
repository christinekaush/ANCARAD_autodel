from pathlib import Path
import pandas as pd

base_path = Path('.\\code\\')

summary = pd.DataFrame(columns=['dice', 
'precision', 
'recall', 
'true_positives', 
'true_negatives'])

for i in base_path.glob('res*'):
    name = str(i)[8:-4]

    with open(str(i)) as f:
        f.readline()
        for line in f.readlines():
            line = list(line.strip().split(','))
            summary.loc[name, line[0]] = line[1]
            #summary.loc[name, 'std'] = line[1]  
    f.close()

print(summary)
summary.to_csv('summary.csv', sep=';', encoding='utf-8')