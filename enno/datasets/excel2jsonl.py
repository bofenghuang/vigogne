import pandas as pd
import numpy as np
import json
import datasets

filepath = "EnnoData - FAQ.xlsx"
filename = "data"  # os.path.splitext(filepath)[0]

df = pd.read_excel(filepath)

a = [{'instruction': x['Question'],
      'input': '',
      'output': x['Réponse David']
      } for _, x in df.iterrows()]
b = [{'instruction': x['Question'],
      'input': '',
      'output': x['Réponse Alain']
      } for _, x in df.iterrows()]
a.extend(b)
# Filtrage des lignes sans réponses
train_instruct = [x for x in a if x['output'] is not np.nan]

print('Nombre de lignes :', len(train_instruct))

# Données d'évaluation
t = [{'instruction': x['Question'],
      'input': '',
      'output': x['Test']
      } for _, x in df.iterrows()]
eval_instruct = [x for x in t if x['output'] is not np.nan]

# Création du json d'entrainement
with open(filename + '-train.json', 'w') as f:
    json.dump(train_instruct, f)

# Création du json
with open(filename + '-eval.json', 'w') as f:
    json.dump(eval_instruct, f)

# Création des jsonl
with open(filename + '-train.jsonl', 'w') as f:
    for x in train_instruct:
        f.write(json.dumps(x) + '\n')

with open(filename + '-eval.jsonl', 'w') as f:
    for x in eval_instruct:
        f.write(json.dumps(x) + '\n')

d = datasets.Dataset.from_json('./datasets/enno-data/data-train.json')