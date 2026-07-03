'''
Introduzione:

Dai dati presenti nella zip del dataset ho scoperto che solo una delle 3 classi è linearmente divisibile, devo 
utilizzare principalmente metodi non lineari

'''
from pathlib import Path

dataset_path = Path(r"iris.csv")

############################# EDA  #########################

import pandas as pd
import numpy as np

#sklearn lib
from sklearn.model_selection import train_test_split


dataset = pd.read_csv(dataset_path, sep=',')

#Vedere se ci sono colonne con valori nulli
print(f"valori nulli:\n{dataset.isnull().any()}")

#Vedere il numero di esempi per classi
conteggio_per_classe = dataset['y'].value_counts().sort_index()
print(conteggio_per_classe)

y = dataset.y.values
X = dataset.drop(columns='y')


'''
FONTE: scikit-learn.org
Come prima cosa dividiamo i sample del dataset in training e testing; alla funzione poniamo:
test_size=0.1 perchè essendo il dataset molto piccolo vogliamo allenare i nostri campioni con più sample possibili
random_size=int in modo che ad ogni iterazione la funzione produca sempre lo stesso split
stratify=y in modo che i due sottoinsiemi siano omogenei rispetto ad y
'''
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.1, random_state=7, stratify=y)

#testiamo che la funzione abbia effettivamente diviso i sample in modo omogeneo:
conteggio_per_classe = y_test.value_counts().sort_index()
print(f"y_test:\n{conteggio_per_classe}")
conteggio_per_classe = y_train.value_counts().sort_index()
print(f"y_train:\n{conteggio_per_classe}")

'''
Lo scalamento deve essere fatto interno al k-fold altrimenti non è attendibile valutare ogni ottimizzazione con il 
Validator
'''
############################# Cross validation - Ottimizzazione iperparametri  #########################

...

############################# Considerazioni - Model selection  #########################

...

'''
conclusioni...
'''
best_model=...

############################# Testing  #########################

...


