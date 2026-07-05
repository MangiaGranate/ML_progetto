'''
Introduzione:

Dai dati presenti nella zip del dataset ho scoperto che solo una delle 3 classi è linearmente divisibile, devo 
utilizzare principalmente metodi non lineari

'''
LABLES = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

from pathlib import Path
dataset_path = Path(r"iris.csv")

import pandas as pd
import numpy as np

# sklearn lib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
#from sklearn.base import clone

# sklearn lib - models
from sklearn.neighbors import KNeighborsClassifier

# sklearn lib - metriche classificazione
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score






############################# EDA  #########################

dataset = pd.read_csv(dataset_path, sep=',')

# Vedere se ci sono colonne con valori nulli
print("Controllo valori nulli:")
print(f"valori nulli:\n{dataset.isnull().any()}")

print('-'*20)   # separatore

# Vedere il numero di esempi per classi
conteggio_per_classe = dataset['y'].value_counts().sort_index()
print("Controllo conteggio classi:")
print(conteggio_per_classe)

print('-'*20)   # separatore

# Trasformo in np.array
y = dataset.y.to_numpy()
X = dataset.drop(columns='y').to_numpy()


'''
FONTE: scikit-learn.org
Come prima cosa dividiamo i sample del dataset in training e testing; alla funzione poniamo:
test_size=0.1 perchè essendo il dataset molto piccolo vogliamo allenare i nostri campioni con più sample possibili
random_size=int in modo che ad ogni iterazione la funzione produca sempre lo stesso split
stratify=y in modo che i due sottoinsiemi siano omogenei rispetto ad y
'''
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.1, random_state=7, stratify=y)

#testiamo che la funzione abbia effettivamente diviso i sample in modo omogeneo:

def _test_classi(array: np.array):
    '''
    Accetta un np.array e conta quanti elementi ci sono
    '''
    classi = {}
    for ele in array:
        if ele not in classi:
            classi[ele] = 1
        else:
            classi[ele]+=1
    return classi

print(f"distribuzione delle classi in y_train:\n{_test_classi(y_train)}")
print('-'*20)   # separatore
print(f"distribuzione delle classi in y_test:\n{_test_classi(y_test)}")


'''
Lo scalamento deve essere fatto interno al k-fold altrimenti non è attendibile valutare ogni ottimizzazione con il 
Validator
StratifiedKFold si preoccupa di generare per ogni fold un Validator diverso e omogeneo

Prima del Cross validation deve essere definito un modello con i suoi iperparametri:
Prima si sceglie il metodo di ottimizzazione degli iperparametri e poi si usa "_valutatore_kfold"  per
testarne le metriche e scegliere il migliore.

Diverse funzioni native di sklearn racchiudono tutti i passaggi di cross validation: ottimizzazione parametri, k-fold e scalamento interno
ad ogni fold.
Un implementazione potrebbe essere l'uso di GridSearchCV() passando come estimator una Pipeline.

In sklearn una Pipeline è uno strumento che concatena più trasformatori, questo ci permette, all'interno di
GridSearchCV, di scalare in ogni fold i dati (sia training che validator) basandoci sono sui dati di
training di quel fold specifico. 


'''

print('-'*20)   # separatore

############################# Cross validation - Ottimizzazione iperparametri  #########################

    # Nearest neighbors - multiclass

iperparametri = {
    'n_neighbors': list(range(1, 10, 2))
    }

gs = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=iperparametri,
    scoring='accuracy',
    cv=5)

gs.fit(X_train, y_train)
print('K-Nearest Neighbor ..................')
print('iperparametri migliori: ', gs.best_params_)
print('accuratezza del modello:', gs.best_score_)
print('-'*20)   # separatore


############################# Considerazioni - Model selection  #########################

...

'''
conclusioni...
'''
best_model=...

############################# Testing  #########################

...


