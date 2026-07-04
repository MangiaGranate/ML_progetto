'''
Introduzione:

Dai dati presenti nella zip del dataset ho scoperto che solo una delle 3 classi è linearmente divisibile, devo 
utilizzare principalmente metodi non lineari

'''
from pathlib import Path
dataset_path = Path(r"iris.csv")

import pandas as pd
import numpy as np

# sklearn lib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


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

print(f"distribbuzione delle classi in y_train:\n{_test_classi(y_train)}")
print('-'*20)   # separatore
print(f"distribbuzione delle classi in y_test:\n{_test_classi(y_test)}")


'''
Lo scalamento deve essere fatto interno al k-fold altrimenti non è attendibile valutare ogni ottimizzazione con il 
Validator
StratifiedKFold si preoccupa di generare per ogni fold un Validator diverso e omogeneo
'''
############################# Cross validation - Ottimizzazione iperparametri  #########################
model = KNeighborsClassifier(n_neighbors= 5)

def _valutatore_kfold(modello: object, X: np.array, y: np.array, verbose: bool = False, splits: int = 15, scalamento: bool = False):
    '''
    Questa funzione accetta un modello (sklearn) con i suoi iperparametri e un training set.
    Sfrutta la funzione StratifiedKFold per valutare le sue prestazioni su k fold omogeneamente suddivisi.
    I risultati finali sono calcolati come media dei risultati parziali di ogni fold.

    Args
    -----
    modello : object
        modello sklearn da testare gia inizializzato con gli iperparametri
        es:
            modello = KNeighborsClassifier(n_neighbors= 5, ...)

    X : np.array
        features del dataset di training da suddividere

    y : np.array
        lable del dataset di training da suddividere    
    
    verbose : bool
        abilita print di debug

    splits : int
        in quanti fold si desidera suddividere il training set

    scalamento : bool
        abilita lo scalamento di ogni validator usando i dati di training in ogni fold
        disabilitato di default

    returns
    -----
    d : dict 
        dizionario contenente le valutazioni finali del modello

    note
    -----
    Questa funzione sarebbe la mia implementazione personale di cross_val_score di sklearn.
    Ho deciso di implementarla in modo che ogni passaggio fosse ben visibile

    '''
    # random_state=7 per rendere il risultato riproducibile
    skf = StratifiedKFold(n_splits=splits, random_state=7)

    for index, (train_index, test_index) in enumerate(skf.split(X, y)):
        if verbose:
            print(f"Fold {index}:")
            print(f"  Train: index={train_index}")
            print(f"  Test:  index={test_index}")

        # Dati per ogni fold:
        X_train = X[train_index]
        X_validate = X[test_index]

        y_train = y[train_index]
        y_validate = y[test_index]

        if scalamento:
            # scalamento dei dati basato sui soli dati di training
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_validate = scaler.transform(X_validate)

        modello.fit(X_train, y_train)
        y_pred = modello.predict(X_validate)


        # alleno il modello
        modello.fit(...)







############################# Considerazioni - Model selection  #########################

...

'''
conclusioni...
'''
best_model=...

############################# Testing  #########################

...


