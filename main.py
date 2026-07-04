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
from sklearn.base import clone

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

Prima del Cross validation deve essere definito un modello con i suoi iperparametri:
Prima si sceglie il metodo di ottimizzazione degli iperparametri e poi si usa "_valutatore_kfold"  per
testarne le metriche e scegliere il migliore.

'''


############################# Cross validation - Ottimizzazione iperparametri  #########################
modello = KNeighborsClassifier(n_neighbors= 1)

def _valutatore_kfold(modello: object, X: np.ndarray, y: np.ndarray, verbose: bool = False, splits: int = 7, scalamento: bool = False):
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

    X : np.ndarray
        features del dataset di training da suddividere

    y : np.ndarray
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
    medie : dict 
        dizionario contenente le medie delle valutazioni finali del modello

    note
    -----
    Questa funzione sarebbe la mia implementazione personale di cross_val_score di sklearn.
    Ho deciso di implementarla in modo che ogni passaggio fosse ben visibile

    '''
    # random_state=7 per rendere il risultato riproducibile
    splits = max(2, splits) # uno split minore di 2 non ha senso
    skf = StratifiedKFold(n_splits=splits, random_state=7, shuffle=True)

    valutazioni = []    # list[dict] valutazioni di ogni fold

    for index, (train_index, test_index) in enumerate(skf.split(X, y)):
        
        # non posso lavorare sulla copia originale del modello passato
        modello_fold = clone(modello)
        '''DEBUG
        if verbose:
            print(f"Fold {index}:")
            print(f"  Train: index={train_index}")
            print(f"  Test:  index={test_index}")
        '''
            
        # Dati per ogni fold:
        X_train = X[train_index]
        X_valid = X[test_index]

        y_train = y[train_index]
        y_valid = y[test_index]

        if scalamento:
            # scalamento dei dati basato sui soli dati di training
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_valid = scaler.transform(X_valid)

        # alleno il modello e interpreto i dati di validation
        modello_fold.fit(X_train, y_train)
        y_pred = modello_fold.predict(X_valid)


        # METRICHE !!!
        valutazioni_fold={
            'Accuracy': accuracy_score(y_valid, y_pred),
            'Precision': precision_score(y_valid, y_pred, average='macro', zero_division=0),
            'Recall': recall_score(y_valid, y_pred, average='macro'),
            'F1-Score': f1_score(y_valid, y_pred, average='macro')
        }
        if verbose: print(f"valutazioni fold {index}\n{valutazioni_fold}")
        valutazioni.append(valutazioni_fold)

    if not valutazioni: return {}   #se vuoto esci

    somme = {}
    
    # Somma tutte le metriche
    for fold in valutazioni:
        for metrica, valore in fold.items():
            if metrica not in somme.keys(): somme[metrica]=0
            somme[metrica] += valore
    
    # Calcola la media per ogni metrica
    n_fold = len(valutazioni)
    medie = {metrica: somma / n_fold for metrica, somma in somme.items()}

    if verbose: print(f"medie valutazioni modello:\n{medie}")
    return medie

_valutatore_kfold(modello, X_train, y_train, verbose = True)



############################# Considerazioni - Model selection  #########################

...

'''
conclusioni...
'''
best_model=...

############################# Testing  #########################

...


