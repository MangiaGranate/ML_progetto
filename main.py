'''
Introduzione:

Dai dati presenti nella zip del dataset ho scoperto che solo una delle 3 classi è linearmente divisibile, devo 
utilizzare principalmente metodi non lineari.
Inizialmente testerò il decision tree e il nearest neighbors come unici classificatori delle tre classi.
In seguito proverei altri modelli singoli per ogni classe (classificazione binaria) e li interpreterei
con un modello 1-vs-all

'''
LABLES = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

from pathlib import Path
dataset_path = Path(r"iris.csv")

import pandas as pd
import numpy as np

# sklearn lib
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
#from sklearn.base import clone

# sklearn lib - models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# sklearn lib - metriche classificazione
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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

# Etichette per i classificatori lineari
y_bin = {}
for lable in LABLES:
    y_bin[lable] = [0] * len(y)
    for i, ele in enumerate(y):
        if ele==lable:
            y_bin[lable][i] = 1
        else:
            y_bin[lable][i] = 0

y_setosa = y_bin['Iris-setosa']
y_versicolor = y_bin['Iris-versicolor']
Y_virginia = y_bin['Iris-virginica']


'''
FONTE: scikit-learn.org
Come prima cosa dividiamo i sample del dataset in training e testing; alla funzione poniamo:
test_size=0.1 perchè essendo il dataset molto piccolo vogliamo allenare i nostri campioni con più sample possibili
random_size=int in modo che ad ogni iterazione la funzione produca sempre lo stesso split
stratify=y in modo che i due sottoinsiemi siano omogenei rispetto ad y
'''
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.30, random_state=7, stratify=y)

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


string = '''
#############################
INIZIO CROSS VALIDATION
#############################
'''
print(string)

############################# Cross validation - Ottimizzazione iperparametri  #########################

    # Nearest neighbors - multiclass

iperparametri = {
    'modello__n_neighbors': [1,2,3,4,5,6,7,8,9,10]
    }
''' NB
Dal momento che viene usata una Pipeline gli iperparametri devono essere riferiti ad un preciso step
di essa, per questo devo mettere modello__ prima di n_neighbors per indiare a pipeline che l'iperparametro
è riferito al modello e nonn allo scaler
'''


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('modello',  KNeighborsClassifier())
])

gs = GridSearchCV(
    estimator=pipeline,
    param_grid=iperparametri,
    scoring='f1_macro',
    cv=5,
    verbose=True)

gs.fit(X_train, y_train)
nome = 'K-Nearest Neighbor'
print(nome + '-'*(40-len(nome)))
print('iperparametri migliori: ', gs.best_params_)
print('accuratezza del modello:', gs.best_score_)
print('-'*20 + '\n')   # separatore




    # Decision tree - multiclass

iperparametri = {
    
    }


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('modello',  KNeighborsClassifier())
])

gs = GridSearchCV(
    estimator=pipeline,
    param_grid=iperparametri,
    scoring='f1_macro',
    cv=5,
    verbose=True)


gs.fit(X_train, y_train)
nome = 'Decision Tree'
print(nome + '-'*(40-len(nome)))
print('iperparametri migliori: ', gs.best_params_)
print('accuratezza del modello:', gs.best_score_)
print('-'*20 + '\n')   # separatore


    # LogisticRegression - classificatori binari

iperparametri = {
    'modello__C': [0.001, 0.01, 0.1, 1, 10, 100],          # Regolarizzazione
    'modello__penalty': ['l2'],                            # Tipo di regolarizzazione
    'modello__solver': ['liblinear', 'saga'],              # Algoritmo di ottimizzazione
    'modello__max_iter': [1000, 2000],                     # Iterazioni massime
    'modello__tol': [1e-4, 1e-3],                          # Tolleranza per la convergenza
    'modello__class_weight': [None, 'balanced']            # Gestione classi sbilanciate
}


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('modello',  LogisticRegression())
])

gs = GridSearchCV(
    estimator=pipeline,
    param_grid=iperparametri,
    scoring='f1_macro',
    cv=5,
    verbose=True)


gs.fit(X_train, y_train)
nome = 'LogisticRegression'
print(nome + '-'*(40-len(nome)))
print('iperparametri migliori: ', gs.best_params_)
print('accuratezza del modello:', gs.best_score_)
print('-'*20 + '\n')   # separatore




    # SVC - classificatori binari

iperparametri = {
    'modello__C': [0.1, 1, 10, 100],                    # Regolarizzazione
    'modello__kernel': ['rbf', 'linear', 'poly'],       # Tipo di kernel
    'modello__gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Coefficiente per kernel rbf/poly
    'modello__degree': [2, 3, 4],                       # Grado per kernel polinomiale
    'modello__class_weight': [None, 'balanced'],        # Gestione classi sbilanciate
    'modello__probability': [True],                     # Abilita predict_proba()
    'modello__decision_function_shape': ['ovr']         # One-vs-Rest per multiclasse
}

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('modello',  SVC())
])

gs = GridSearchCV(
    estimator=pipeline,
    param_grid=iperparametri,
    scoring='f1_macro',
    cv=5,
    verbose=True)


gs.fit(X_train, y_train)
nome = 'SVC'
print(nome + '-'*(40-len(nome)))
print('iperparametri migliori: ', gs.best_params_)
print('accuratezza del modello:', gs.best_score_)
print('-'*20 + '\n')   # separatore

############################# Considerazioni - Model selection  #########################

'''
conclusioni...
'''

parametri = {
    'n_neighbors' : 5,
    #'criterion': 'entropy'
}

best_model=KNeighborsClassifier(n_neighbors=5)
best_model=DecisionTreeClassifier(criterion='entropy')

def test(best_model):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scal = scaler.transform(X_train)
    X_test_scal = scaler.transform(X_test)

    best_model.fit(X_train_scal, y_train)
    y_pred = best_model.predict(X_test_scal)

############################# Testing  #########################

    accuracy_score, precision_score, recall_score, f1_score
    accuratezza = accuracy_score(y_test, y_pred)
    print(f"l'accurazetta del modello {best_model} è di: {accuratezza}")

test(best_model)

