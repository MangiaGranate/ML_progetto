'''
Introduzione:

Dai dati presenti nella zip del dataset ho scoperto che solo una delle 3 classi è linearmente divisibile, devo 
utilizzare principalmente metodi non lineari.
Inizialmente testerò il decision tree e il nearest neighbors come unici classificatori delle tre classi.
In seguito proverei altri modelli singoli per ogni classe (classificazione binaria) e li interpreterei
con un modello 1-vs-all

'''
LABLES = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
TRAT = 120      # quanti trattini mettere come separatore ad ogni sezione


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

string = '''
#############################
EDA
#############################
'''
print(string)


dataset = pd.read_csv(dataset_path, sep=',')

# Vedere se ci sono colonne con valori nulli
print("Controllo valori nulli:")
print(f"valori nulli:\n{dataset.isnull().any()}")

print('='*TRAT + '\n')   # separatore

# Vedere il numero di esempi per classi
conteggio_per_classe = dataset['y'].value_counts().sort_index()
print("Controllo conteggio classi:")
print(conteggio_per_classe)

print('='*TRAT + '\n')   # separatore

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
print('='*TRAT + '\n')   # separatore
print(f"distribuzione delle classi in y_test:\n{_test_classi(y_test)}")


# Etichette per i classificatori binari

y_bin = {}
for lable in LABLES:
    y_bin[lable] = [0] * len(y_train)
    for i, ele in enumerate(y_train):
        if ele==lable:
            y_bin[lable][i] = 1
        else:
            y_bin[lable][i] = 0

y_setosa = y_bin['Iris-setosa']
y_versicolor = y_bin['Iris-versicolor']
y_virginica = y_bin['Iris-virginica']


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



############################# Cross validation - Ottimizzazione iperparametri  #########################
string = '''
#############################
INIZIO CROSS VALIDATION
#############################
'''
print(string)

if False:
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
    print(nome + '-'*max(0,(TRAT-len(nome))))
    print('iperparametri migliori: ', gs.best_params_)
    print('accuratezza del modello:', gs.best_score_)
    print('='*TRAT + '\n')   # separatore




        # Decision tree - multiclass

    iperparametri = {
        'modello__max_depth': [3, 5, 7, 10, None],
        'modello__min_samples_split': [2, 5, 10, 20],
        'modello__min_samples_leaf': [1, 2, 4, 8],
        'modello__max_features': [None, 'sqrt', 'log2'],
        'modello__criterion': ['gini', 'entropy'],
        'modello__class_weight': [None, 'balanced']
    }


    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('modello',  DecisionTreeClassifier())
    ])

    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=iperparametri,
        scoring='f1_macro',
        cv=5,
        verbose=True)


    gs.fit(X_train, y_train)
    nome = 'Decision Tree'
    print(nome + '-'*max(0,(TRAT-len(nome))))
    print('iperparametri migliori: ', gs.best_params_)
    print('accuratezza del modello:', gs.best_score_)
    print('='*TRAT + '\n')   # separatore




        # LogisticRegression - classificatori binari
    '''
    Alcune combinazioni di parametri della LogReg che la funzione "GridSearchCV" testa non sono
    valide per questo a run time alcuni fold non producono risultati validi !
    '''

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

    '''
    CLASSIFICATORI BINARI: LogReg e SVC devono essere ottimizzati per ognuna delle 3 classi; dopo la
    definizione di GridSearchCV() deve essere chiamata con ogni etichetta.
    Non posso usare la y_train che contiene più valori ma y_bin che contiene solo 1 o 0 a seconda 
    dell'appartenenza del sample alla classe positiva
    '''

    for etichetta in LABLES :

        gs.fit(X_train, y_bin[etichetta])
        nome = f'LogisticRegression({etichetta})'
        print(nome + '-'*max(0,(TRAT-len(nome))))
        print('iperparametri migliori: ', gs.best_params_)
        print('accuratezza del modello:', gs.best_score_)
        print('='*TRAT + '\n')   # separatore  




        # SVC - classificatori binari

    iperparametri = {
        'modello__C': [0.1, 1, 10, 100],                    # Regolarizzazione
        'modello__kernel': ['rbf', 'linear'],               # Tipo di kernel
        'modello__gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Coefficiente per kernel rbf/poly
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


    for etichetta in LABLES :

        gs.fit(X_train, y_bin[etichetta])
        nome = f'SVC({etichetta})'
        print(nome + '-'*max(0,(TRAT-len(nome))))
        print('iperparametri migliori: ', gs.best_params_)
        print('accuratezza del modello:', gs.best_score_)
        print('='*TRAT + '\n')   # separatore
else:
    print("non eseguo la fase di Cross Validation")
############################# Considerazioni - Model selection  #########################
'''
conclusione...
'''

best_model_hpar = {'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5}
best_model=DecisionTreeClassifier(**best_model_hpar)


############################# Testing  #########################

string = '''
#############################
INIZIO FASE DI TESTING
#############################
'''
print(string)

# scalamento dati di testing e training
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scal = scaler.transform(X_train)
X_test_scal = scaler.transform(X_test)


# Modello multiclasse migliore
best_model.fit(X_train_scal, y_train)
y_pred = best_model.predict(X_test_scal)

accuratezza = accuracy_score(y_test, y_pred)
print(f"l'accurazetta del modello {best_model} è di: {accuratezza}")


# Sistema 1-vs-all Softmax

iperparametri = {'C': 0.001, 'class_weight': None, 'max_iter': 1000, 'penalty': 'l2', 'solver': 'liblinear', 'tol': 0.0001}
modello_setosa = LogisticRegression(**iperparametri)
modello_setosa.fit(X_train_scal, y_setosa)

iperparametri = {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
modello_versicolor = SVC(**iperparametri)
modello_versicolor.fit(X_train_scal, y_versicolor)

iperparametri = {'C': 1, 'gamma': 'scale', 'kernel': 'linear'}
modello_virginica = SVC(**iperparametri)
modello_virginica.fit(X_train_scal, y_virginica)

modelli_binari = [modello_setosa, modello_versicolor, modello_virginica]

def my_softmax(p0: float, p1: float, p2: float):
    """
    Applica la funzione softmax a 3 probabilità (o punteggi) grezze.
    Restituisce un array di 3 probabilità che sommano a 1.
    
    Args
    -----
    p0, p1, p2 : float
        punteggi/probabilità grezze dei 3 modelli binari.
    
    Return
    -----
    softmax_probs : array
        array delle singole probabilità normalizzate
        
    """

    P = np.array([p0, p1, p2])
    P = P - np.max(P)
    exp = np.exp(P)
    softmax_probs = exp / np.sum(exp)
    
    return softmax_probs

def my_softmax_system(X: np.ndarray, modelli: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Questa funzione accetta un dataset e una lista di modelli di classificazione binaria.
    Ogni campione viene elaborato da tutti i modelli e la softmax viene applicata
    ai punteggi per ottenere le probabilità finali.

    Args
    -----
    X : np.ndarray
        dataset da risolvere (shape: n_samples, n_features)

    modelli : list[object]
        lista di modelli binari già addestrati (devono avere .decision_function())

    Return
    -----
    y_pred : np.ndarray
        etichette predette per ogni sample (classe con probabilità massima)

    prob : np.ndarray
        matrice di probabilità (n_samples, n_modelli) dove ogni riga è la distribuzione softmax

    Note
    -----
    La funzione non verifica che i modelli siano binari.
    """
    
    punteggi_list = []
    for modello in modelli:
        try:

            scores = modello.decision_function(X)
            punteggi_list.append(scores)
        except Exception as e:
            print(f"!!! modello non valido: {e}")

            punteggi_list.append(np.zeros(X.shape[0]))

    punteggi_matrix = np.column_stack(punteggi_list)
    
    # implementazione della softmax
    punteggi_stable = punteggi_matrix - np.max(punteggi_matrix, axis=1, keepdims=True)
    exp = np.exp(punteggi_stable)
    prob = exp / np.sum(exp, axis=1, keepdims=True)
    y_pred = np.argmax(prob, axis=1)


    return y_pred, prob


y_pred, prob = my_softmax_system(X= X_train_scal, modelli = modelli_binari)

for i, ele in enumerate(y_pred):
    if ele ==0 : y_pred[i] = 'Iris-setosa'
    elif ele ==1 : y_pred[i] = 'Iris-versicolor'
    elif ele ==2 : y_pred[i] = 'Iris-virginica'
    else: y_pred[i] = 'error'

for i, ele in enumerate(prob):
    print(f"sample{i} -> {y_pred[i]}")
    for i, prob_classe in enumerate(ele):
        print(f"{i}:{prob_classe}")






'''
Testo i due classificatori binari per setosa e virginica per trovare la classe che è
linearmente separabile


# TESTING DI LOGREG(Iris-setosa)

print('='*TRAT + '\n')   # separatore

iperparametri_LogReg_setosa = {'C': 0.001, 'class_weight': None, 'max_iter': 1000, 'penalty': 'l2', 'solver': 'liblinear', 'tol': 0.0001}

modello_setosa = LogisticRegression(**iperparametri_LogReg_setosa)

modello_setosa.fit(X_train_scal, y_setosa)
y_pred = modello_setosa.predict(X_test_scal)

y_test_setosa = []

# trasformo y_test in 0 o 1 a seconda dell'appartenenza a Iris-setosa
y_test_setosa = []
for i, ele in enumerate(y_test):
    if ele=='Iris-setosa':
        y_test_setosa.append(1)
    else:
        y_test_setosa.append(0)

accuratezza = accuracy_score(y_test_setosa, y_pred)
print(f"l'accurazetta del modello {modello_setosa} è di: {accuratezza}")



# TESTING DI LOGREG(Iris-virginica)

print('='*TRAT + '\n')   # separatore

iperparametri_LogReg_virginica = {'C': 10, 'class_weight': None, 'max_iter': 1000, 'penalty': 'l2', 'solver': 'saga', 'tol': 0.0001}
modello_virginica = LogisticRegression(**iperparametri_LogReg_virginica)

modello_virginica.fit(X_train_scal, y_virginica)
y_pred = modello_virginica.predict(X_test_scal)

# trasformo y_test in 0 o 1 a seconda dell'appartenenza a Iris-virginica
y_test_virginica = []
for i, ele in enumerate(y_test):
    if ele=='Iris-virginica':
        y_test_virginica.append(1)
    else:
        y_test_virginica.append(0)

accuratezza = accuracy_score(y_test_virginica, y_pred)
print(f"l'accurazetta del modello {modello_virginica} è di: {accuratezza}")


'''
