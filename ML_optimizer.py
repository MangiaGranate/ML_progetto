from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
import optuna

"""
Dovumentazione di riferimento:
optuna: https://www.geeksforgeeks.org/machine-learning/optuna/
        https://optuna.org/
sklearn: https://scikit-learn.org/stable/api/sklearn.html

Optuna ci permette di trovare per ogni tipo di modello gli iperparametri migliori; a patto che venga definita una funzione obiettivo nella quale riportiamo
il range di ricerca di ogni iperparametri.
Fatto questo ci penserà la libreria a inizializzare lo studio e a portarlo a termine con .optimize().
Nello specifico la funzione obiettivo tenterà ogni combinazione permessa di iperparametri restituendo l'accuratezza per ogni tentativo (sotto forma di media delle varie iterazioni - metodo Cross Validation 
con metrica R2)
Inizializzeremo lo studio in modo che trovi l'iterazione della funzione obiettivo avente restituito R2 maggiore (l'accuratezza maggiore).
L'oggetto "studio" ci fornità effettivamente gli iperparametri adeguati
Ho una funzione per ogni tipo di modello (Gradient Boosting, Random Forest e SVR); ognuma fornità gli iperparametri migliori e l'accuratezza studiata.

n_jobs=-1 in cross_val_score per calcolare i fold in processi diversi in parallelo

"""

def optimize_svr_rbf(X, y, trials=100, timeout=600):    
    """
    Ottimizza gli iperparametri di un modello SVR con kernel RBF tramite Optuna.
    Accetta il dataset di training gia normalizzato se serve

    Parameter
    -----
    X : ndarray
        Numpy array 2D delle features di tutti i sample,
        dove ogni riga corrisponde a un sample

    y : ndarray
        Numpy array 1D dei valori MOS normalizzati in scala 0-1 (y/9)

    trials : int
        Numero di tentativi che optuna deve fare

    timeout : int
        Numero di secondi massimi dedicati ad ogni studio: dopo i quali viene interrotto


    Returns
    -----
    dict con chiavi:

        model : str
            Nome del modello ("svr_rbf")

        accuracy : int o float
            Accuratezza testata tramite cross validation, metrica R2

        params : dict
            Dizionario contenente i parametri ottimali del modello
    """


    def obiettivo(trial):
        """
        Funzione obiettivo per SVR con kernel RBF.
        Esplora C, epsilon e gamma,
        restituisce la media R2 su cross validation a 5 fold.

        Parameter
        -----
        trial : Any
            doc: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
            Un valutatore della funzione, serve a contenere iperparametri e tipo di modello

        Return
        -----
        scores.mean() : int o float
            Media dei valori d'accuratezza di tutti i fold generati dalla funzione di cross validation per ogni trial

        Note
        -----
        Parametri di ottimizzazione:
            SVR: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
        """




        # SVR: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

        C       = trial.suggest_float("C", 1e-2, 1e3, log=True)
        epsilon = trial.suggest_float("epsilon", 1e-3, 1.0, log=True)
        gamma   = trial.suggest_float("gamma", 1e-4, 1, log=True)

        


        # TRAINING & TESTING con cross validation
        # METRICHE: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names

        #SCALER - MODELLO Questo scaler vale solo all'interno dello studio
        pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma))
        ])


        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        scores = cross_val_score(pipeline, X, y, cv=kf, scoring='r2', n_jobs=-1)
        return scores.mean()


    study = optuna.create_study(direction='maximize')
    study.optimize(obiettivo, n_trials=trials, show_progress_bar=True, timeout=timeout)

    # Risultati
    best          = study.best_trial
    best_accuracy = best.value
    best_params   = {name: value for name, value in best.params.items()}

    print(f"Miglior accuracy: {best_accuracy}")
    print(f"Parametri       : {best_params}")

    


    return {
        "model"    : "svr_rbf",
        "accuracy" : best_accuracy,
        "params"   : best_params
    }


def optimize_svr_poly(X, y, trials=100, timeout=600):
    """
    Ottimizza gli iperparametri di un modello SVR con kernel Poly tramite Optuna.

    Parameter
    -----
    X : ndarray
        Numpy array 2D delle features di tutti i sample,
        dove ogni riga corrisponde a un sample

    y : ndarray
        Numpy array 1D dei valori MOS normalizzati in scala 0-1 (y/9)

    trials : int
        Numero di tentativi che optuna deve fare

    Returns
    -----
    dict con chiavi:

        model : str
            Nome del modello ("svr_poly")

        accuracy : int o float
            Accuratezza testata tramite cross validation, metrica R2

        params : dict
            Dizionario contenente i parametri ottimali del modello
    """

    def obiettivo(trial):
        """
        Funzione obiettivo per SVR con kernel Poly.
        Esplora C, epsilon, degree, gamma e coef0,
        restituisce la media R2 su cross validation a 5 fold.

        Parameter
        -----
        trial : Any
            doc: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
            Un valutatore della funzione, serve a contenere iperparametri e tipo di modello

        Return
        -----
        scores.mean() : int o float
            Media dei valori d'accuratezza di tutti i fold generati dalla funzione di cross validation per ogni trial

        Note
        -----
        Parametri di ottimizzazione:
            SVR: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
        """

        # SVR: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
        C       = trial.suggest_float("C", 1e-2, 1e3, log=True)
        epsilon = trial.suggest_float("epsilon", 1e-3, 1.0, log=True)
        degree  = trial.suggest_int("degree", 2, 5)
        gamma   = trial.suggest_float("gamma", 1e-4, 1, log=True)
        coef0   = trial.suggest_float("coef0", -1, 1)

        #SCALER - MODELLO Questo scaler vale solo all'interno dello studio, non è associato all'oggetto Model
        pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  SVR(kernel="poly", C=C, epsilon=epsilon, degree=degree, gamma=gamma, coef0=coef0))
        ])

        # TRAINING & TESTING
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        # METRICHE: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names
        scores = cross_val_score(pipeline, X, y, cv=kf, scoring='r2', n_jobs=-1)
        return scores.mean()


    study = optuna.create_study(direction='maximize')
    study.optimize(obiettivo, n_trials=trials, show_progress_bar=True, timeout=timeout)

    # Risultati
    best          = study.best_trial
    best_accuracy = best.value
    best_params   = {name: value for name, value in best.params.items()}

    print(f"Miglior modello : svr_poly")
    print(f"Miglior accuracy: {best_accuracy}")
    print(f"Parametri       : {best_params}")

    return {
        "model"    : "svr_poly",
        "accuracy" : best_accuracy,
        "params"   : best_params
    }


def optimize_svr_sigmoid(X, y, trials=100, timeout=600):
    """
    Ottimizza gli iperparametri di un modello SVR con kernel Sigmoid tramite Optuna.

    Parameter
    -----
    X : ndarray
        Numpy array 2D delle features di tutti i sample,
        dove ogni riga corrisponde a un sample

    y : ndarray
        Numpy array 1D dei valori MOS normalizzati in scala 0-1 (y/9)

    trials : int
        Numero di tentativi che optuna deve fare

    Returns
    -----
    dict con chiavi:

        model : str
            Nome del modello ("svr_sigmoid")

        accuracy : int o float
            Accuratezza testata tramite cross validation, metrica R2

        params : dict
            Dizionario contenente i parametri ottimali del modello
    """

    def obiettivo(trial):
        """
        Funzione obiettivo per SVR con kernel Sigmoid.
        Esplora C, epsilon, gamma e coef0,
        restituisce la media R2 su cross validation a 5 fold.

        Parameter
        -----
        trial : Any
            doc: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
            Un valutatore della funzione, serve a contenere iperparametri e tipo di modello

        Return
        -----
        scores.mean() : int o float
            Media dei valori d'accuratezza di tutti i fold generati dalla funzione di cross validation per ogni trial

        Note
        -----
        Parametri di ottimizzazione:
            SVR: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
        """

        # SVR: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
        C       = trial.suggest_float("C", 1e-2, 1e3, log=True)
        epsilon = trial.suggest_float("epsilon", 1e-3, 1.0, log=True)
        gamma   = trial.suggest_float("gamma", 1e-4, 1, log=True)
        coef0   = trial.suggest_float("coef0", -1, 1)



        #SCALER - MODELLO Questo scaler vale solo all'interno dello studio, non è associato all'oggetto Model
        pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  SVR(kernel="sigmoid", C=C, epsilon=epsilon, gamma=gamma, coef0=coef0))
        ])

        # TRAINING & TESTING
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        # METRICHE: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names
        scores = cross_val_score(pipeline, X, y, cv=kf, scoring='r2', n_jobs=-1)
        return scores.mean()


    study = optuna.create_study(direction='maximize')
    study.optimize(obiettivo, n_trials=trials, show_progress_bar=True, timeout=timeout)

    # Risultati
    best          = study.best_trial
    best_accuracy = best.value
    best_params   = {name: value for name, value in best.params.items()}

    print(f"Miglior modello : svr_sigmoid")
    print(f"Miglior accuracy: {best_accuracy}")
    print(f"Parametri       : {best_params}")

    return {
        "model"    : "svr_sigmoid",
        "accuracy" : best_accuracy,
        "params"   : best_params
    }


def optimize_gradient_boosting(X, y, trials=100, timeout=600):
    """
    Ottimizza gli iperparametri di un modello GradientBoostingRegressor tramite Optuna.

    Parameter
    -----
    X : ndarray
        Numpy array 2D delle features di tutti i sample,
        dove ogni riga corrisponde a un sample

    y : ndarray
        Numpy array 1D dei valori MOS normalizzati in scala 0-1 (y/9)

    trials : int
        Numero di tentativi che optuna deve fare

    Returns
    -----
    dict con chiavi:

        model : str
            Nome del modello ("gradient_boosting")

        accuracy : int o float
            Accuratezza testata tramite cross validation, metrica R2

        params : dict
            Dizionario contenente i parametri ottimali del modello
    """

    def obiettivo(trial):
        """
        Funzione obiettivo per GradientBoostingRegressor.
        Esplora n_estimators, learning_rate e max_depth,
        restituisce la media R2 su cross validation a 5 fold.

        Parameter
        -----
        trial : Any
            doc: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
            Un valutatore della funzione, serve a contenere iperparametri e tipo di modello

        Return
        -----
        scores.mean() : int o float
            Media dei valori d'accuratezza di tutti i fold generati dalla funzione di cross validation per ogni trial

        Note
        -----
        Parametri di ottimizzazione:
            GradientBoostingRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
        """

        n_estimators  = trial.suggest_int("n_estimators", 50, 300)
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        max_depth     = trial.suggest_int("max_depth", 2, 8)

        modello = GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=0
        )

        # TRAINING & TESTING
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        # METRICHE: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names
        scores = cross_val_score(modello, X, y, cv=kf, scoring='r2', n_jobs=-1)
        return scores.mean()


    study = optuna.create_study(direction='maximize')
    study.optimize(obiettivo, n_trials=trials, show_progress_bar=True, timeout=timeout)

    # Risultati
    best = study.best_trial
    best_accuracy = best.value
    best_params = {name: value for name, value in best.params.items()}

    print(f"Miglior modello : gradient_boosting")
    print(f"Miglior accuracy: {best_accuracy}")
    print(f"Parametri       : {best_params}")

    return {
        "model"    : "gradient_boosting",
        "accuracy" : best_accuracy,
        "params"   : best_params
    }


def optimize_random_forest(X, y, trials=100, timeout=600):
    """
    Ottimizza gli iperparametri di un modello RandomForestRegressor tramite Optuna.

    Parameter
    -----
    X : ndarray
        Numpy array 2D delle features di tutti i sample,
        dove ogni riga corrisponde a un sample

    y : ndarray
        Numpy array 1D dei valori MOS normalizzati in scala 0-1 (y/9)

    trials : int
        Numero di tentativi che optuna deve fare

    Returns
    -----
    dict con chiavi:
    
        model : str
            Nome del modello ("random_forest")

        accuracy : int o float
            Accuratezza testata tramite cross validation, metrica R2

        params : dict
            Dizionario contenente i parametri ottimali del modello
    """

    def obiettivo(trial):
        """
        Funzione obiettivo per RandomForestRegressor.
        Esplora n_estimators e max_depth,
        restituisce la media R2 su cross validation a 5 fold.

        Parameter
        -----
        trial : Any
            doc: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
            Un valutatore della funzione, serve a contenere iperparametri e tipo di modello

        Return
        -----
        scores.mean() : int o float
            Media dei valori d'accuratezza di tutti i fold generati dalla funzione di cross validation per ogni trial

        Note
        -----
        Parametri di ottimizzazione:
            RandomForestRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
        """

        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth    = trial.suggest_int("max_depth", 3, 20)

        modello = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=0
        )

        # TRAINING & TESTING
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        # METRICHE: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names
        scores = cross_val_score(modello, X, y, cv=kf, scoring='r2', n_jobs=-1)
        return scores.mean()


    study = optuna.create_study(direction='maximize')
    study.optimize(obiettivo, n_trials=trials, show_progress_bar=True, timeout=timeout)

    # Risultati
    best = study.best_trial
    best_accuracy = best.value
    best_params = {name: value for name, value in best.params.items()}

    print(f"Modello : random_forest")
    print(f"Miglior accuracy: {best_accuracy}")
    print(f"Parametri       : {best_params}")

    return {
        "model"    : "random_forest",
        "accuracy" : best_accuracy,
        "params"   : best_params
    }

    



"""
Le seguenti funzioni sono di prova, servono per sperimentare i range di iperparametri e i valutatori degli studi...
"""



def optimize_svr_rbf_test_parametri(X, y, trials=100, timeout=600):    
    """
    Ottimizza gli iperparametri di un modello SVR con kernel RBF tramite Optuna.
    Accetta il dataset di training gia normalizzato se serve

    Parameter
    -----
    X : ndarray
        Numpy array 2D delle features di tutti i sample,
        dove ogni riga corrisponde a un sample

    y : ndarray
        Numpy array 1D dei valori MOS normalizzati in scala 0-1 (y/9)

    trials : int
        Numero di tentativi che optuna deve fare

    timeout : int
        Numero di secondi massimi dedicati ad ogni studio: dopo i quali viene interrotto


    Returns
    -----
    dict con chiavi:

        model : str
            Nome del modello ("svr_rbf")

        accuracy : int o float
            Accuratezza testata tramite cross validation, metrica R2

        params : dict
            Dizionario contenente i parametri ottimali del modello
    """


    def obiettivo(trial):
        """
        Funzione obiettivo per SVR con kernel RBF.
        Esplora C, epsilon e gamma,
        restituisce la media R2 su cross validation a 5 fold.

        Parameter
        -----
        trial : Any
            doc: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
            Un valutatore della funzione, serve a contenere iperparametri e tipo di modello

        Return
        -----
        scores.mean() : int o float
            Media dei valori d'accuratezza di tutti i fold generati dalla funzione di cross validation per ogni trial

        Note
        -----
        Parametri di ottimizzazione:
            SVR: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
        """




        # SVR: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

        C = trial.suggest_float('C', 1e2, 1e3)
        gamma = trial.suggest_float('gamma', 1e-2, 1 )
        epsilon = trial.suggest_float('epsilon', 1, 10)

        


        # TRAINING & TESTING con cross validation
        # METRICHE: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names

        #SCALER - MODELLO Questo scaler vale solo all'interno dello studio, non è associato all'oggetto Model
        pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma))
        ])


        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        scores = cross_val_score(pipeline, X, y, cv=kf, scoring='r2', n_jobs=-1)
        return scores.mean()


    study = optuna.create_study(direction='maximize')
    study.optimize(obiettivo, n_trials=trials, show_progress_bar=True, timeout=timeout)

    # Risultati
    best          = study.best_trial
    best_accuracy = best.value
    best_params   = {name: value for name, value in best.params.items()}

    print(f"Miglior accuracy: {best_accuracy}")
    print(f"Parametri       : {best_params}")

    


    return {
        "model"    : "svr_rbf",
        "accuracy" : best_accuracy,
        "params"   : best_params
    }


def optimize_svr_rbf_test_metrica(X, y, trials=100, timeout=600):    
    """
    Ottimizza gli iperparametri di un modello SVR con kernel RBF tramite Optuna.
    Accetta il dataset di training gia normalizzato se serve

    Parameter
    -----
    X : ndarray
        Numpy array 2D delle features di tutti i sample,
        dove ogni riga corrisponde a un sample

    y : ndarray
        Numpy array 1D dei valori MOS normalizzati in scala 0-1 (y/9)

    trials : int
        Numero di tentativi che optuna deve fare

    timeout : int
        Numero di secondi massimi dedicati ad ogni studio: dopo i quali viene interrotto


    Returns
    -----
    dict con chiavi:

        model : str
            Nome del modello ("svr_rbf")

        accuracy : int o float
            Accuratezza testata tramite cross validation, metrica R2

        params : dict
            Dizionario contenente i parametri ottimali del modello
    """


    def obiettivo(trial):
        """
        Funzione obiettivo per SVR con kernel RBF.
        Esplora C, epsilon e gamma,
        restituisce la media R2 su cross validation a 5 fold.

        Parameter
        -----
        trial : Any
            doc: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
            Un valutatore della funzione, serve a contenere iperparametri e tipo di modello

        Return
        -----
        scores.mean() : int o float
            Media dei valori d'accuratezza di tutti i fold generati dalla funzione di cross validation per ogni trial

        Note
        -----
        Parametri di ottimizzazione:
            SVR: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
        """




        # SVR: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

        C       = trial.suggest_float("C", 1e-2, 1e3, log=True)
        epsilon = trial.suggest_float("epsilon", 1e-3, 1.0, log=True)
        gamma   = trial.suggest_float("gamma", 1e-4, 1, log=True)

        


        # TRAINING & TESTING con cross validation
        # METRICHE: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names

        #SCALER - MODELLO Questo scaler vale solo all'interno dello studio, non è associato all'oggetto Model
        pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma))
        ])


        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        scores = cross_val_score(pipeline, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
        return scores.mean()


    study = optuna.create_study(direction='maximize')
    study.optimize(obiettivo, n_trials=trials, show_progress_bar=True, timeout=timeout)

    # Risultati
    best          = study.best_trial
    best_accuracy = best.value
    best_params   = {name: value for name, value in best.params.items()}

    print(f"Miglior accuracy: {best_accuracy}")
    print(f"Parametri       : {best_params}")

    


    return {
        "model"    : "svr_rbf",
        "accuracy" : best_accuracy,
        "params"   : best_params
    }


def optimize_svr_rbf_test_parametriemetrica(X, y, trials=100, timeout=600):    

    """
    Ottimizza gli iperparametri di un modello SVR con kernel RBF tramite Optuna.
    Accetta il dataset di training gia normalizzato se serve

    Parameter
    -----
    X : ndarray
        Numpy array 2D delle features di tutti i sample,
        dove ogni riga corrisponde a un sample

    y : ndarray
        Numpy array 1D dei valori MOS normalizzati in scala 0-1 (y/9)

    trials : int
        Numero di tentativi che optuna deve fare

    timeout : int
        Numero di secondi massimi dedicati ad ogni studio: dopo i quali viene interrotto


    Returns
    -----
    dict con chiavi:

        model : str
            Nome del modello ("svr_rbf")

        accuracy : int o float
            Accuratezza testata tramite cross validation, metrica R2

        params : dict
            Dizionario contenente i parametri ottimali del modello
    """


    def obiettivo(trial):
        """
        Funzione obiettivo per SVR con kernel RBF.
        Esplora C, epsilon e gamma,
        restituisce la media R2 su cross validation a 5 fold.

        Parameter
        -----
        trial : Any
            doc: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
            Un valutatore della funzione, serve a contenere iperparametri e tipo di modello

        Return
        -----
        scores.mean() : int o float
            Media dei valori d'accuratezza di tutti i fold generati dalla funzione di cross validation per ogni trial

        Note
        -----
        Parametri di ottimizzazione:
            SVR: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
        """




        # SVR: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

        C = trial.suggest_float('C', 1e2, 1e3)
        gamma = trial.suggest_float('gamma', 1e-2, 1 )
        epsilon = trial.suggest_float('epsilon', 1, 10)

        


        # TRAINING & TESTING con cross validation
        # METRICHE: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names

        #SCALER - MODELLO Questo scaler vale solo all'interno dello studio, non è associato all'oggetto Model
        pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma))
        ])


        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        scores = cross_val_score(pipeline, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
        return scores.mean()


    study = optuna.create_study(direction='maximize')
    study.optimize(obiettivo, n_trials=trials, show_progress_bar=True, timeout=timeout)

    # Risultati
    best          = study.best_trial
    best_accuracy = best.value
    best_params   = {name: value for name, value in best.params.items()}

    print(f"Miglior accuracy: {best_accuracy}")
    print(f"Parametri       : {best_params}")

    


    return {
        "model"    : "svr_rbf",
        "accuracy" : best_accuracy,
        "params"   : best_params
    }

# gia testate quelle in alto: pessimi risultati


