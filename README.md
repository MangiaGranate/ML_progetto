# ML_progetto
Progetto universitario: addestramento di più modelli ML con ottimizzazione iperparametri e model selection basandosi sul dataset "Iris" di "R.A. Fisher" (https://archive.ics.uci.edu/dataset/53/iris)

# Relazione: 

RELAZIONE_ML


			Introduzione

In questa relazione tratteremo un task di classificazione multiclasse; Dati 4 features dovremo estrapolare la classe di appartenenza tra le tre possibili. Ho deciso di sviluppare in parallelo due soluzioni diverse:
1. un singolo modello multiclasse ottimizzato 
2. un sistema 1-vs-all in cui per ogni classe ottimizzo un singolo modello di classificazione binaria (caso più interessante in quanto sfrutta la natura del dataset come vedremo)

Informazioni sul Dataset:
il dataset Iris offre 150 sample da 4 features float e una label che può assumere 3 classi di valori: Iris-setosa, Iris-versicolor e Iris-virginica.
Il dataset è gia omogeneo in quanto offre 50 campioni per ogni classe e questo semplifica molto il lavoro.

Tra le informazioni del dataset (Iris.names) ho appreso che una delle 3 classi è liearmente separabile dalle altre e intendo sfruttare questa caratteristica nel sistema 1-vs-all.

Il grande problema di questo dataset è che contiene pochi sample (150), per questo motivo ho deciso di tenere solo il 10% (15) dei campioni dei dati come testing set per il modello finale.

Sul training set ho deciso di adottare la k-fold con k=5 per suddividere il dataset di training durante l’ottimizzazione degli iperparametri e la scelta del modello


		Scelte dei modelli da testare

Trattandosi di un task di classificazione multiclasse sicuramente vorrei testare i modelli DecisionTreeClassifier() e KneighborsClassifier(); oltre a questi vorrei testare un sistema 1-vs-all.

DecisionTreeClassifier e KneighborsClassifier verranno ottimizzate singolarmente in quanto gia supportano il multiclass; per quanto riguarda il metodo 1-vs-all non voglio usare la soluzione di sklearn OneVsRestClassifier() perché mi vincola ad usare la stessa tipologia di modello per tutte le classi.

		1-vs-all

Per sfruttare il fatto che una classe sia linearmente separabile dalle altre ho deciso che tra i modelli da scegliere per ogni classe userò un LogisticRegression() e un SVC() con kernel sia lineare che gaussiano. Dovrò quindi per ogni classe allenare tutte le tipologie di modello (lineari e non). Mi aspetto che la cross validation mi selezioni 2 modelli non lineari e uno lineare.

La funzione di aggregazione che interpreterà i modelli (gia ottimizzati) deve essere fatta a mano visti i limiti di  OneVsRestClassifier()

Fino alla fase di testing quindi i modelli binari verranno ottimizzati singolarmente, sarà solo durante il testing che dovrò definire una funzione di aggregazione per interpretare le probabilità.

Softmax – Addestrati e ottimizzati i tre modelli binari estrapolo in fase di testing le loro probabilità e le interpreto con una funzione softmax in modo da ottenere non solo la classe più probabile sencondo il modello ma anche il grado di sicurezza che ha. Questa funzione è tale da testituire 3 probabilità la cui somma è uguale a 1, accettando i singoli punteggi di probabilità di ogni modello di classificazione binario.
		

		_valutatore_kfold()

FONTE:	doc StratifiedKFold

Inizialmente ho pensato di implementare le funzioni del cross validation a mano per rendere i passaggi espliciti e per questo ho deciso di definire “_valutatore_kfold”.

“_valutatore_kfold” si occupa di testare un modello gia definito (iperparametri gia fissati) in k-fold omogenei, restituendo la media dei valutatori.

L’idea era quella di chiamare “_valutatore_kfold” all’interno di un ciclo che testasse per ogni modello le varie combinazioni di iperparametri.

Ad ogni fold il dataset veniva partizionato tramite StratifiedKFold() che, a differenza di Kfold(), divide il dataset in sottogruppi omogenei.


In seguito ho deciso però di adottare le funzioni di libreria e tenere _valutatore_kfold come dimostrazione del funzionamento di StratifiedKFold()

		Ottimizzazione iperpatametri: GrifSearchCV()

L’ottimizzazione degli iperparametri è eseguita tramite la funzione GridSearchCV().
Questa funzione esegue l’intero processo di cross validation variando gli iperparametri e provando tutte le combinazioni fornite.
	
		Configurazione di GridSearchCV()

FONDE:	doc GridSearchCV

La voce estimator di GridSearchCV ci permette di specificare quale modello utilizzare nell’ottimizzazione, in particolare estimator deve essere una funzione che implementi il metodo .fit() per accettare i dati ti training e il metodo .predict() per poter confrontare il modello con i dati di testing/validator.

La voce cv di GridSearchCV ci permette di specificare la procedura di testing del modello (non le metriche da usare che invece sono scelte in scoring); in particolare ho deciso di usare la funzione StratifierKFold() per ottenere fold omogenei.
(se a cv viene fornito un int la funzione usa di default la StratifierKFold)

In fine per i modelli che necessitano di scalamento voglio che ad ogni fold i dati di training e validator vengano scalati basandosi sui soli dati di training di quel fold; per fare questo fornisco all’estimator di GridSearchCV una pipeline.

Era possibile anche scalare i dati prima del cross validation tuttavia eseguendo lo scalamento ad ogni fold non c’è data leaking neanche durante l’ottimizzazione degli iperparametri.

FONDE:	doc Pipeline

Pipeline di Sklearn - L’oggetto Pipeline di sklearn ci permette di concatenare più oggetti sklearn che verranno eseguite in successione ad ogni fold.

Nello specifico si definisce una lista di oggetti i quali devono avere i metodi .fit() e .transform(); l’ultimo oggetto può avere .predict() (cioè deve essere un "estimatore", come un classificatore o un regressore).

Nel nostro caso, per i modelli che ne hanno bisogno, passiamo a  GridSearchCV() una pipeline contenente uno scaler in modo da scalare i dati per ogni fold ed evitare che la dimenzione delle features infuenzi la loro importanza a training time.


		EDA

Durante la prima fase di EDA, subito dopo aver estratto il dataset tramite pandas, non ho dovuto alterare il dataset in quanto non vi erano valori nulli e le i sample al suo interno erano già omogenee.

Tuttavia ho dovuto generare le etichette per i classificatori binari in quanto, per ogni modello, le lable non dovevano assumere il valore della classe bensì un valore binario (0 o 1) a seconda dell’appartenenza della classe studiata.


		Ricerca della classe linearmente separabile

Durante la cross validation del modello LogisticRegression sono stati ottimizzati gli iperparametri per ognuna delle tre possibili classi; tuttavia è risultata un accuratezza sospetta del 100% sulla classe “Iris-setosa”

log:

Fitting 5 folds for each of 96 candidates, totalling 480 fits
LogisticRegression(Iris-setosa)-----------------------------------------------------------------------------------------
iperparametri migliori:  {'modello__C': 0.001, 'modello__class_weight': None, 'modello__max_iter': 1000, 'modello__penalty': 'l2', 'modello__solver': 'liblinear', 'modello__tol': 0.0001}
accuratezza del modello: 1.0
=============================

Mentre le altre due classi hanno ricevuto un accuratezza di 0.72 0.96. Dal momento che, dagli appunti del dataset, una delle classi è linearmente separabile dalle altre non è detto che si tratti di un caso di Overfitting ma per sicurezza ho ritenuto opportuno testare il modello con i dati di testing sul quale non è stato allenato.  (vedere il file di log per tutti i risultati...)

Dal testing risulta un accuratezza di 0.95; come prima ipotesi mi sembra corretto dire che la classe “Iris-setosa” sia tra le tre quella linearmente separabile dalle altre; tuttavia un punteggio di 0.96 della classe “Iris-virignica” (ottenuto in cross validation) è sospetto per una classe che non dovrebbe esserlo. Ho deciso di testare con il dataset di testing anche questa classe per vedere se l’accuratezza scende.
(questi due test si trovano commentati in fondo al file main.py)

Alla fine del test sul modello virginica ho ottenuto un accuratezza dello 0.64; ne deduco che questa classe non sia perfettamente separabile linearmente dalle altre.

Conclusione – Dopo la cross validation il modello più performante per la classe ‘Iris-virginica’ è si l’SVC ma è stato selezionato un kernel lineare. Questa classe sembra a prima vista linearmente separabile dalle altre; tuttavia quando abbiamo testato sul dataset di testing abbiamo ottenuto un accuratezza bassa (0.64) inoltre lo stesso provider del dataset ci ha avvisato che solo una classe è linearmente separabile dalle altre.
In conclusione sembra che ‘Iris-setosa’ si trovi lontana dalle altre due classi nello spazio delle featuers e sia facimente separabile linearmente da entrambe mentre ‘Iris-versicolor’ e ‘Iris-virginica’ siano vicine e non linearmente separabili. Il cross validation ci ha dato un accuratezza alta e ci ha proposto un modello lineare per ‘Iris-virginica’ perché nel 1-vs-all il modello ha riconoscuto correttamente tutti i sample di ‘Iris-setosa’ e quelli più lontani di ‘Iris-versicolor’ ma questo risultato potrebbe essere forviante: un sample di ‘Iris-versicolor’ potrebbe confondere il modello virginica; la mia speranza è che il modello versicolor (SVC non lineare)invece distingua sufficientemente bene i prori sample da quelli di ‘Iris-virginica’.

Rimango dell’idea che sia opportuno per il 1-vs-all tenere per ‘Iris-virginica’ un modello lineare sperando che il modello di ‘Iris-versicolor’ discrimini sufficientemente questi sample di mezzo, confermando così la soluzione trovata durante il cross validation.

Classe				modello					punteggio CV
‘Iris-setosa’			LogReg (solver lineare)		1.0
‘Iris-verticolor’		SVC kernel non lineare		0.98
‘Iris-virginica’		SVC kernel lineare			0.97


Conclusione finale – Il sistema 1-vs-all è più accurato del miglior modello multiclasse (0.977 vs 0.911) probabilmente perché riesce a sfruttare meglio la natura del dataset.

Vedere il file di log per tutte le metriche di classificazione


l'accurazetta del modello multiclasse migliore è di: 0.91111111111
la precisione del modello multiclasse migliore è di: 
[1.         0.86666667 0.86666667]
recall del modello multiclasse migliore è di: 
[1.         0.86666667 0.86666667]
f1 del modello multiclasse migliore è di: 
[1.         0.86666667 0.86666667]

=============================

l'accurazetta del sistema Softmax è di: 0.9777777
la precisione del sistema Softmax è di: 
[1.     1.     0.9375]
recall del sistema Softmax è di: 
[1.         0.93333333 1.        ]
f1 del sistema Softmax è di: 
[1.         0.96551724 0.96774194]



		Iperparametri ottimizzati:


	K-Nearest-Neighbors
In questo modello ottimizziamo n_neighbors, se troppo piccolo tenderà ad essere troppo sensibile al rumode overfitting; se troppo grande tenderà a restituisce valori poco specifici, underfitting. L’ottimizzazione sceglie anche l’algoritmo da adottare (algorithm) e la metrica per la distanza (metric).
Un altro iperparametro ottimizzato è il size delle foglie (leaf_size)


	Decision Tree
max_depth – per fissare la profondità massima dell’albero, se troppo profondo rischia di fare overfitting
min_samples_split / min_samples_leaf – per fissare il numero minimo di sample per dividere un nodo o che deve avere un nodo foglia
max_features – numero massimo di features considerate per i test
criterios – è il criterio di valutazione dello split
class_weight – il peso per ogni classe in base al numero di sample nel dataset, non serve se il dataset è bilanciato come in questo caso.

	Logistic Regression
C – regolazione del modello, parametri che indica quanto penalizzare gli errori di classificazione durante l’addestramento; se troppo piccolo il modello tenderà a sbagliare troppo (underfitting); se troppo grande il modello sarà troppo rigoroso (overfitting)
penalty – è il tipo di penalità da applicare ai coefficienti in modo che non siano più grandi del necessario
solver – algoritmo di ottimizzazione dei parametri (con max_iter numero massimo di iterazioni)
tol – tolleranza nel criterio di stop 

	SVC
Come nella Logistic regression troviamo la regolazione C
Kernel – tipo di kernel da applicare per trasformare il features space (gamma che definisce l’influenza del singolo sample)

