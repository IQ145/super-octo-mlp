# super-octo-mlp
-PER CAPOLAVORO
Rete neurale che individua se una figura è un triangolo, rombo o quadrato.


SETUP:
1)Scaricare xampp o un server web locale.
2)Scaricare tutti i file e metterli sotto la cartella dove si desidera operare xampp
3)Aprire il control panel di xampp e attivare il modulo Apache
4)Aprire un browser e inserire localhost/ cartella presente il file.

COME USARE MLP:
1)Una volta aperto il file correttamente, premere il pulsante -Comincia-
2)Aspettare finché non si vedono dei punti blu e rossi su tutti i 3 piani cartesiani.
3)Modificare il Training Frequency a piacimento (o lasciarlo invariato)
4)Cliccare -Train!- e aspettare finché non avviene una notifica alert.
5)Chiudi l'alert e crea una figura tra: Rombo, Triangolo, Quadrato
6)Premi -GUESS- e attendi per vedere un alert

7)Vedrai probabilmente una struttura del genere: MLP think is: 1at: 0.9999811548780263%

la prima cifra (0,1,2) sta a significare la figura (triangolo, quadrato, rombo)
0.9999811548780263% è quanto probabile si sente giusto il sistema

COSA SUCCEDE ALL'INTERNO:
1)I dati in formato di matrice 28x28 presenti nel file data.json vengono presi e convertiti in un array di 784 elementi.
2)Questi array vengono centrati e trasformati per essere modificabili da PCA
2)PCA (Analisi delle componenti principali) prende l'insieme di tutti questi dati e li converte in coordinate x e y
3)Si creano 3 MLP (Multi Layered Perceptron) e ad ogni MLP gli si assegna due tipologie di dati: 1)I dati della figura interessata. 2)Tutti gli altri dati.
Quindi ogni MLP è specializzata in osservare solo una singola figura.

Training: 
1)Ad ogni MLP gli viene dato in volte tante volte gli stessi dati.
2)Dopo ogni training, i suoi pesi si correggono di poco fino ad
