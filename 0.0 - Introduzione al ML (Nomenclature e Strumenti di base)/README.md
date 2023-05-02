# <span style="color:red;">Machine Learning 0.0:</span> <span style="color:blue;">Introduzione al ML</span>
___
## Traccia del problema
In questo primo esercizio vogliamo fingiamo di avere un semplice ***dataset*** che contenga la velocità di alcuni veicoli e su queste informazioni applicare i principali strumenti matematici.

## Dataset
Un ***dataset*** è molto banalmente una struttura che contiene un insieme di informazioni da dare in pasto alla nostra intelligenza artificiale, in altre parole, un dataset è il blocco di dati che utilizza la IA per **apprendere** e diventare *"brava"* a risolvere quel determinato problema.
Nel nostro caso il dataset è estremamente banale e consta di una serie di veicoli:
|MANUFACTER|COLOR|AGE|SPEED|TELEPASS|
|----------|-----|---|-----|--------|
|BMW|red|5|99|Y|
|Volvo|black|7|86|Y|
|Volkswagen|gray|8|87|N|
|Volkswagen|white|7|88|Y|
|Ford|white|2|111|Y|
|Volkswagen|white|17|86|Y|
|Tesla|red|2|103|Y|
|BMW|black|9|87|Y|
|Volvo|gray|4|94|N|
|Ford|white|11|78|N|
|Toyota|gray|12|77|N|
|Volkswagen|white|9|85|N|
|Toyota|blue|6|86|Y|

Analizzando il dataset di cui sopra possiamo dire alcune cose, per esempio, che il colore più popolare è il bianco o che l'auto più vecchia ha 17 anni, ma, oltre a queste banali informazioni cosa possiamo stabilire?
Quello che potremmo fare è stabilire, in base alla velocità con cui viene superato il casello, se l'auto sia o meno munita di telepass e il ML è proprio quello che fa al caso nostro, infatti, dando questo dataset ad una rete neurale è possibile addestrarla affinchè quest'ultima riesca a stabilire se un veicolo sia o meno munito telepass.

## Strumenti matematici di base
Nel nostro codice di esempio abbiamo messo sotto esame solamente la ***features*** *"speed"* raccogliendo in un vettore la velocità di passaggio di ogni veicolo.
A questo punto su questo insieme di dati possiamo matematicamente estrapolare alcune basilari informazioni come:
### **MEDIA** *(mean)*
Molto semplicemente calcolare la media aritmetica, nel nostro caso data da:

$$
\color{#DC006C}mean(speed) = \frac{99+86+87+88+111+86+103+87+94+78+77+85+86}{13}=89.77
$$

⚠: *In python possiamo calcolarci la media di un set di numeri utilizzando il metodo **"mean()"** di numpy.*

```python
import numpy as np
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
mean = np.mean(speed)
```
### **MEDIANO** *(median)*
Il mediano è l'elemento che si trova nel mezzo del vettore di dati dopo averlo riordinato in ordine crescente, nel nostro caso il mediano equivale a:

$$
\color{#DC006C}median(speed) = 77,78,85,86,86,86,\underline{\textbf{87}},87,88,94,99,103,111 = 87
$$

⚠: *In python possiamo calcolarci il mediano di un set di numeri utilizzando il metodo **"median()"** di numpy, non è necessario riordinare il vettore, in quanto, sarà fatto automaticamente dal metodo.*
```python
import numpy as np
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
median = np.median(speed)
```

### **MODALITA'** *(mode)*
La *mode* è il valore che appare con maggior frequenza nella collezione di dati, nel nostro caso:

$$
\color{#DC006C}mode(speed) = 99,\underline{\textbf{86}},87,88,111,\underline{\textbf{86}},103,87,94,78,77,85,\underline{\textbf{86}} = 86
$$

⚠: *La funzione **"mode()"** di **scipy.stats** ci permette di calcolarci proprio la modalità, in output ci verrà ritornato un oggetto che contiene il valore con la maggior frequenza e il numero di occorrenze di quest'ultimi.*
```python
from scipy import stats
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
mode = stats.mode(speed)
```
