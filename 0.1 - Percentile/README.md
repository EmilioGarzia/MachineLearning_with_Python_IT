# <span style="color:red;">Machine Learning 0.1:</span> <span style="color:blue;">Calcolare il Percentile</span>
___
## Traccia del problema
In questo semplice esercizio ci preoccupiamo di individuare il percentile di 50 circa l'età di tutti i cittadini di una strada.
Poniamo caso dunque che il nostro dataset sia del tipo:
|Nome|Età|
|----|---|
|Alfio|5|
|Antonio|31|
|Anna|43|
|Michela|48|
|Angelo|50|
|Manuele|41|
|Luigi|7|
|Maria|11|
|Bruna|15|
|Emilio|39|
|Ciro|80|
|Tonya|82|
|Rodolfo|32|
|Goku|2|
|Vegeta|8|
|Crillin|6|
|Bulma|25|
|Broly|36|
|Tarles|27|
|Bardak|61|
|Seripa|31|

## Percentile
Nel caso del dataset di cui sopra potremmo utilizzare il percentile per rispondere alla domanda:
> **Q:** Qual'è il 50° percentile del nostro dataset?\
>**A:** La risposta a questa domanda è: 31.0

Questo significa che il $50\\%$ delle persone nella strada ha un'età pari o inferiore a $31$ anni.

In definitiva possiamo dire che il percentile è un dato molto utile quando ragioniamo a livello statistico e ci permette di conoscere quale percentuale di valori è inferiore.

Il calcolo del percentile in *python* è possibile grazie alla funzione ***percentile()*** di *numpy*, nella quale passiamo in input la collezione di dati e il percentile che vogliamo calcolarci:

```python
import numpy as np
ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]
percent_50 = np.percentile(ages, 50)
```
