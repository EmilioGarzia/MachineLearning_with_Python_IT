# <span style="color:red;">Machine Learning 2.3:</span> <span style="color:blue;">Regressione Multipla</span>
___

## Traccia del problema

Sino ad ora abbiamo applicato la regressione solamente tra due sole *features* del dataset e abbiamo applicato *regressione lineare* o *regressione polinomiale* in base alla distribuzione dei dati, vogliamo ora applicare una nuova tipologia di regressione chiamata ***regressione multipla*** che ci permeterrà di addestrare un modello tenendo conto di due o più *variabili indipendenti*.

L'esercizio che faremo riguarderà le emissioni di CO2 di un veicolo, nello specifico, vogliamo definire le emissioni di `CO2` di un veicolo in base al suo `peso` e al suo `volume`, in questo caso `CO2` sarà la variabile dipendente, mentre le altre due saranno quelle indipendenti, per convenzione si indica con `X` il set delle variabili indipendenti e con `y` la variabile dipendente.

## Dataset

Le informazioni del nostro dataset sono tutte contenute in un file *CSV*, negli scenari reali, infatti, i dati del dataset sono contenuti in un file che sia *CSV*, *MAT*, *TXT*, *ecc.*.

Di seguito, tutte le informazioni contenute nel dataset:

Costruttore|Modello|Volume ($cm^3$)|Peso ($kg$)|CO2 ($g$)
|--|----|-----|----|-----|
Toyota|Aygo|1000|790|99
Mitsubishi|Space Star|1200|1160|95
Skoda|Citigo|1000|929|95
Fiat|500|900|865|90
Mini|Cooper|1500|1140|105
VW|Up!|1000|929|105
Skoda|Fabia|1400|1109|90
Mercedes|A-Class|1500|1365|92
Ford|Fiesta|1500|1112|98
Audi|A1|1600|1150|99
Hyundai|I20|1100|980|99
Suzuki|Swift|1300|990|101
Ford|Fiesta|1000|1112|99
Honda|Civic|1600|1252|94
Hundai|I30|1600|1326|97
Opel|Astra|1600|1330|97
BMW|1|1600|1365|99
Mazda|3|2200|1280|104
Skoda|Rapid|1600|1119|104
Ford|Focus|2000|1328|105
Ford|Mondeo|1600|1584|94
Opel|Insignia|2000|1428|99
Mercedes|C-Class|2100|1365|99
Skoda|Octavia|1600|1415|99
Volvo|S60|2000|1415|99
Mercedes|CLA|1500|1465|102
Audi|A4|2000|1490|104
Audi|A6|2000|1725|114
Volvo|V70|1600|1523|109
BMW|5|2000|1705|114
Mercedes|E-Class|2100|1605|115
Volvo|XC70|2000|1746|117
Ford|B-Max|1600|1235|104
BMW|216|1600|1390|108
Opel|Zafira|1600|1405|109
Mercedes|SLK|2500|1395|120

In python possiamo servirci del modulo ***pandas*** per poter leggere il contenuto di un file *CSV*, dunque, cominciamo subito con la lettura del file.

```python
import pandas

dataset = pandas.read_csv("dataset.csv")
```

## Regressione multipla

In python possiamo servirci del modulo *sklearn* e più nello specifico possiamo utilizzare il metodo `LinearRegression()` di *linear_model* per poter istanziare un oggetto di tipo *regressione lineare*, in quanto, alla base di questo modello vi sarà comunque la regressione lineare, tuttavia, i modelli che utilizzano la regressione per due o più variabili indipendenti sono detti modelli a regressione multipla, infatti, è anche possibile applicare regressione polinomiale su più variabili ed avere cosi un modello che implementa la regressione multipla basata su regressione polinomiale, tutto questo per specificare che la regressione mutipla non è una tecnica, ma, solamente un modo per definire le regressioni che lavorano sulla correlazioni tra più variabili indipendenti.

## Codice

```python
import pandas
from sklearn import linear_model

#leggo le informazioni dal file CSV
dataset = pandas.read_csv("dataset.csv")

#definisco le variabili indipendenti (X) e la variabile dipendente (y)
X = dataset[["Weight", "Volume"]]
y = dataset[["CO2"]]

#applico la regressione multipla e faccio un fitting dei dati
regression = linear_model.LinearRegression()
regression.fit(X, y)
print(regression)

#predico un valore di CO2 in base a due variabili
predictionCO2 = regression.predict([[2300,1300]])
print(predictionCO2)
```

I punti salienti del codice sono tre:

Quando istanziamo un oggetto che rappresenti la nostra regressione lineare

`regression = linear_model.LinearRegression()`

quando addestriamo la regressione lineare con i nostri dati, ovvero, quando eseguiamo un fit dei dati sulla nostra regressione lineare

`regression.fit(X, y)`

infine, andiamo a predire una possibile emissione di `CO2` su un veicolo con $\color{#dc0066}peso=2300kg$ e $\color{#dc0066}volume=1300cm^3$

`predictionCO2 = regression.predict([[2300,1300]])`

##### OUTPUT

`107.2087328`