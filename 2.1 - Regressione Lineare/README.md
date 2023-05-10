# <span style="color:red;">Machine Learning 2.1:</span> <span style="color:blue;">Regressione Lineare per la previsione dei dati</span>
___
## Traccia del problema
In questo esercizio vogliamo utilizzare la ***regressione lineare*** per predire un valore, nello specifichiamo vogliamo allenare un sistema affinchè riesca a predire il peso o l'altezza di un individuo in baso all'input passato, che sia peso nel caso in cui volessimo predire l'altezza o l'altezza nel caso in cui volessimo predire il peso.

⚠: *Come già accennato nell'esercizio precedente gli algoritmi di regressione lineare sono basati sulla funzione lineare, pertanto, raccomandiamo di recuperare la lezione precedente per comprendere appieno quanto vedremo d'ora in avanti.*

## Dataset
Prima di poter prevedere eventuali informazioni il nostro modello deve essere addestrato con dei dati già noti, nello specifico andremo ad applicare quello che si definisce ***apprendimento supervisionato***, ovvero, forniremo al nostro modello un dataset nella quale è contenuta sia l'informazione di input che quella di output, successivamente il modello sarà in grado di predire nuove informazioni basandosi sugli input e gli output che gli sono stati dati durante la fase di apprendimento.
Dunque, provvediamo subito ad implementare subito un semplice dataset che contenga pochi dati dove ognuna delle singole informazioni del set si compone di due sole features, ovvero, il peso e l'altezza di un individuo umano.

|PESO (kg)|ALTEZZA (cm)|
|---------|------------|
|65|180|
|75|180|
|60|173|
|93|193|
|66|164|
|84|188|
|67|175|
|89|170|
|71|180|
|63|170|
|51|161|
|58|167|
|67|165|
|51|153|
|60|170|
|55|165|
|55|155|
|58|170|
|76|170|
|55|165|
|79|188|

In questo dataset abbiamo raccolto il peso di individui a caso, inoltre, dei vari individui abbiamo anche registrato la loro altezza, dando in pasto queste informazioni al nostro modello saremo poi in grado di stabilire in maniera del tutto *probabilistica* il peso o l'altezza di un individuo estraneo al dataset, ovviamente, il modello sarà in grado solamente se gli viene somministrata quanto meno una informazione di partenza, in questo caso il peso o l'altezza dell'individuo in analisi.

## Applichiamo la regressione lineare
In questa sezione vogliamo percorrere step by step tutte le operazioni che ci porteranno poi a comprendere al massimo il concetto di regressione lineare.

### Osservare i dati su un grafico a dispersione *(scatter)*

La prima cosa che faremo è osservare i dati del dataset su di un grafico a dispersione in modo da avere già ad occhio un idea della natura e del comportamento dei nostri dati di partenza, più che altro c'interessa sapere la relazione che vi è tra i valori $x$ e $y$, in questo caso la relazione che vi è tra $peso$ e $altezza$, dunque, provvediamo a mandare in output il grafico utilizzando le librerie *numpy* e *matplotlib*:

```python
import matplotlib.pyplot as plt
import numpy as np

peso = np.array([65, 75, 60, 93, 66, 84, 67, 89, 71, 63, 51, 58, 67, 51, 60, 55, 55, 58, 76, 55, 79])
altezza = np.array([180, 180, 173, 193, 164, 188, 175, 170, 180, 170, 161, 167, 165, 153, 170, 165, 155, 170, 170, 165, 188])

plt.scatter(peso, altezza, c="black")
plt.title("ESERCIZIO: Regressione lineare")
plt.xlabel("peso (kg)")
plt.ylabel("altezza (cm)")
plt.show()
```

L'output che otterremo eseguendo il codice di cui sopra sarà:

![scatter del dataset](image/scatter.png)

Ora non ci resta che trovare la regressione lineare ideale al nostro problema, affinchè il nostro modello sia poi in grado di prevedere valori futuri.

### Individuare la retta ideale

A questo punto non ci resta che trovare la retta ideale che mostri al meglio la relazione tra le due variabili del dataset, la retta ideale è quella che minimizza la distanza tra i punti del nostro grafico e la retta stessa, fortunatamente il modulo *scipy* di python ci viene in aiuto e ci mette a disposizione un metodo che ci ritorna tutti i parametri ideali per la regressione lineare del nostro specifico problema.

Nello specifico andremo ad utilizzare il metodo *linregress()*, tale metodo ci ritornerà più valori:

1. ***slope*** &rarr; *La pendenza della nostra retta.*
1. ***intercept*** &rarr; *Il punto di start della nostra retta.*
1. ***r*** &rarr; *La relazione che vi è tra le variabili passate in input alla funzione, questa variabile assume un valore compreso nell'intervallo* $[-1; 1]$ *dove il valore* $0.0$ *indica che non vi è alcuna relazione tra le variabili, mentre, un valore che sia* $1$ *oppure* $-1$ *indica che tra le due variabili vi è una forte relazione e che dunque vale la pena orientarsi verso la regressione lineare per la predizione di informazioni*.
1. ***p*** &rarr; *In questa variabile è contenuto il* $p_{value}$ *per il momento ci basti sapere che è semplicemente un valore che misura quanto sia probabile ottenere una relazione lineare tra i dati del dataset.*
1. ***std_err*** &rarr; *In questa variabile è contenuto il valore di* ***deviazione standard*** *ovvero quel valore che indica quanto sono distribuiti i dati, nello specifico, diremo che quanto più è basso questo valore maggiore sarà la vicinanza dei dati al valore medio, diversamente, una deviazione standard elevata indica che i dati sono lontani dal valore medio.*

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

peso = np.array([65, 75, 60, 93, 66, 84, 67, 89, 71, 63, 51, 58, 67, 51, 60, 55, 55, 58, 76, 55, 79])
altezza = np.array([180, 180, 173, 193, 164, 188, 175, 170, 180, 170, 161, 167, 165, 153, 170, 165, 155, 170, 170, 165, 188])

#calcolo i parametri ideali della mia retta lineare
slope, intercept, r, p, std_err = stats.linregress(peso, altezza)

plt.scatter(peso, altezza, c="black") 
plt.plot(peso, slope*peso+intercept, c="green") #mandiamo a schermo la retta di regressione lineare
plt.title("ESERCIZIO: Regressione lineare")
plt.xlabel("peso (kg)")
plt.ylabel("altezza (cm)")
plt.show()
```

Il codice di cui sopra mostrerà un grafico nella quale è contenuto uno scatter dei dati del dataset ed una retta di colore verde che rappresente la nostra regressione lineare, si noti come questa sia la retta con la minor distanza tra i vari punti dello scatter.

![regressione lineare](image/linear_regression.png)

⚠: *In questo caso applicare la regressione lineare ha senso, in quanto, il valore di relazione è:* $\color{#DC006C}r=0.77$ *, tale valore dimostra che vi è una forte relazione tra le due varibili poi, in quanto,* $0.77$ *è un numero molto vicino a* $1$*.*

### Predire valori con la regressione lineare

A questo punto abbiamo praticamente vinto e possiamo utilizzare la retta individuata per predire il valore di nuove altezze e di nuovi pesi, nello specifico possiamo predire:

* L' ***altezza*** di un individuo dato il suo peso con l'equazione: $\color{#DC006C}altezza=(slope*peso)+intercept$

* Il ***peso*** di un individuo data la sua altezza con la formula inversa dell'equazione precedente: $\color{#DC006C}peso=(altezza-intercpet)/slope$

Provvediamo subito a mettere in pratica quanto detto e proviamo a predire il peso di un individuo e l'altezza di un altro individuo utilizzando la regressione lineare in base ai dati del nostro dataset.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

peso = np.array([65, 75, 60, 93, 66, 84, 67, 89, 71, 63, 51, 58, 67, 51, 60, 55, 55, 58, 76, 55, 79])
altezza = np.array([180, 180, 173, 193, 164, 188, 175, 170, 180, 170, 161, 167, 165, 153, 170, 165, 155, 170, 170, 165, 188])

#calcolo i parametri ideali della mia retta lineare
slope, intercept, r, p, std_err = stats.linregress(peso, altezza)

#predizione altezza
nuovo_peso1 = 65
nuova_altezza1 = slope*nuovo_peso1+intercept

#predizione peso (formula inversa)
nuova_altezza2 = 180
nuovo_peso2 = (nuova_altezza2 - intercept)/slope

plt.scatter(peso, altezza, c="black")               #scatter del dataset
plt.plot(peso, slope*peso+intercept, c="green")     #plot della retta di regressione lineare
plt.scatter(nuovo_peso1, nuova_altezza1, c="red")   #predizione della nuova altezza
plt.scatter(nuovo_peso2, nuova_altezza2, c="blue")  #predizione del nuovo peso
plt.title("ESERCIZIO: Regressione lineare")
plt.xlabel("peso (kg)")
plt.ylabel("altezza (cm)")
plt.show()
```

L'output del codice di cui sopra produce un grafico in cui individuiamo in:

* ***nero*** &rarr; *I punti dei dati del dataset di partenza*.
* ***verde*** &rarr; *La retta della regressione lineare*.
* ***rosso*** &rarr; *La previsione circa l'altezza del nuovo individuo il cui peso è* $65kg$*.*
* ***blu*** &rarr; *La previsione circa il peso del nuovo individuo la cui altezza è* $180cm$*.*

![grafico con predizione](image/prediction.png)

## Approfondimenti: Indice di correlazione di Pearson ($r$)

Il valore della variabile `r` esprime l'*indice di correlazione di Perason*, tale valore esprime proprio l'eventuale relazione di linearità tra due variabili e come già detto nell'esercizio precedente il suo valore è un numero reale compreso nell'intervallo $[-1,1]$ dove: 

$$
\color{#DC0066}0=Nessuna\ correlazione
$$  

$$
\color{#DC0066}-1=Correlazione\ perfettamente\ negativa
$$

$$
\color{#DC0066}1=Correlazione\ perfettamente\ positiva
$$

* Correlazione ***positiva*** significa che quando la ***variabile indipendente*** aumenta, allora, anche la ***variabile dipendente*** aumenta, l'aumento sarà ***direttamente proporzionale*** nel caso in cui il valore della correlazione di *Pearson* fosse uguale a $\color{#dc0066}+1$.

* Correlazione ***negativa*** significa che quando la ***variabile indipendente*** aumenta, allora, la ***variabile dipendente*** decrementa, qualora il valore di correlazione di *Pearson* tra le due variabili fosse uguale a $\color{#dc0066}-1$, allora, sapremo che le due variabili sono ***inversamente proporzionali***.

Dunque, possiamo tenere in considerazione tale valore per valutare se applicare o meno la regressione lineare.

## Approfondimenti: Coefficiente di Determinazione ($R^2$)

Il *coefficiente di determinazione* è un valore reale compreso nell'intervallo $[0,1]$ ed esprime la variabilità dei dati e la correttezza del modello che abbiamo scelto, molto semplicemente potremmo vedere questo valore come l'espressione della bontà della linea *(regressione lineare)* o della curva *(regressione polinomiale)* che abbiamo utilizzato nel modello

$$
\color{#DC0066}0=La\ linea\ del\ modello\ è\ pessima 
$$

$$
\color{#DC0066}1=La\ linea\ del\ modello\ è\ ottimale
$$

Dunque, tale valore ci aiuta a capire quanto bene la linea o la curva si addice ai dati del dataset che abbiamo raffigurato nel grafico a dispersione *(scatter)*.

Nel nostro caso abbiamo solamente due variabili, pertanto, il nostro coefficiente di determinazione è facilmente calcolabile effettuando il quadrato dell'indice di correlazione di Pearson, questo è detto ***coefficiente di determinazione semplice*** e si indica con $\textbf r^2$.

$$
\color{#DC0066}r^2=(correlazioneDiPearson)^2
$$

⚠️: *Nel caso in cui vi fossero più variabili non potremmo più applicare ***l'indice di determinazione semplice**** ($r^2$) *, ma, necesseiteremmo del coefficiente di determinazione* ($R^2$).

In questo esempio mostriamo come ottenere il coefficiente di determinazione semplice $r^2$ nel nostro caso.

```python
import numpy as np
from scipy import stats

x = np.array([89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40])
y = np.array([21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15])

slope, intercept, r, p, std_err = stats.linregress(x,y)

#coefficiente di determinazione semplice
r2 = r**2
print(r2) 
```

## Quando la regressione lineare non è una buona scelta

La regressione lineare è applicabile quando appunto i dati contenuti nel dataset sono distribuiti in maniera lineare, dunque, quando la ***relazione*** tra i dati del dataset ottenuta dalla funzione *linregress()* è un valore vicino a $1$ o $-1$, qualora questa relazione fosse vicina allo zero, allora, non c'è relazione tra le variabili, dunque, è sconsigliato un approccio con regressione lineare.

In questo esercizio abbiamo volutamente implementato un set di dati basato su due *features (variabili)* la cui distribuzione non è per nulla lineare, i dati in questione sono:

||||||||||||||||||||||
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|$\color{#DC0066}x$|89|43|36|36|95|10|66|34|38|20|26|29|48|64|6|5|36|66|72|40|
|$\color{#DC0066}y$|21|46|3|35|67|95|53|72|58|10|26|34|90|33|38|20|56|2|47|15|

Possiamo mandare a schermo con *python* la distribuzione di questi dati e la loro rispettiva linea di regressione lineare con il codice:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
x = np.array([89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40])
y = np.array([21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15])

slope, intercept, r, p, std_err = stats.linregress(x,y)
plt.scatter(x,y)
plt.plot(x, slope*x+intercept, c="red")
plt.show()
```

Il grafico che otterremo sarà il seguente:

![dati non lineari](image/nonLinear.png)

Già ad una prima occhiata visiva ci si rende conto che la distribuzione non è affatto lineare e a confermare tale ipotesi vi è anche la retta della regressione lineare che ha una pendenza prossima allo $0$, inoltre, anche il coefficiente di correlazione di Pearson `r` è estremamente vicino allo $0$, infatti, nel nostro caso: $\color{#DC0066}r=0.013$ e questo dato è un ulteriore indice di poca linearità, dunque, a questo giro non possiamo affidarci alla regressione lineare.
