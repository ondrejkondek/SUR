# SUR Projekt

[Zadanie - link](https://www.fit.vutbr.cz/study/courses/SUR/public/projekt_2019-2020/SUR_projekt2019-2020.txt)

## Inštalácia závislostí

Potrebuješ všetky knižnice, ktoré sú importované - napr. Keras, Tensorflow, ... 
Maximálne python3.7 - na novšom to bohužiaľ nejde


## Jednotlivé moduly

#### data_augm.py
Vytvorí požadované dáta - augmentáciou - Scale, grain, rotation 

#### remove_wavs.py
Odstráni wav súbory z danej zložky

#### load_data.py
Načíta trénovacie dáta, upraví do požadovaného formátu a uloží do pickle súborov

#### prepare_testdata.py
Načíta testovacie dáta, upraví do požadovaného formátu a uloží do pickle súborov

#### train_model.py
Vytvorí a nátrenuje model pomocou CNN a uloží ho ako "model.h5"

#### test_model.py
Pomocou modelu predikuje výsledky daných dát
