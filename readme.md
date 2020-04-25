# SUR Projekt

[Zadanie - link](https://www.fit.vutbr.cz/study/courses/SUR/public/projekt_2019-2020/SUR_projekt2019-2020.txt)

## Inštalácia závislostí

Potreba knižnic, ktoré sú importované - Tensorflow - Keras, numpy... 
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

## Zložka models

#### model_1.h5 / model_train_all.h5
Model, ktorý bol natrénovaný na všetkých daných dátach (nie na rozdeleni v repozitary - ale na vsetkom) - verzia 1 - dáva menej targetov

#### model_2.h5 / model_train_all3.h5
Model, ktorý bol natrénovaný na rovnakých dátach ako verzia 1 - dáva viac targetov ako vezia 1
