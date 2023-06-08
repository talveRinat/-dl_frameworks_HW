# Отчет

## Задача:
Проведите эксперимент с обучением модели, используя такие инструменты для оптимизации ML-моделей и отслеживания экспериментов, как Optuna, Hyperopt, Pandas, Polars и DVC.

## Данные 
[Ethos-Hate_speech](https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/blob/master/ethos/ethos_data/Ethos_Dataset_Binary.csv)

## Подготовка данных 
Данные подготавливаються этим [скриптом](https://github.com/talveRinat/dl_frameworks_HW/blob/8403807784b8407aa0e183ef4c062677cc0c7f4b/preprocess.py)

## Модель
Использовал Random forest 

## Поиск параметров
Использовал Optuna для поиска параметров:
- n_estimators 
- max_depth
- min_samples_split
- min_samples_leaf

## DVC
Сохранил модель, рапорт и набор данных 

Summery report можно найти [тут](https://github.com/talveRinat/dl_frameworks_HW/blob/8403807784b8407aa0e183ef4c062677cc0c7f4b/reports)

Храниться в формате json

![Screenshot 2023-06-08 at 17.44.16.png](reports%2Fimg%2FScreenshot%202023-06-08%20at%2017.44.16.png)