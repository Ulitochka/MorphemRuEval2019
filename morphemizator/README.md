Morphemaizer
==========================================

## Описание задачи

Создание моделей для решения задачи морфемной сегментации.

## Формат набора данных

- Tokens_chars + pos -> lemma
```

```

## Система оценки

- Скрипт estimator.py рассчитывает метрику ...


## Порядок действий

- Клонирование репозитория. `git clone https://github.com/Ulitochka/OpenNMT-py.git`
- Установка библиотек. `pip install -r requirements.txt`
- Подготовка данных. `python3 -m morphemizator.morphem_data_set_former --folds 10`
- Создание словарей для обучения моделей. `./preprocess_morphem.sh` 
- Обучение моделей. `./train_morphem.sh`
- Предикт на отложенном тестовом множестве: `./predict_morphem.sh` 
- Оценка на тестовом множестве: `./estimate_morphem.sh`. 
