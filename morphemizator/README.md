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

- Клонирование репозитория. `git clone https://github.com/Ulitochka/MorphemRuEval2019`
- Установка библиотек. `pip install -r requirements.txt`
- Подготовка данных. `python3 -m morphemizator.morphem_data_set_former --folds 10`
- Обучение моделей. `python3 nn_experiment.py  --hidden_states 256 --epochs 10 --batch_size 32`
