Lemmatizer
==========================================

## Описание задачи

Создание моделей для решения задачи лемматизации.

## Формат набора данных

- Tokens_chars + pos -> lemma
```
ə m ə r ə VERB
```

## Система оценки

- Скрипт estimator.py рассчитывает метрику accuracy на основе совпадения строк на основе предикта модели на отложенном
тестовом корпусе.


## Порядок действий

- Клонирование репозитория. `git clone https://github.com/OpenNMT/OpenNMT-py`
- Установка библиотек для лемматизатора. В каталоге `/MorphemRuEval2019/lemmatizer/OpenNMT-py/` выполнить 
`pip install -r requirements.txt, python3 setup.py install`
- Подготовка данных. `python3 -m lemmatizer.lemma_data_set_former --folds 5 --language эвенкийский / селькупский`
- Создание словарей для обучения моделей. `./preprocess.sh` В скрипте необходимо прописать тип данных: token_pos, token_chars_pos, 
token_chars, sentence_tokens, sentence_chars_pos, sentence_chars.
- Обучение моделей. `./train.sh`
- Предикт на отложенном тестовом множестве: `./predict.sh` 
- Оценка на тестовом множестве: `./estimate.sh`. 


## Результаты

- Эвенк. ``
