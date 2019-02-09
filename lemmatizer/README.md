Lemmatizer
==========================================

## Описание задачи

Создание моделей для решения задачи лемматизации.

## Формат набора данных

- Token + pos -> lemma
```
uːt͡ʃakiltin%NOUN
```

- Tokens_chars + pos -> lemma
```
ə m ə r ə VERB
```

- Tokens_chars -> lemma
```
ə m ə r ə
```

- Sentence_tokens+pos -> sentence_lemma
```
['tirga%ADV t͡ʃaŋitil%NOUN ŋinakintin%NOUN əmərən%VERB ďulatin%NOUN', 'tirga t͡ʃaŋiti ŋinakin əmə ďu']
```

- Sentence_tokens_chars+pos -> sentence_lemma
```
ď u w u n NOUN & h ə g d i ŋ ə ADJ & b i t ͡ ʃ o ː n VERB
```

- Sentence_tokens_chars -> sentence_lemma
```
n u & i d u & k a & b a l d i t ͡ ʃ a ː w & b i & m o h a d u ː & a h a
```

## Система оценки

- Скрипт estimator.py рассчитывает метрику accuracy на основе совпадения строк на основе предикта модели на отложенном
тестовом корпусе.


## Порядок действий

- Клонирование репозитория. `git clone https://github.com/Ulitochka/OpenNMT-py.git`
- Установка библиотек. `pip install -r requirements.txt`
- Подготовка данных. `python3 -m lemmatizer.lemma_data_set_former --folds 10`
- Создание словарей для обучения моделей. `./preprocess.sh` В скрипте необходимо прописать тип данных: token_pos, token_chars_pos, 
token_chars, sentence_tokens, sentence_chars_pos, sentence_chars.
- Обучение моделей. `./train.sh`
- Предикт на отложенном тестовом множестве: `./predict.sh` 
- Оценка на тестовом множестве: `./estimate.sh`. 
