import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from pymorphy2 import MorphAnalyzer
import re
from string import punctuation

ds = pd.read_csv('reviews.csv', sep='\t') #Загрузили датасет
ds.duplicated().sum() #повторяются в работе столько-то рецензий
#Удалим дубликаты и неуникальные значения. Дублирующиеся данные могут не вызывать ошибок, но отнимать много времени для получения результата.
df = ds.drop_duplicates(subset=['review'])
# разобьем общий датасет и сделаем в будущем классификацию на позитивный и негативный
data_pos=df[df['sentiment'] == 'positive'][['review', 'sentiment']]
data_neg = df[df['sentiment'] == 'negative'][['review', 'sentiment']]
# # Удаление стоп-слов, токенизация, лемматизация
# ### Если по-научному то сегментация и морфологический анализ
stop_words = stopwords.words("russian")
morph = MorphAnalyzer()

def get_clean_tokens(sentence):
    #преобразование в одинаковый регистр
    tokens = re.findall("\w+", sentence.lower()) #\w+ - один или больше alphanumeric-символов, то же, что и [a-zA-Z0-9]+
    tokens_no_stops = [word for word in tokens if (word not in punctuation) and (word not in stop_words)]
    tokens_no_singles = [token for token in tokens_no_stops if len(token) > 4]
    lemmatized_tokens = [morph.parse(token)[0].normal_form for token in tokens_no_singles]
    return lemmatized_tokens

#Разбиение позитивных токенов
pos_tok = [get_clean_tokens(sentence) for sentence in data_pos['review']]
pos_tok_join = [' '.join(sent) for sent in pos_tok]
#Негативных
neg_tok = [get_clean_tokens(sentence) for sentence in data_neg['review']]
neg_tok_join = [' '.join(sent) for sent in neg_tok]
# ### триграммы, облака слов и все прочее сделать из Эмоции_практика. Посмотреть еще какие-нибудь системы анализа и тоже сделать
#Объединение
data_full = pd.concat([data_pos, data_neg])
# # Векторизация текста Word2Vec
data_full.head(10)
data_full['Preprocessed_texts'] = data_full.apply(lambda row: get_clean_tokens(row['review']), axis=1)
data_full.head(10)
from gensim.models import Word2Vec
model = Word2Vec(sentences=data_full['Preprocessed_texts'],
                               min_count=5,
                               vector_size=100)
def sent_vec(sent):
    vector_size = model.wv.vector_size
    wv_res = np.zeros(vector_size)
    ctr = 1
    for w in sent:
        if w in model.wv:
            ctr += 1
            wv_res += model.wv[w]
    wv_res = wv_res/ctr
    return wv_res

data_full['vec'] = data_full['Preprocessed_texts'].apply(sent_vec)
mapping = {'negative': 0, 'positive': 1}
data_full.replace({'sentiment': mapping}, inplace=True)