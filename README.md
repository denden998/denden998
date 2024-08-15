import pandas as pd
import numpy as np
from gensim import corpora, models, similarities
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel

df = pd.read_csv('../data/aita_pp.csv')

lemmas_split = [lemma.split() for lemma in df['pp_text']]

dictionary = corpora.Dictionary(lemmas_split)

dictionary.filter_extremes(no_below=10, no_above=0.4)
dictionary.compactify()

dictionary.save('../data/aita_lda.dict')


corpus = [dictionary.doc2bow(text) for text in lemmas_split]

lda_model = LdaModel(corpus=corpus,     
                     id2word=dictionary, 
                     num_topics=10,      
                     random_state=100,   
                     passes=2, 
                     per_word_topics=False)
