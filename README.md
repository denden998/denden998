- 👋 Hi, I’m @denden998
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...

<!---
denden998/denden998 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->



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
