import sframe                            # see below for install instruction
import matplotlib.pyplot as plt          # plotting
import numpy as np                       # dense matrices
from scipy.sparse import csr_matrix      # sparse matrices

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances as eud

import unpack_dict as ud

# wiki's original columns - [URI, name, text] - name is the theme/topic of the text/article
wiki = sframe.SFrame('people_wiki.gl/')

wiki = wiki.add_row_number() # add row number (id) starting at 0, now wiki's columns are - [id, URI, name, text]

def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']
    
    return csr_matrix( (data, indices, indptr), shape)

# preprocessed bag of word sparse matrix - word_count
word_count = load_sparse_csr('people_wiki_word_count.npz')
map_index_to_word = sframe.SFrame('people_wiki_map_index_to_word.gl/')

print map_index_to_word.head()
print map_index_to_word.shape # number of rows = number of distinct words in corpus
print word_count.shape        # number of cols = number of distinct words in corpus

model = NearestNeighbors(metric='euclidean', algorithm='brute')
model.fit(word_count)

# find where Obaba's article is listed in Input data set
#print wiki[wiki['name'] == 'Barack Obama']

# 1st arg: word count vector, record 35817 is where Obama's article is
distances, id = model.kneighbors(word_count[35817], n_neighbors=10) 

knn = sframe.SFrame({
                            'distance' : distances.flatten(),
                            'id' : id.flatten()
                    })

print wiki.join(knn, on = 'id').sort('distance')[['id', 'name', 'distance']]

wiki['word_count'] = ud.unpack_dict(word_count, map_index_to_word) # add 'word_count' as a new column to wiki

def top_words(name):
    """
    Get a table of the most frequent words in the given person's wikipedia page.
    """
    row = wiki[wiki['name'] == name]
    word_count_table = row[['word_count']].stack('word_count', new_column_name=['word','count'])
    return word_count_table.sort('count', ascending = False)
    
    
obama_words = top_words('Barack Obama')
print obama_words

barrio_words = top_words('Francisco Barrio')
print barrio_words

combined_words = obama_words.join(barrio_words, on='word')
combined_words = combined_words.rename({'count':'Obama', 'count.1':'Barrio'})
combined_words.sort('Obama', ascending=False)

common_words = set(["the", "in", "and", "of", "to"])

def has_top_words(word_count_vector):
    # extract the keys of word_count_vector and convert it to a set
    unique_words = set(word_count_vector.keys())
    # return True if common_words is a subset of unique_words
    # return False otherwise
    return common_words.issubset(unique_words) # .issubset() method requires 2 python sets

wiki['has_top_words'] = wiki['word_count'].apply(has_top_words)

# use has_top_words column to answer the quiz question
print sum(wiki['has_top_words'])

print 'Output from your function:', has_top_words(wiki[32]['word_count'])
print 'Correct output: True'
print 'Also check the length of unique_words. It should be 167'

print 'Output from your function:', has_top_words(wiki[33]['word_count'])
print 'Correct output: False'
print 'Also check the length of unique_words. It should be 188'


tf_idf = load_sparse_csr('people_wiki_tf_idf.npz')
wiki['tf_idf'] = ud.unpack_dict(tf_idf, map_index_to_word)

model_tf_idf = NearestNeighbors(metric='euclidean', algorithm='brute')
model_tf_idf.fit(tf_idf)

distances, indices = model_tf_idf.kneighbors(tf_idf[35817], n_neighbors=10)

neighbors = sframe.SFrame({'distance':distances.flatten(), 'id':indices.flatten()})
print wiki.join(neighbors, on='id').sort('distance')[['id', 'name', 'distance']]

def top_words_tf_idf(name):
    row = wiki[wiki['name'] == name]
    word_count_table = row[['tf_idf']].stack('tf_idf', new_column_name=['word','weight'])
    return word_count_table.sort('weight', ascending=False)

obama_tfidf = top_words_tf_idf('Barack Obama')
print obama_tfidf

schiliro_tfidf = top_words_tf_idf('Phil Schiliro')
print schiliro_tfidf

combined_tfidf = obama_tfidf.join(schiliro_tfidf, on='word')
combined_tfidf = combined_tfidf.rename({'weight':'Obama', 'weight.1':'Barrio'})
combined_tfidf.sort('Obama', ascending=False)

common_words = set(["obama", "law", "democratic", "senate", "presidential"])

wiki['has_top_words_tfidf'] = wiki['word_count'].apply(has_top_words)

# use has_top_words column to answer the quiz question
print sum(wiki['has_top_words_tfidf'])

obama_input_rowID = wiki[wiki['name'] == 'Barack Obama']['id']
biden_input_rowID = wiki[wiki['name'] == 'Joe Biden']['id']
print eud(tf_idf[obama_input_rowID], tf_idf[biden_input_rowID])