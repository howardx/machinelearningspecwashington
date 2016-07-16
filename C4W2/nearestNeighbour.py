import sframe                            # see below for install instruction
import matplotlib.pyplot as plt          # plotting
import numpy as np                       # dense matrices
from scipy.sparse import csr_matrix      # sparse matrices

from sklearn.neighbors import NearestNeighbors

import unpack_dict as ud

wiki = sframe.SFrame('people_wiki.gl/')
wiki = wiki.add_row_number()             # add row number, starting at 0

#print wiki.column_names()

def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']
    
    return csr_matrix( (data, indices, indptr), shape)
    
word_count = load_sparse_csr('people_wiki_word_count.npz')
map_index_to_word = sframe.SFrame('people_wiki_map_index_to_word.gl/')

print map_index_to_word.column_names()
print map_index_to_word.head()

model = NearestNeighbors(metric='euclidean', algorithm='brute')
model.fit(word_count)

# find where Obaba's article is listed in Input data set
#print wiki[wiki['name'] == 'Barack Obama']

# 1st arg: word count vector, record 35817 is where Obama's article is
distances, id = model.kneighbors(word_count[35817], n_neighbors=10) 

knn = sframe.SFrame({
                            'distance' : distances.flatten(),
                            'id' : indices.flatten()
                    })

print wiki.join(knn, on = 'id').sort('distance')[['id', 'name', 'distance']]

wiki['word_count'] = ud.unpack_dict(word_count, map_index_to_word)