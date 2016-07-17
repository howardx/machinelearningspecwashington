import numpy as np                                             # dense matrices
import sframe                                                  # see below for install instruction
from scipy.sparse import csr_matrix                            # sparse matrices
from scipy.sparse.linalg import norm                           # norms of sparse matrices
from sklearn.metrics.pairwise import pairwise_distances        # pairwise distances
from copy import copy                                          # deep copies
import matplotlib.pyplot as plt                                # plotting

import time

wiki = sframe.SFrame('people_wiki.gl/')
wiki = wiki.add_row_number()             # add row number, starting at 0

def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']   
    return csr_matrix( (data, indices, indptr), shape)

corpus = load_sparse_csr('people_wiki_tf_idf.npz')
# 59071 documents (input data points), 547979 distinct words/tfidf (features or dimensions)
print corpus.shape
map_index_to_word = sframe.SFrame('people_wiki_map_index_to_word.gl/')

def generate_random_vectors(num_vector, dim):
    return np.random.randn(dim, num_vector)
    
np.random.seed(0) # set seed=0 for consistent results

'''
Generate 16 random vectors of dimension 547979 (16 boundaries, 17 bins)
'''
random_vectors = generate_random_vectors(num_vector=16, dim=547979)
print random_vectors.shape

doc0 = corpus[0, :] # vector of tf-idf values for document 0 - input data point 0
# True if positive sign; False if negative, see if input data point 0 is in bin 0
print doc0.dot(random_vectors[:, 0]) >= 0 
# True if positive sign; False if negative, see if input data point 0 is in bin 1
print doc0.dot(random_vectors[:, 1]) >= 0 

print doc0.dot(random_vectors) >= 0 # should return an array of 16 True/False bits
print np.array(doc0.dot(random_vectors) >= 0, dtype=int) # display index bits in 0/1's

'''
compute bit indices of ALL documents (input data points), the bits with 1 assigned means that
this particular input data point falls into that bin, if it's 0 means it's not in that bin
'''
index_bits = np.array(corpus.dot(random_vectors) >= 0, dtype = int)

# By the rules of binary/decimal conversion, we just need to compute the dot product 
# between the document vector and the vector consisting of powers of 2
# [32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
powers_of_two = (1 << np.arange(15, -1, -1))
          
int_bin_num = index_bits.dot(powers_of_two)
print index_bits.shape # 59071 binary numbers (bucket number) each with 16 bits
print int_bin_num.shape # 59071 decimal numbers (inteer) converted from the above binary numbers

'''
enumerate(thing), where thing is either an iterator or a sequence, returns a iterator 
that will return (0, thing[0]), (1, thing[1]), (2, thing[2]), and so forth

bucket and bin are DIFFERENT
bin are areas segregated by "decision boundaries" - random vectors
bucket are elements in the hash table
'''
def train_lsh(data, num_vector=16, seed=None):  
    dim = data.shape[1]
    if seed is not None:
        np.random.seed(seed)
    random_vectors = generate_random_vectors(num_vector, dim)
  
    powers_of_two = 1 << np.arange(num_vector-1, -1, -1)
  
    table = {} # hash table, key - bucket number : value - document (input data) ID list
    
    ''' Partition data points into bins/buckets, returns data_size : bucket_count matrix '''
    bin_index_bits = (data.dot(random_vectors) >= 0)
  
    ''' Encode bucket number (was in bits) into integers, returns data_size : 1 (bucket #) matrix '''
    bin_indices = bin_index_bits.dot(powers_of_two)
    
    # Update `table` so that `table[i]` is the list of document ids with bin index equal to i.
    for data_index, bin_index in enumerate(bin_indices):
        if bin_index not in table:
            # If no list yet exists for this bin, assign the bin an empty list.
            table[bin_index] = [data_index]
        # Fetch the list of document ids associated with the bin and add the document id to the end.
        else:
            table[bin_index].append(data_index)

    model = {'data': data,
             'bin_index_bits': bin_index_bits,
             'bin_indices': bin_indices,
             'table': table,
             'random_vectors': random_vectors,
             'num_vector': num_vector}
    return model
    
# Checkpoint
LSH_model = train_lsh(corpus, num_vector=16, seed=143)
table = LSH_model['table']
if   0 in table and table[0]   == [39583] and \
   143 in table and table[143] == [19693, 28277, 29776, 30399]:
    print 'Passed!'
else:
    print 'Check your code.'

    
#print wiki[wiki['name'] == 'Barack Obama'] # input data point index - 35817
#print wiki[wiki['name'] == 'Joe Biden'] # input data point index - 24478
#print wiki[wiki['name']=='Wynn Normington Hugh-Jones']
    
#print LSH_model['bin_indices'][35817] # Obama's document

from itertools import combinations

num_vector = 16
search_radius = 3

'''
 3
C      (no repeat)
 16
'''
#for diff in combinations(range(num_vector), search_radius):
#    print diff
    
def search_nearby_bins(query_bin_bits, table, search_radius=2, initial_candidates=set()):
    """
    For a given query vector data point and trained LSH model, return all candidate neighbors for
    the query data point among all bins within the given search radius.
    
    Example usage
    -------------
    >>> model = train_lsh(corpus, num_vector=16, seed=143)
    >>> q = model['bin_index_bits'][0]  # vector for the first document
  
    >>> candidates = search_nearby_bins(q, model['table'])
    """
    num_vector = len(query_bin_bits)
    powers_of_two = 1 << np.arange(num_vector-1, -1, -1)
    
    # Allow the user to provide an initial set of candidates.
    candidate_set = copy(initial_candidates)
    
    for different_bits in combinations(range(num_vector), search_radius):       
        # Flip the bits (n_1,n_2,...,n_r) of the query bin to produce a new bit vector.
        ## Hint: you can iterate over a tuple like a list
        alternate_bits = copy(query_bin_bits)

        for i in different_bits:
            alternate_bits[i] = not query_bin_bits[i] # flip True/False bit
        
        # Convert the new bit vector to an integer index
        nearby_bin = alternate_bits.dot(powers_of_two)
        
        # Fetch the list of documents belonging to the bin indexed by the new bit vector.
        # Then add those documents to candidate_set
        # Make sure that the bin exists in the table!
        # Hint: update() method for sets lets you add an entire list to the set
        if nearby_bin in table:
            candidate_set.update(table[nearby_bin])
    return candidate_set
    
# CheckPoint
obama_bin_index = LSH_model['bin_index_bits'][35817] # bin index of Barack Obama
candidate_set = search_nearby_bins(obama_bin_index, LSH_model['table'], search_radius=0)
if candidate_set == set([35817, 21426, 53937, 39426, 50261]):
    print 'Passed test'
else:
    print 'Check your code'
print 'List of documents in the same bin as Obama: 35817, 21426, 53937, 39426, 50261'

# CheckPoint
candidate_set = search_nearby_bins(obama_bin_index, LSH_model['table'], 
                                   search_radius=1, initial_candidates=candidate_set)
if candidate_set == set([39426, 38155, 38412, 28444, 9757, 41631, 39207, 59050, 47773, 53937, 21426, 34547,
                         23229, 55615, 39877, 27404, 33996, 21715, 50261, 21975, 33243, 58723, 35817, 45676,
                         19699, 2804, 20347]):
    print 'Passed test'
else:
    print 'Check your code'
     
'''
Function query()
collect all NN candidates and compute their true distance to the query data point
'''
def query(vec, model, k, max_search_radius):
    data = model['data']
    table = model['table']
    random_vectors = model['random_vectors']
    num_vector = random_vectors.shape[1]
        
    # Compute bin index for the query vector, in bit representation.
    bin_index_bits = (vec.dot(random_vectors) >= 0).flatten()
    
    # Search nearby bins and collect candidates
    candidate_set = set()
    for search_radius in xrange(max_search_radius+1):
        candidate_set = search_nearby_bins(bin_index_bits, table, 
                                           search_radius, initial_candidates=candidate_set)
    
    # Sort candidates by their true distances from the query
    nearest_neighbors = sframe.SFrame({'id':candidate_set})
    candidates = data[np.array(list(candidate_set)),:]
    
    # use scikit-learn to compute pairwise cosine distance between the query data points and NN candidates
    nearest_neighbors['distance'] = pairwise_distances(candidates, vec, metric='cosine').flatten()
    
    return nearest_neighbors.topk('distance', k, reverse=True), len(candidate_set)
    
print query(corpus[35817,:], LSH_model, k=10, max_search_radius=3)

result, num_candidates_considered = query(corpus[35817,:], LSH_model, k=10, max_search_radius=3)
print result.join(wiki[['id', 'name']], on='id').sort('distance')


num_candidates_history = []
query_time_history = []
max_distance_from_query_history = []
min_distance_from_query_history = []
average_distance_from_query_history = []

for max_search_radius in xrange(20):
    start=time.time()
    # Perform LSH query using Barack Obama, with max_search_radius
    result, num_candidates = query(corpus[35817,:], LSH_model, k=10,
                                   max_search_radius=max_search_radius)
    end=time.time()
    query_time = end-start  # Measure time
    
    print 'Radius:', max_search_radius
    # Display 10 nearest neighbors, along with document ID and name
    print result.join(wiki[['id', 'name']], on='id').sort('distance')
    print np.sum(result['distance'].to_numpy())/9
    
    # Collect statistics on 10 nearest neighbors
    average_distance_from_query = result['distance'][1:].mean()
    max_distance_from_query = result['distance'][1:].max()
    min_distance_from_query = result['distance'][1:].min()
    
    num_candidates_history.append(num_candidates)
    query_time_history.append(query_time)
    average_distance_from_query_history.append(average_distance_from_query)
    max_distance_from_query_history.append(max_distance_from_query)
    min_distance_from_query_history.append(min_distance_from_query)

    
