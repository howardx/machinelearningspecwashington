import sframe
from scipy.sparse import csr_matrix
import numpy as np


def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']
    
    return csr_matrix( (data, indices, indptr), shape)
    

def unpack_dict(sparse_matrix, sframe_lookup):
    # 'category' column from sframe sorted on 'index', converted to python list
    # 'category' is essentially the word list from the entire corpus
    lookup_list = list(sframe_lookup.sort('index')['category'])
    
    nonZeroValues = sparse_matrix.data
    nonZeroRows = sparse_matrix.indices # CSR format
    nonZeroCols = sparse_matrix.indptr  # CSR format
    
    num_row = sparse_matrix.shape[0] # number of rows

    return [
            { k : v for k, v in zip (
                # for each word_id in sparse matrix, use it as index to access list of 'category' (lookup_list)
                [lookup_list[word_id] for word_id in nonZeroRows[ nonZeroCols[i] : nonZeroCols[i+1] ]], # key
                
                # get all data (cell values) in sparse matrix, convert to a python list
                nonZeroValues[ nonZeroCols[i] : nonZeroCols[i+1] ].tolist()                             # value
            )}
            for i in xrange(num_row)
           ]


# Tests -
def unitTest():
    wordCount = load_sparse_csr('people_wiki_word_count.npz')
    #map_index_to_word = sframe.SFrame('people_wiki_map_index_to_word.gl/')

    # wiki['word_count'] = ud.unpack_dict(wordCount, map_index_to_word)
    print type(wordCount.data)
    print wordCount.data.shape # non-zero cell value in sparse matrix
    
    print type(wordCount.indices)
    print wordCount.indices.shape # non-zero row index in sparse matrix
    
    print type(wordCount.indptr)
    print wordCount.indptr.shape # non-zero column/cell index in sparse matrix
    
    i = 0
    non_0_col_indices_given_row_0_1 = wordCount.indices[wordCount.indptr[i] : wordCount.indptr[i+1]]
    
    i = 0
    non_0_col_vals_given_row_0_1 = wordCount.data[ wordCount.indptr[i] : wordCount.indptr[i+1]]
    
    print len(non_0_col_indices_given_row_0_1) == len(non_0_col_vals_given_row_0_1)
    
    i = 1
    non_0_col_indices_given_row_1_3 = wordCount.indices[wordCount.indptr[i] : wordCount.indptr[i+2]]
    i = 0
    non_0_col_indices_given_row_0_3 = wordCount.indices[wordCount.indptr[i] : wordCount.indptr[i+3]]
    
    '''
    in slice of non-zero rows (row vectors) 0 to 3 (end exclusive slice), 
    the number of non-zero values/cells/columns equals to the sum of
    
    -number of nonZero values/cells/columns in slice of nonZero rows (row vectors) 0 to 1 (end exclusive slice)
    AND
    -number of nonZero values/cells/columns in slice of nonZero rows (row vectors) 1 to 3 (end exclusive slice)
    '''
    print len(non_0_col_indices_given_row_0_3) == ( len(non_0_col_indices_given_row_0_1) +
                                                    len(non_0_col_indices_given_row_1_3) )


#unitTest()