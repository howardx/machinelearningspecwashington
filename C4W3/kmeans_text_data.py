import sframe                                                  # see below for install instruction
import matplotlib.pyplot as plt                                # plotting
import numpy as np                                             # dense matrices
from scipy.sparse import csr_matrix                            # sparse matrices
from sklearn.preprocessing import normalize                    # normalizing vectors
from sklearn.metrics import pairwise_distances                 # pairwise distances
import sys      
import os

def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']

    return csr_matrix( (data, indices, indptr), shape)

wiki = sframe.SFrame('people_wiki.gl/')
tf_idf = load_sparse_csr('people_wiki_tf_idf.npz')
map_index_to_word = sframe.SFrame('people_wiki_map_index_to_word.gl/')

tf_idf = normalize(tf_idf) # normalize tfidf, mitigate text length effect

def get_initial_centroids(data, k, seed=None):
    '''RANDOMLY choose k data points as initial centroids'''
    if seed is not None: # useful for obtaining consistent results
        np.random.seed(seed)
    n = data.shape[0] # number of data points
        
    # Pick K indices from range [0, N).
    rand_indices = np.random.randint(0, n, k)
    
    # Keep centroids as dense format, as many entries will be nonzero due to averaging.
    # As long as at least one document in a cluster contains a word,
    # it will carry a nonzero weight in the TF-IDF vector of the centroid.
    centroids = data[rand_indices,:].toarray()
    return centroids
    
    
# Get the TF-IDF vectors for documents 100 through 102.
queries = tf_idf[100:102,:]

# Compute pairwise distances from every data point to each query vector.
test_dist = pairwise_distances(tf_idf, queries, metric='euclidean')
'''
distances[i,j] is assigned the distance between the ith row of X
(i.e., X[i,:]) and the jth row of Y (i.e., Y[j,:]) 
'''
centroids = tf_idf[0:3,:] # K = 3, initialize 3 centroids with the first 3 rows of tf_idf
distances = pairwise_distances(tf_idf, centroids, metric='euclidean')
'''
Centroids     |  c1     |  c2     | c3  ......
Observations
o1            | dist1   |  dist2  |   dist3
o2            | dist4   |  dist5  |   dist6
o3       ......
......
'''
dist_430th_observation = distances[430, 1]

# Checkpoint - Test cell
'''
numpy.allclose(a, b, rtol = 1e-05, atol = 1e-08, equal_nan = False)

Returns True if two arrays are element-wise equal within a tolerance.
The tolerance values are positive, typically very small numbers
'''
if np.allclose(dist_430th_observation, pairwise_distances(tf_idf[430,:], tf_idf[1,:])):
    print('Pass')
else:
    print('Check your code again')

'''
numpy.apply_along_axis(func1d, axis, arr, *args, **kwargs)
Apply a function to 1-D slices along the given axis.

axis = 0 -> column by column apply
axis = 1 -> row by row apply
'''
closest_cluster = np.apply_along_axis(np.argmin, axis = 1, arr = distances)

# Checkpoint - Test cell
reference = [list(row).index(min(row)) for row in distances]
if np.allclose(closest_cluster, reference):
    print('Pass')
else:
    print('Check your code again')
    
if len(closest_cluster)==59071 and \
   np.array_equal(np.bincount(closest_cluster), np.array([23061, 10086, 25924])):
    print('Pass') # count number of data points for each cluster
else:
    print('Check your code again.')
    

def assign_clusters(data, centroids):
    # Compute distances between each data point and the set of centroids:
    distances_from_centroids = pairwise_distances(data, centroids, metric='euclidean')
    
    # Compute cluster assignments for each data point:
    cluster_assignment = np.apply_along_axis(np.argmin, axis = 1, arr = distances_from_centroids)
    return cluster_assignment


# Checkpoint
if np.allclose(assign_clusters(tf_idf[0:100:10], tf_idf[0:8:2]), np.array([0, 1, 1, 0, 0, 2, 0, 2, 2, 1])):
    print('Pass')
else:
    print('Check your code again.')
    

'''  ADJUST CLUSTERS CENTROIDS AFTER INCLUDING NEW OBSERVATIONS '''

data = np.array([[1., 2., 0.],
                 [0., 0., 0.],
                 [2., 2., 0.]])
centroids = np.array([[0.5, 0.5, 0.],
                      [0., -0.5, 0.]])
                      
cluster_assignment = assign_clusters(data, centroids)
print cluster_assignment   # prints [0 1 0]
print data[cluster_assignment==1] # prints the first data point in input data - because it's in cluster 1

print data[cluster_assignment==0]
print data[cluster_assignment==2] # empty set
print data[cluster_assignment==3] # empty set (invalid input)

print data[cluster_assignment==0].mean(axis=0) # find new centroid by calculating means of axis


def revise_centroids(data, k, cluster_assignment):
    new_centroids = []
    for i in xrange(k):
        # Select all data points that belong to cluster i. K is the number of clusters
        member_data_points_in_cluster_i = data[cluster_assignment == i]
        # Compute the mean of the data points
        centroid = member_data_points_in_cluster_i.mean(axis = 0)
        '''
        numpy.matrix.A1
        Return self as a flattened (1-D) ndarray.
        '''
        centroid = centroid.A1
        new_centroids.append(centroid)
    new_centroids = np.array(new_centroids)
    return new_centroids
    
# CHECKPOINT
result = revise_centroids(tf_idf[0:100:10], 3, np.array([0, 1, 1, 0, 0, 2, 0, 2, 2, 1]))
if np.allclose(result[0], np.mean(tf_idf[[0,30,40,60]].toarray(), axis=0)) and \
   np.allclose(result[1], np.mean(tf_idf[[10,20,90]].toarray(), axis=0))   and \
   np.allclose(result[2], np.mean(tf_idf[[50,70,80]].toarray(), axis=0)):
    print('Pass')
else:
    print('Check your code')


''' ASSESSING K-MEANS CLUSTERING CONVERGENCE '''

'''
How can we tell if the k-means algorithm is converging? 
We can look at the cluster assignments and see if they stabilize over time. 
In fact, we'll be running the algorithm until the cluster assignments stop changing at all

The smaller the distances, the more homogeneous the clusters are.
In other words, we'd like to have "tight" clusters

distances - sum of squared distances data points and their associated cluster centers
'''
def compute_heterogeneity(data, k, centroids, cluster_assignment):
    heterogeneity = 0.0
    for i in xrange(k):
        # Select all data points that belong to cluster i
        member_data_points = data[cluster_assignment==i, :]
        
        if member_data_points.shape[0] > 0: # check if i-th cluster is non-empty
            # Compute distances from centroid to data points
            distances = pairwise_distances(member_data_points, [centroids[i]], metric='euclidean')
            squared_distances = distances**2
            heterogeneity += np.sum(squared_distances)   
    return heterogeneity


''' PUTTING K-MEANS ALGORITHM ALTOGETHER '''

def kmeans(data, k, initial_centroids, maxiter, record_heterogeneity = None, verbose = False):
    '''This function runs k-means on given data and initial set of centroids.
       maxiter: maximum number of iterations to run.
       record_heterogeneity: (optional) a list, to store the history of heterogeneity as function of iterations
                             if None, do not store the history.
       verbose: if True, print how many data points changed their cluster labels in each iteration'''
    centroids = initial_centroids[:] # centroids' initialization are passed in
    prev_cluster_assignment = None
    
    for itr in xrange(maxiter):        
        if verbose:
            print(itr)
        # Step 1. Make cluster assignments using nearest centroids
        cluster_assignment = assign_clusters(data, centroids)
            
        # Step 2. Compute a new centroid for each of the k clusters, 
        # averaging all data points assigned to that cluster.
        centroids = revise_centroids(data, k, cluster_assignment)
            
        # Check for convergence: if none of the assignments changed, stop
        # if once no data points switch cluster, then no data points will never swtich again.
        if prev_cluster_assignment is not None and \
          (prev_cluster_assignment == cluster_assignment).all():
            break
        
        # Print number/count of new assignments (data points swtiched cluster)
        if prev_cluster_assignment is not None:
            num_changed = np.sum(prev_cluster_assignment != cluster_assignment)
            if verbose:
                print('    {0:5d} elements changed their cluster assignment.'.format(num_changed))   
        
        # Record heterogeneity convergence metric
        if record_heterogeneity is not None:
            score = compute_heterogeneity(data, k, centroids, cluster_assignment)
            record_heterogeneity.append(score)
        
        prev_cluster_assignment = cluster_assignment[:]        
    return centroids, cluster_assignment
    
def plot_heterogeneity(heterogeneity, k):
    plt.figure(figsize=(7,4))
    plt.plot(heterogeneity, linewidth=4)
    plt.xlabel('# Iterations')
    plt.ylabel('Heterogeneity')
    plt.title('Heterogeneity of clustering over time, K={0:d}'.format(k))
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

# QUIZ
k = 3
heterogeneity = []
initial_centroids = get_initial_centroids(tf_idf, k, seed=0)
centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                       record_heterogeneity=heterogeneity, verbose=True)
plot_heterogeneity(heterogeneity, k)


k = 10
heterogeneity = {}
import time
start = time.time()
for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
    initial_centroids = get_initial_centroids(tf_idf, k, seed)
    centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                           record_heterogeneity=None, verbose=False)
    # To save time, compute heterogeneity only once in the end
    heterogeneity[seed] = compute_heterogeneity(tf_idf, k, centroids, cluster_assignment)
    print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity[seed]))
    cluster_population = []
    for i in xrange(k):
        cluster_population.append(sum(cluster_assignment == i))
    print "the number of observations within the largest cluster is: " + str(max(cluster_population))
    sys.stdout.flush()
end = time.time()
print(end-start)


''' VISUALIZE CLUSTERING RESULTS '''

def plot_k_vs_heterogeneity(k_values, heterogeneity_values):
    plt.figure(figsize=(7,4))
    plt.plot(k_values, heterogeneity_values, linewidth=4)
    plt.xlabel('K')
    plt.ylabel('Heterogeneity')
    plt.title('K vs. Heterogeneity')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

filename = 'kmeans-arrays.npz'

heterogeneity_values = []
k_list = [2, 10, 25, 50, 100] # multiple runs of K-means, with different K values

if os.path.exists(filename):
    arrays = np.load(filename)
    centroids = {}
    cluster_assignment = {} # {K value : cluster assignment list}
    for k in k_list:
        print k
        sys.stdout.flush()
        centroids[k] = arrays['centroids_{0:d}'.format(k)]
        cluster_assignment[k] = arrays['cluster_assignment_{0:d}'.format(k)]
        score = compute_heterogeneity(tf_idf, k, centroids[k], cluster_assignment[k])
        heterogeneity_values.append(score)
    
    plot_k_vs_heterogeneity(k_list, heterogeneity_values)

else:
    print('File not found. Skipping.')


def visualize_document_clusters(wiki, tf_idf, centroids, cluster_assignment, k,
                                map_index_to_word, display_content=True):
    '''wiki: original dataframe
       tf_idf: data matrix, sparse matrix format
       map_index_to_word: SFrame specifying the mapping betweeen words and column indices
       display_content: if True, display 8 nearest neighbors of each centroid'''
    
    print('==========================================================')

    # Visualize each cluster c
    for c in xrange(k):
        # Cluster heading
        print('Cluster {0:d}    '.format(c)),
        # Print top 5 words with largest TF-IDF weights in the cluster
        idx = centroids[c].argsort()[::-1]
        for i in xrange(5): # Print each word along with the TF-IDF weight
            print('{0:s}:{1:.3f}'.format(map_index_to_word['category'][idx[i]], centroids[c,idx[i]])),
        print('')
        
        if display_content:
            # Compute distances from the centroid to all data points in the cluster,
            # and compute nearest neighbors of the centroids within the cluster.
            distances = pairwise_distances(tf_idf, [centroids[c]], metric='euclidean').flatten()
            distances[cluster_assignment!=c] = float('inf') # remove non-members from consideration
            nearest_neighbors = distances.argsort()
            # For 8 nearest neighbors, print the title as well as first 180 characters of text.
            # Wrap the text at 80-character mark.
            for i in xrange(8):
                text = ' '.join(wiki[nearest_neighbors[i]]['text'].split(None, 25)[0:25])
                print('\n* {0:50s} {1:.5f}\n  {2:s}\n  {3:s}'.format(wiki[nearest_neighbors[i]]['name'],
                    distances[nearest_neighbors[i]], text[:90], text[90:180] if len(text) > 90 else ''))
        print('==========================================================')
        
k = 10
visualize_document_clusters(wiki, tf_idf, centroids[k], cluster_assignment[k], k, map_index_to_word)

'''
numpy.bincount(x, weights=None, minlength=None)
Count number of occurrences of each value in array of non-negative ints.

similar to histogram, but using each unique value as a bar
'''
np.bincount(cluster_assignment[10])


k=100
visualize_document_clusters(wiki, tf_idf, centroids[k], cluster_assignment[k], k,
                            map_index_to_word, display_content=False)
                            
'''  Another sign of too large K is having lots of small clusters '''
print sum(np.bincount(cluster_assignment[100]) < 236)