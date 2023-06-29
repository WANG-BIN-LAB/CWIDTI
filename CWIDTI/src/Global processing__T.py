import scipy.io as sio
from scipy.sparse import coo_matrix
import numpy as np
import sys
import pdb

def _load_network(filename, mtrx='adj'):
    print ("### Loading [%s]..." % (filename))
    if mtrx == 'adj':
        i, j, val = np.loadtxt(filename).T
        row = np.array(i)
        col = np.array(j)
        data = np.array(val)
        col1 = col.astype(np.int32)
        row1 = row.astype(np.int32)
        # pdb.set_trace()
        A = coo_matrix((data, (row1 - 1, col1 - 1)))
        A = A.todense()
        A = np.squeeze(np.asarray(A))
        if A.shape[0]!=A.shape[1]:
            if A.shape[0]<A.shape[1]:
                cha = A.shape[1] - A.shape[0]
                new_array_row = np.array([0.0] * A.shape[1])
                for h in range(0, cha):
                    A = np.row_stack((A, new_array_row))
            if A.shape[0]>A.shape[1]:
                 cha1 = A.shape[0] - A.shape[1]
                 new_array_column = np.array([0.0] * A.shape[0])
                 for k in range(0, cha1):
                     A= np.column_stack((A, new_array_column))
        if A.min() < 0:
            print ("### Negative entries in the matrix are not allowed!")
            A[A < 0] = 0
            print ("### Matrix converted to nonnegative matrix.")
            print

        if (A.T == A).all():
            pass
        else:
            print ("### Matrix not symmetric!")
            A = A + A.T
            print ("### Matrix converted to symmetric.")
    else:
        print ("### Wrong mtrx type. Possible: {'adj', 'inc'}")
    A = A - np.diag(np.diag(A))
    A = A + np.diag(A.sum(axis=1) == 0)
    print(A.shape)
    return A


def load_networks(filenames, mtrx='adj'):

    Nets = []
    for filename in filenames:
        Nets.append(_load_network(filename, mtrx))
    return Nets
def _net_normalize(X):
    """
    Normalizing networks according to node degrees.
    """
    if X.min() < 0:
        print ("### Negative entries in the matrix are not allowed!")
        X[X < 0] = 0
        print ("### Matrix converted to nonnegative matrix.")
        print
    if (X.T == X).all():
        pass
    else:
        print ("### Matrix not symmetric.")
        X = X + X.T - np.diag(np.diag(X))
        print ("### Matrix converted to symmetric.")

    # normalizing the matrix
    deg = X.sum(axis=1).flatten()
    deg = np.divide(1., np.sqrt(deg))
    deg[np.isinf(deg)] = 0
    D = np.diag(deg)
    X = D.dot(X.dot(D))

    return X


def net_normalize(Net):
    if isinstance(Net, list):
        for i in range(len(Net)):
            Net[i] = _net_normalize(Net[i])
    else:
        Net = _net_normalize(Net)

    return Net


def _scaleSimMat(A):
    """Scale rows of similarity matrix"""
    A = A - np.diag(np.diag(A))
    A = A + np.diag(A.sum(axis=0) == 0)
    col = A.sum(axis=0)
    A = A.astype(np.float)/col[:, None]

    return A


def RWR(A, K=3, alpha=0.98):
    """Random Walk on graph"""
    A = _scaleSimMat(A)
    # Random surfing
    n = A.shape[0]
    P0 = np.eye(n, dtype=float)
    P = P0.copy()
    M = np.zeros((n, n), dtype=float)
    for i in range(0, K):
        P = alpha*np.dot(P, A) + (1. - alpha)*P0
        M = M + P

    return M


def PPMI_matrix(M):
    """ Compute Positive Pointwise Mutual Information Matrix"""
    M = _scaleSimMat(M)
    n = M.shape[0]
    col = np.asarray(M.sum(axis=0), dtype=float)
    col = col.reshape((1, n))
    row = np.asarray(M.sum(axis=1), dtype=float)
    row = row.reshape((n, 1))
    D = np.sum(col)

    np.seterr(all='ignore')
    PPMI = np.log(np.divide(D*M, np.dot(row, col)))
    PPMI[np.isnan(PPMI)] = 0
    PPMI[PPMI < 0] = 0

    return PPMI


if __name__ == "__main__":
    dataset="drugbank"
    InPath = "./data/"+dataset+ '/' +"interaction.txt"
    data = np.loadtxt(InPath)
    drugnum,targetnum= data.shape
    path_to_string_nets = './data1/' + dataset +'/'

    string_nets = [
        'target_mismatch_kernel_3_1_similarity_AllSimScores',
        'target_mismatch_kernel_3_2_similarity_AllSimScores',
        'target_mismatch_kernel_4_1_similarity_AllSimScores',
        'target_mismatch_kernel_4_2_similarity_AllSimScores',
        'target_spectrum_kernel_3_similarity_AllSimScores',
        'target_spectrum_kernel_4_similarity_AllSimScores',
                    ]

    filenames = []
    for net in string_nets:
        filenames.append(path_to_string_nets + net + '_T1.txt')
    # Load STRING networks
    Nets = load_networks(filenames)
    # Compute RWR + PPMI
    for i in range(0, len(Nets)):
        print
        print ("### Computing PPMI for network: %s" % (string_nets[i]))
        Nets[i] = RWR(Nets[i])
        Nets[i] = PPMI_matrix(Nets[i])
        if(Nets[i].shape[0]<targetnum):
           # pdb.set_trace()
           cha=targetnum-Nets[i].shape[0]
           new_array_row = np.array([0.0] * Nets[i].shape[1])
           for h in range(0,cha):
               Nets[i]=np.row_stack((Nets[i],new_array_row))
           print(Nets[i].shape[0])
           cha1=targetnum-Nets[i].shape[1]
           new_array_column = np.array([0.0] * Nets[i].shape[0])
           for k in range(0, cha1):
               Nets[i]=np.column_stack((Nets[i], new_array_column))
        print(Nets[i].shape)
        print ("### Writing output to file...")
        simmat = './data2/'+dataset+ '/' +dataset+ '_t' + str(i+1) + '_' + string_nets[i] + '_K3_alpha0.98_T2.mat'
        sio.savemat(simmat, {'Net':Nets[i]})
    aveNets = np.zeros((Nets[0].shape[0], Nets[0].shape[1]))
    sumNets = np.zeros(len(Nets))
    for i in range(0, Nets[0].shape[0]):
        for j in range(0, Nets[0].shape[1]):
            for h in range(0, len(Nets)):
                sumNets[h] = Nets[h][i][j]
            averag = np.mean(sumNets)
            aveNets[i][j] = averag
    aveNetssmat = './data2/' + dataset +'/'+ dataset + '_t_' +'aveNetsmat.mat'
    sio.savemat(aveNetssmat, {'aveNets': aveNets})
# python Tpreprocessing.py DrugBank
