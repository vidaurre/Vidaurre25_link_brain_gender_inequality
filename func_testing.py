import numpy as np
from scipy.stats import rankdata, norm


def quantile_normalize_to_gaussian(X):
    """
    Quantile normalize data so that each column has approximately
    a standard normal distribution (mean 0, var 1).
    
    Parameters
    ----------
    X : np.ndarray, shape (n, )
        Input data.
    
    Returns
    -------
    X_normalized : np.ndarray, shape (n, )
        Transformed data with Gaussian marginals.
    """
    # X = np.asarray(X, dtype=float)

    # Get ranks (1..n)
    ranks = rankdata(X[:,0], method='average')
    
    # Convert ranks to uniform quantiles in (0,1)
    quantiles = (ranks - 0.5) / len(X)
    
    # Map uniform quantiles to Gaussian
    X_normalized = norm.ppf(quantiles)
    
    return np.expand_dims(X_normalized,1)

def standard_tftest_subset(X,y,index=0,nperm=10000):

    r2 = np.zeros(nperm)
    r2_0 = np.zeros(nperm)

    X = np.copy(X)
    (Nsubj,p) = X.shape
    
    if len(y.shape)==1:
        y = np.expand_dims(y,1)

    X0 = np.copy(X[:,np.setdiff1d(np.arange(p),  np.array([index]) )])
    X0 = np.concatenate((np.ones((X0.shape[0],1)),X0),axis=1)
    X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1)

    stdy = np.sqrt( np.sum((y - np.mean(y)) ** 2 ) ) 

    for j in range(nperm):

        if j > 0: 
            pe = np.random.permutation(Nsubj)
            X = X[pe,:]
            X0 = X0[pe,:]
        
        b0 = np.linalg.inv(X0.T @ X0) @ (X0.T @ y)    
        b = np.linalg.inv(X.T @ X) @ (X.T @ y)    
        sqerr0 = np.sqrt( np.sum((y - X0 @ b0) ** 2 ) )
        sqerr = np.sqrt( np.sum((y - X @ b) ** 2 ) )

        r2[j] = 1 - sqerr / stdy
        r2_0[j] = 1 - sqerr0 / stdy
        
    d = r2 - r2_0
    pvalue = np.sum(d[0]<=d) / nperm

    return (pvalue,r2,r2_0)

def standard_tftest(X,y,nperm=10000,verbose=False):

    X = np.copy(X)
    y = np.copy(y)
    X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
    # X = sm.add_constant(X)

    (Nsubj,p) = X.shape
    tstat = np.zeros((nperm,p))
    beta = np.zeros((nperm,p))
    r2 = np.zeros(nperm)
    fstat = np.zeros(nperm)
    # y -= np.mean(y)

    df1 = p - 1
    df2 = Nsubj - p
    X2 = np.sum(X**2, axis=0)[:, np.newaxis]
    tss = np.sum((y-np.mean(y)) ** 2, axis=0)

    for j in range(nperm):

        if j > 0: X = X[np.random.permutation(Nsubj),:]

        # model = sm.OLS(y, X).fit()
        # t_stats = model.tvalues
        # f_stat = model.fvalue
        # fstat[j] = f_stat
        # tstat[j,:] = t_stats

        b = np.linalg.inv(X.T @ X) @ (X.T @ y)   
        ypred = X @ b
        r = y - ypred
        rss = np.sum(r ** 2, axis=0)  # Residual sum of squares (shape: q)
        r2[j] = 1 - (rss / tss)
        fstat[j] = (r2[j] / df1) / ((1 - r2[j]) / df2)
        r_variance = rss / df2
        se_beta = np.squeeze(np.sqrt(r_variance / X2))
        tstat[j,:] = b[:,0] / se_beta
        beta[j,:] = b[:,0]
        if verbose and (j % 1000)==0: print('Perm ' + str(j))

    pvaluef = np.sum(fstat[0]<=fstat) / nperm
    pvaluet = np.sum(np.abs(tstat[0,:])<=np.abs(tstat),axis=0) / nperm

    return (pvaluef,fstat,pvaluet,tstat,r2,beta)



def aggregated_tftest(X,y,N,nperm=10000,verbose=False):

    X = np.copy(X)
    y = np.copy(y)
    X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
    Ne = np.expand_dims(N,axis=1)

    (Nc,p) = X.shape
    Nsubj = np.sum(N)
    tstat = np.zeros((nperm,p))
    beta = np.zeros((nperm,p))
    r2 = np.zeros(nperm)
    fstat = np.zeros(nperm)
    # y -= np.mean(y)

    df1 = p - 1
    df2 = Nsubj - p
    

    X2 = np.sum(X**2 * Ne, axis=0)[:, np.newaxis]
    my = np.sum(y * N) / Nsubj
    tss = np.sum(((y-my) ** 2) * N, axis=0)

    for j in range(nperm):

        yp = np.copy(y)

        if j > 0:
            # create permutation
            yp = np.expand_dims(y,axis=0) @ np.random.dirichlet(N,Nc).T  
        else:
            yp = np.copy(y)

        # model = sm.OLS(y, X).fit()
        # t_stats = model.tvalues
        # f_stat = model.fvalue
        # fstat[j] = f_stat
        # tstat[j,:] = t_stats

        b = np.linalg.inv(X.T @ np.diag(N) @ X) @ (X.T @ np.diag(N) @ y)   
        ypred = X @ b
        r = y - ypred
        rss = np.sum((r ** 2) * N, axis=0)  # Residual sum of squares (shape: q)
        r2[j] = 1 - (rss / tss)
        fstat[j] = (r2[j] / df1) / ((1 - r2[j]) / df2)
        r_variance = rss / df2
        se_beta = np.squeeze(np.sqrt(r_variance / X2))
        tstat[j,:] = b / se_beta
        beta[j,:] = b
        if verbose and (j % 1000)==0: print('Perm ' + str(j))

    pvaluef = np.sum(fstat[0]<=fstat) / nperm
    pvaluet = np.sum(np.abs(tstat[0,:])<=np.abs(tstat),axis=0) / nperm

    return (pvaluef,fstat,pvaluet,tstat,r2,beta)



def test_correlations(X,y,nperm=10000,verbose=False):

    X = np.copy(X)
    y = np.copy(y)
    (Nsubj,p) = X.shape

    y -= np.mean(y,axis=0)
    y /= np.std(y,axis=0)
    if len(y.shape)==1: y = np.expand_dims(y,axis=1)
    X -= np.mean(X,axis=0)
    X /= np.std(X,axis=0)

    tstat = np.zeros((nperm,p))
    corr = np.zeros((nperm,p))

    for j in range(nperm):

        if j > 0: X = X[np.random.permutation(Nsubj),:]
        r = X.T @ y / (Nsubj)
        r = r[:,0]
        tstat[j,:] = r * np.sqrt( (Nsubj-2) / (1- (r**2)) )
        corr[j,:] = r

        if verbose and (j % 1000)==0: print('Perm ' + str(j))
        
    pvaluet = np.sum(np.abs(tstat[0,:])<=np.abs(tstat),axis=0) / nperm
    
    return (pvaluet,tstat,corr)



def cluster_permutation_correction(observed_stats, permuted_stats, alpha=0.05):
    """
    Cluster-based permutation correction using raw test statistics.

    Parameters
    ----------
    observed_stats : array, shape (T,)
        Test statistic (e.g. t-values) for observed data.
    permuted_stats : array, shape (n_permutations, T)
        Test statistics for each permutation under the null.
    alpha : float
        Cluster-forming threshold (uncorrected).

    Returns
    -------
    cluster_pvals : np.ndarray, shape (T,)
        Corrected p-values for each timepoint.
    clusters : list of lists
        Indices for each significant cluster.
    """

    T = observed_stats.shape[0]
    n_perms = permuted_stats.shape[0]

    # Cluster-forming threshold: use standard normal quantile or permutation-based cutoff
    thr = np.quantile(permuted_stats, 1 - alpha)  # one-sided

    def get_clusters(stat_arr):
        """Return clusters (lists of indices) above threshold."""
        above = stat_arr > thr
        clusters = []
        cluster = []
        for i, val in enumerate(above):
            if val:
                cluster.append(i)
            elif cluster:
                clusters.append(cluster)
                cluster = []
        if cluster:
            clusters.append(cluster)
        return clusters

    def cluster_stat(stat_arr, cluster):
        """Cluster statistic = sum of stats in cluster (could also use len)."""
        return np.sum(stat_arr[cluster])

    # Observed clusters
    observed_clusters = get_clusters(observed_stats)
    observed_cluster_stats = [cluster_stat(observed_stats, c) for c in observed_clusters]

    # Null distribution: max cluster stat per permutation
    null_max_stats = []
    for p in range(n_perms):
        clusters = get_clusters(permuted_stats[p])
        if clusters:
            max_stat = np.max([cluster_stat(permuted_stats[p], c) for c in clusters])
        else:
            max_stat = 0
        null_max_stats.append(max_stat)
    null_max_stats = np.array(null_max_stats)

    # Corrected p-values: for each observed cluster, compare against null
    cluster_pvals = np.ones(T)
    sig_clusters = []
    for cluster, obs_stat in zip(observed_clusters, observed_cluster_stats):
        p_corr = np.mean(null_max_stats >= obs_stat)
        cluster_pvals[cluster] = p_corr
        if p_corr < 0.05:
            sig_clusters.append(cluster)

    return cluster_pvals, sig_clusters
