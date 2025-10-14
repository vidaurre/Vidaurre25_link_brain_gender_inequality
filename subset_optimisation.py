import numpy as np


def correlation_quick(v1,v2,N):
    if (len(v1.shape)>1) or (len(v2.shape)>1): raise Exception('incorrect shape')
    sN = np.sum(N)
    mv1 = np.sum(v1 * N) / sN
    mv2 = np.sum(v2 * N) / sN
    v1 = np.copy(v1)
    v2 = np.copy(v2)
    v1 -= mv1
    v2 -= mv2
    cov = np.sum(v1 * v2 * N)
    std1 = np.sqrt(np.sum(v1 * v1 * N))
    std2 = np.sqrt(np.sum(v2 * v2 * N))
    if (std1==0) or (std2==0):
        return 1.0
    return cov / (std1 * std2)


def correlation_slow(v1,v2):
    N = len(v1)
    mv1 = np.sum(v1) / N
    mv2 = np.sum(v2) / N
    wv1 = v1 - mv1
    wv2 = v2 - mv2
    cov = np.sum(wv1 * wv2)
    std1 = np.sqrt(np.sum(wv1 * wv1))
    std2 = np.sqrt(np.sum(wv2 * wv2))
    return cov / (std1 * std2)


def count_to_indices(c,indices_countries):
    # c is how many we want from each country
    # indices_countries: which positions belong to each country
    indices = np.zeros(len(indices_countries)).astype(bool)
    for j in range(len(c)):
        indices_j = np.where(indices_countries == j)[0]
        N = len(indices_j)
        if N < c[j]:
            raise Exception('wrong')
        if N > c[j]:
            jj = indices_j[np.random.choice(N,c[j],replace=False)]
        else:
            jj = indices_j
        indices[jj] = True
    return indices

def random_indices(N,k):
    Nc = len(N)
    indices = np.round(np.random.uniform(size=Nc) * N).astype(int)
    indices[indices>N] = N[indices>N] # can't have more than there are
    s_indices = np.sum(indices)        
    if s_indices > k:
        # indices = np.round(indices * (k/s_indices) ).astype(int)
        j = 0
        while j < (s_indices - k):
            i = np.random.choice(Nc)
            if indices[i]>0: 
                indices[i] -= 1
                j += 1   
    elif s_indices < k:
        j = 0
        while j < (k - s_indices ):
            i = np.random.choice(Nc)
            if indices[i]<N[i]: 
                indices[i] += 1
                j += 1        
    return indices


def heuristic_max_corr_quick(matrix, N, k, 
                             max_iter=10000, cooling_rate = 0.9999, repetitions = 10, verbose=False):
    """
    Select a subset of k rows from 'matrix' (with 2 columns) such that the Pearson correlation
    between the two columns in the subset is maximized.
    
    Args:
      matrix (np.ndarray): Input matrix of shape (Ncountries, 2)
      N : vector of shape (Ncountries): how many people per country
      k (int): Number of rows to select.
      max_iter (int): Maximum number of iterations for the local search.
    
    Returns:
      best_subset (np.ndarray): matrix of shape (Ncountries, 2), with how many people per country are chosen
      best_corr (float): The Pearson correlation of the subset.
    """
    Nc = matrix.shape[0]
    Nall = np.sum(N)
    if k > np.sum(Nall):
        raise ValueError("k must be less than or equal to the number of rows in matrix.")

    best_corr_final = 1.0

    for r in range(repetitions):

        # Start with a random subset of indices.
        best_indices = random_indices(N,k)
        best_corr = 0 
        for j in range(matrix.shape[1]-1):
            best_corr += abs(correlation_quick(matrix[:,0],matrix[:,j+1],best_indices))

        current_indices = best_indices.copy()
        current_corr = best_corr

        temp = 1  
        
        # Local search: try to improve the correlation by swapping one row in and one row out.
        for it in range(max_iter):
            # Randomly choose one index from the current subset and one from outside it.
            jj = 0
            while True:
                if np.sum(current_indices!=0) == 1:
                    idx_out = np.where(current_indices!=0)[0][0]
                    break                
                idx_out = np.random.choice(Nc)
                if (current_indices[idx_out]>0): 
                    break
                jj += 1
                if jj>1000: raise Exception('cannot sample (1)')
            jj = 0
            while True:
                idx_in = np.random.choice(Nc)
                if (idx_out != idx_in) and (current_indices[idx_in] < N[idx_in]): 
                    break
                jj += 1
                if jj>1000: raise Exception('cannot sample (2)')
            max_in = N[idx_in] - current_indices[idx_in]
            max_out = current_indices[idx_out]
            max_ch = min(max_out,max_in)
            N_ch = max(round(np.random.choice(max_ch) * temp),1)
            temp *= cooling_rate
            
            # Create a candidate subset by swapping the chosen rows.
            candidate_indices = current_indices.copy()
            candidate_indices[idx_in] = candidate_indices[idx_in] + N_ch
            candidate_indices[idx_out] = candidate_indices[idx_out] - N_ch
            candidate_corr = 0
            for j in range(matrix.shape[1]-1):
                candidate_corr += abs(correlation_quick(matrix[:,0],matrix[:,j+1],candidate_indices))

            # Compute the acceptance probability
            delta = abs(current_corr) - abs(candidate_corr)
            if delta > 0:
                accept = True  # Always accept an improvement
            else:
                probability = np.exp(delta / temp)  # Acceptance probability for worse solutions
                accept = np.random.random() < probability

            # Accept or reject the new solution
            if accept:
                current_indices = candidate_indices
                current_corr = candidate_corr

                # Update best solution found
                if abs(candidate_corr) < abs(best_corr):
                    best_indices = candidate_indices
                    best_corr = current_corr

            if verbose and (it % 5000) == 0:
                print(f"Iteration {it}, Temp: {temp:.6f}, Best Corr: {best_corr:.6f}")

        if verbose: print(f"Repetition {r}, Best Corr: {best_corr:.6f}")

        if abs(best_corr) < abs(best_corr_final):
            best_indices_final = best_indices
            best_corr_final = best_corr

    # print(str(best_corr))

    return best_indices_final, best_corr_final

