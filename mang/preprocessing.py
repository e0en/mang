import numpy as np

def center(X, mu=None):
    if mu == None:
        mu = X.mean(0)
        return (X - mu, mu)
    else:
        return X - mu

def un_center(X_c, mu):
    return X_c + mu

def whiten(X, stat=None):
    if stat == None:
        N = X.shape[0]
        (X_c, mu) = center(X)
        # tmp = X_c + 0.1*np.random.randn(*X.shape)
        std = np.sqrt((X_c*X_c).sum(0)/N)
        std += 0.1*(std < 0.1)
        X_c = (X_c/std)
        return (X_c, (mu, std))
    else:
        (mu, std) = stat
        X_c = center(X, mu)
        X_c = (X_c/std)
        return X_c

def un_whiten(X_c, stat):
    (mu, std) = stat
    return un_center(X_c*std, mu)

def pca_whiten(X, stat=None, ratio=0.99, dim=None):
    if stat == None:
        (X_c, mu) = center(X)
        (val, vec) = np.linalg.eigh(np.cov(X_c.T))
        idx = np.argsort(val)
        D = len(idx)
        eigval_sum = val.sum()
        eigval_accum = 0
        idx_cutoff = 0
        for i in idx:
            eigval_accum += val[i]
            if dim == None and eigval_accum >= (1 - ratio)*eigval_sum:
                idx_cutoff = i
                break
            elif dim != None and i == D - dim:
                idx_cutoff = i
                break
        idx = idx[idx_cutoff:]
        print len(idx)

        d = val[idx]
        P = vec[:, idx]
        '''
        (d, P) = np.linalg.eigh(np.dot(X_c.T, X_c))
        d += regularizer
        '''
        d = np.sqrt(d)
        d_inv = 1./d
        N = X.shape[0]
        W = P*d_inv
        Winv = (P*d).T
        X_c = np.dot(X_c, W)

        return (X_c, (mu, W, Winv))
    else:
        (mu, W, Winv) = stat
        (X_c, mu) = center(X)
        X_c = np.dot(X_c, W)
        return X_c

def un_pca_whiten(X_c, stat):
    (mu, W, Winv) = stat
    X = un_center(np.dot(X_c, Winv), mu)
    return X

def zca_whiten(X, stat=None, regularizer=0.0001):
    if stat == None:
        N = X.shape[0]
        X_c = (X.T - X.mean(1)).T
        (X_c, mu) = center(X_c)
        (d, P) = np.linalg.eigh(np.dot(X_c.T, X_c)/(N - 1.))
        d += regularizer
        d_inv = 1./np.sqrt(d)
        W = np.dot(P*d_inv, P.T)
        X_c = np.dot(X_c, W)
        X_c = np.clip(X_c, -4, 4)
        return (X_c, (mu, W))
    else:
        (mu, W) = stat
        X_c = (X.T - X.mean(1)).T
        X_c = center(X_c, mu)
        X_c = np.dot(X_c, W)
        X_c = np.clip(X_c, -4, 4)
        return X_c

def un_zca_whiten(X_c, stat):
    raise NotImplementedError
