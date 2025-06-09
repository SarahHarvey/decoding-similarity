import numpy as np
import numpy.typing as npt
import dsutils

class LinearCKA:
    """
    Compute linear Centered Kernel Alignment.
    """

    def __init__(self, center_columns=True):
        self.center_columns = center_columns

    def fit(self, X, Y):
        pass

    def score(self, X, Y):
        """
        Parameters
        ----------
        X : ndarray
            (num_samples x num_neurons) matrix of activations.
        Y : ndarray
            (num_samples x num_neurons) matrix of activations.

        Returns
        -------
        dist : float
            Similarity score between X and Y.
        """

        Nx = np.shape(X)[1]
        Ny = np.shape(Y)[1]

        if self.center_columns:
            X = X - np.mean(X, axis=0)
            Y = Y - np.mean(Y, axis=0)

        # Compute angular distance between (sample x sample) covariance matrices.
        KX = (1/Nx)*X@np.transpose(X)
        KY = (1/Ny)*Y@np.transpose(Y)    
        cka_score = np.trace(KX @ KY)/(np.linalg.norm(KX,ord='fro')*np.linalg.norm(KY,ord='fro'))

        return cka_score


class LinearDecodingSimilarity:
    """
    Compute linear decoding similarity.

    Parameters a and b dictate the regularization assumed when performing linear decoding.  Defaults to a = 0 and b = 1.  See https://arxiv.org/abs/2411.08197 for more details. 
    """

    def __init__(self, center_columns=True, a=0, b=1):
        self.center_columns = center_columns
        self.a = a
        self.b = b

    def fit(self, X, Y):
        pass
        
    
    def score(self, X, Y, Cz):
        """
        Parameters
        ----------
        X : ndarray
            (num_samples x num_neurons) matrix of activations.
        Y : ndarray
            (num_samples x num_neurons) matrix of activations.
        Cz: ndarray
            (num_samples x num_samples) decoding task covariance
        a : float
        b : float

        Returns
        -------
        dist : float
            Similiarity score between X and Y with respect to decoding task with covariance Cz.
        """

        if self.center_columns:
            X = X - np.mean(X, axis=0)
            Y = Y - np.mean(Y, axis=0)

        M = np.shape(X)[0]
        Nx = np.shape(X)[1]
        Ny = np.shape(Y)[1]

        if M != np.shape(Y)[0]:
            raise ValueError(
                "Both representations need to have the same number of samples (inputs).")

        if (Nx > M and self.b == 0) or (Ny > M and self.b == 0) :
            raise ValueError(
                "At least one of the Neuron x Neuron covariance matrices is rank deficient since #neurons > #inputs, so b cannot be 0.")

        CX = (1/M)*(X.T)@X
        CY = (1/M)*(Y.T)@Y

        GX = self.a*CX + self.b*np.identity(len(CX))
        GY = self.a*CY + self.b*np.identity(len(CY))

        # Compute inner product between (sample x sample) normalized covariance matrices.
        KX = (1/Nx)*(1/M)*X@np.linalg.inv(GX)@(X.T) # This could be made faster
        KY = (1/Ny)*(1/M)*Y@np.linalg.inv(GY)@(Y.T) 

        ds = np.trace(KX@Cz@KY)/np.sqrt((np.trace(KX@Cz@KX)*np.trace(KY@Cz@KY)) )
        
        return ds



class LinearDecodingSimilarityMulti:
    """
    Compute linear decoding similarity.

    Parameters a and b dictate the regularization assumed when performing linear decoding.  Defaults to a = 0 and b = 1.  See https://arxiv.org/abs/2411.08197 for more details. 
    """

    def __init__(self, center_columns=True, a=0, b=1, whiten = False, alpha=1):
        self.center_columns = center_columns
        self.a = a
        self.b = b
        self.whiten = whiten
        self.alpha = alpha

    def cache(self, reps, returnGinv = False):
        
        cached = []
        
        for k in range(len(reps)):
            if not isinstance(reps[k],np.ndarray):
                X = reps[k].numpy()
            else:
                X = reps[k]
                
            if self.center_columns:
                X = X - np.mean(X, axis=0)

            # X = X/np.linalg.norm(X, ord='fro')
                
            M = np.shape(X)[0]
            Nx = np.shape(X)[1]

            if self.whiten:
                # Cx = (X.T)@X
                # if np.linalg.matrix_rank(Cx) < Cx.shape[0]:
                #     print("matrix is singular")
                #     cached.append([])
                # else:
                # XTXinv = np.linalg.pinv((X.T)@X)
                # evalues, evectors = np.linalg.eigh(XTXinv)
                # XTXinvsqrt = evectors * np.sqrt(evalues) @ evectors.T 
                # GXinv = self.alpha**2 * np.identity(Nx) + self.alpha * (1-self.alpha)*XTXinvsqrt + (1-self.alpha)**2 * (XTXinv)
                # GX = np.linalg.inv(GXinv)

                XZ, Z = dsutils.whiten(X, alpha=0.5)

                if returnGinv == True:
                    GXinv = Z@Z
                    cached.append(GXinv)
                else: 
                    KX = (1/Nx)*(1/M)*XZ@(XZ.T) # This could be made faster
                    cached.append(KX)

            else: 
                if self.a == 0 and self.b == 1:
                    if returnGinv == True:
                        GXinv = np.identity(Nx)
                        cached.append(GXinv)
                    else:
                        # KX = (1/Nx)*(1/M)*X@(X.T) # This could be made faster
                        KX = (1/M)*X@(X.T) # This could be made faster

                        cached.append(KX)   
                else:
                    CX = (1/M)*(X.T)@X
                    GX = self.a*CX + self.b*np.identity(len(CX))
                    if returnGinv == True:
                        cached.append(np.linalg.inv(GX))
                    else: 
                        # KX = (1/Nx)*(1/M)*X@np.linalg.inv(GX)@(X.T) # This could be made faster
                        KX = (1/M)*X@np.linalg.inv(GX)@(X.T) # This could be made faster

                        cached.append(KX)
            
            print(k)
        print("Done caching.")
            
        return cached
    
    def score(self, reps, Cz, cached):  # TO DO:  make caching optional
        """
        Parameters
        ----------
        X : ndarray
            (num_samples x num_neurons) matrix of activations.
        Y : ndarray
            (num_samples x num_neurons) matrix of activations.
        Cz: ndarray
            (num_samples x num_samples) decoding task covariance
        a : float
        b : float

        Returns
        -------
        dist : float
            Similiarity score between X and Y with respect to decoding task with covariance Cz.
        """
        ds = np.zeros((len(reps), len(reps)))
        for i in range(len(reps)):
            for j in range(len(reps)):
                if j < i:
                  # if True:  

                    
                    # if self.center_columns:
                    #     X = reps[i] - np.mean(reps[i], axis=0)
                    #     Y = reps[j] - np.mean(reps[j], axis=0)
                    # else:
                    X = reps[i]
                    Y = reps[j]

                    KX = cached[i]
                    KY = cached[j]

                    M = np.shape(X)[0]
                    Nx = np.shape(X)[1]
                    Ny = np.shape(Y)[1]

                    if M != np.shape(Y)[0]:
                        raise ValueError(
                            "Both representations need to have the same number of samples (inputs).")

                    if (Nx > M and self.b == 0) or (Ny > M and self.b == 0) :
                        raise ValueError(
                            "At least one of the Neuron x Neuron covariance matrices is rank deficient since #neurons > #inputs, so b cannot be 0.")


                    # Compute inner product between (sample x sample) normalized covariance matrices.
            
                    ds[i,j] = np.trace(KX@Cz@KY)/np.sqrt((np.trace(KX@Cz@KX)*np.trace(KY@Cz@KY)) )  

            print(i)

        ds = ds + ds.T
        return ds

    def distance(self, reps, Cz, cached):  # TO DO:  make caching optional, add parallel option
        """
        Parameters
        ----------
        X : ndarray
            (num_samples x num_neurons) matrix of activations.
        Y : ndarray
            (num_samples x num_neurons) matrix of activations.
        Cz: ndarray
            (num_samples x num_samples) decoding task covariance
        a : float
        b : float

        Returns
        -------
        dist : float
            Similiarity score between X and Y with respect to decoding task with covariance Cz.
        """
        nmodels = len(cached)
        dd = np.zeros((nmodels, nmodels))
        for i in range(nmodels):
            for j in range(nmodels):
                if j < i:

                    # X = reps[i]
                    # Y = reps[j]


                    KX = cached[i]
                    KY = cached[j]

                    # M = np.shape(X)[0]
                    # Nx = np.shape(X)[1]
                    # Ny = np.shape(Y)[1]

                    # if M != np.shape(Y)[0]:
                    #     raise ValueError(
                    #         "Both representations need to have the same number of samples (inputs).")

                    # if (Nx > M and self.b == 0) or (Ny > M and self.b == 0) :
                    #     raise ValueError(
                    #         "At least one of the Neuron x Neuron covariance matrices is rank deficient since #neurons > #inputs, so b cannot be 0.")

            
                    # dd[i,j] = np.trace(KX@Cz@KX)/(np.linalg.norm(KX, ord='fro')**2) + np.trace(KY@Cz@KY)/(np.linalg.norm(KY, ord='fro')**2) - 2*np.trace(KX@Cz@KY)/((np.linalg.norm(KX, ord='fro')) * (np.linalg.norm(KY, ord='fro')) )

                    dd[i,j] = np.sqrt(np.trace(KX@Cz@KX) + np.trace(KY@Cz@KY) - 2*np.trace(KX@Cz@KY))

                    # dd[i,j] = np.sqrt(2 - 2*(np.trace(KX@Cz@KY) / (np.sqrt(np.trace(KX@Cz@KX) * np.trace(KY@Cz@KY)))))
 
            print(i)

        dd = dd + dd.T
        return dd
    
    def testscore(self, reps_train, reps_test, Cz, cached):  # TO DO:  make caching optional
        """
        Parameters
        ----------
        X : ndarray
            (num_samples x num_neurons) matrix of activations.
        Y : ndarray
            (num_samples x num_neurons) matrix of activations.
        Cz: ndarray
            (num_samples x num_samples) decoding task covariance
        a : float
        b : float

        Returns
        -------
        dist : float
            Similiarity score between X and Y with respect to decoding task with covariance Cz.
        """
        ds = np.zeros((len(reps_train), len(reps_train)))
        for i in range(len(reps_train)):
            for j in range(len(reps_train)):
                if j < i:

                    X = reps_train[i]
                    Y = reps_train[j]

                    if self.center_columns:
                        X = X - np.mean(X, axis=0)
                        Y = Y - np.mean(Y, axis=0)

                    Xtest = reps_test[i]
                    Ytest = reps_test[j]

                    if self.center_columns:
                        Xtest = Xtest - np.mean(Xtest, axis=0)
                        Ytest = Ytest - np.mean(Ytest, axis=0)

                    GinvX = cached[i]
                    GinvY = cached[j]

                    M = np.shape(X)[0]
                    Nx = np.shape(X)[1]
                    Ny = np.shape(Y)[1]

                    KX = (1/Nx)*(1/M)*Xtest @ GinvX @ (X.T)
                    KY = (1/Ny)*(1/M)*Ytest @ GinvY @ (Y.T)


                    # if M != np.shape(Y)[0]:
                    #     raise ValueError(
                    #         "Both representations need to have the same number of samples (inputs).")

                    if (Nx > M and self.b == 0) or (Ny > M and self.b == 0) :
                        raise ValueError(
                            "At least one of the Neuron x Neuron covariance matrices is rank deficient since #neurons > #inputs, so b cannot be 0.")


                    # Compute inner product between (sample x sample) normalized covariance matrices.
            
                    ds[i,j] = np.trace(Cz @ (KY.T) @ KX)/np.sqrt((np.trace(KX.T @ KX @ Cz)*np.trace(KY.T @ KY @Cz)) )  

            print(i)

        ds = ds + ds.T
        return ds



def sq_bures_metric(X: npt.NDArray, Y: npt.NDArray, normalize: str) -> float:
    """Compute the square of the Bures distance between two
     positive-definite matrices.
    """
    Nx = np.shape(X)[1]
    Ny = np.shape(Y)[1]

    A = (1/Nx)*X@(X.T) 
    B = (1/Ny)*Y@(Y.T) 

    if normalize == "fro":
        A = A / np.linalg.norm(A, ord="fro")
        B = B / np.linalg.norm(B, ord="fro")
    elif normalize == "trace1":
        A = A / np.linalg.trace(A)
        B = B / np.linalg.trace(B)
    
    va, ua = np.linalg.eigh(A)
    vb, ub = np.linalg.eigh(B)
    sva = np.sqrt(np.maximum(va, 0.0))
    svb = np.sqrt(np.maximum(vb, 0.0))
    return (
        np.sum(va) + np.sum(vb) - 2 * np.sum(
            np.linalg.svd(
                (sva[:, None] * ua.T) @ (ub * svb[None, :]),
                compute_uv=False
            )
        )
    )

def sq_proc_dist(X: npt.NDArray, Y: npt.NDArray) -> float:
    """Compute the square of the Procrustes distance between two
     positive-definite matrices.
    """
    Nx = np.shape(X)[1]
    Ny = np.shape(Y)[1]

    A = (1/Nx)*(X.T)@X 
    B = (1/Ny)*(Y.T)@Y 
    # A = (X.T)@X 
    # B = (Y.T)@Y    
    
    va, ua = np.linalg.eigh(A)
    vb, ub = np.linalg.eigh(B)
    sva = np.sqrt(np.maximum(va, 0.0))
    svb = np.sqrt(np.maximum(vb, 0.0))
    return (
        np.sum(va) + np.sum(vb) - 2 * (1/Nx) * (1/Ny) * np.sum(
            np.linalg.svd(
                 (X.T)@Y ,
                compute_uv=False
            )
        )
    )


def kernel_euc_dist(X: npt.NDArray, Y: npt.NDArray, normalize: str) -> float:
    """Compute the square of the Procrustes distance between two
     positive-definite matrices.
    """
    Nx = np.shape(X)[1]
    Ny = np.shape(Y)[1]

    A = (1/Nx)*X@(X.T) 
    B = (1/Ny)*Y@(Y.T) 
    # A = (X.T)@X 
    # B = (Y.T)@Y  

    if normalize == "fro":
        A = A / np.linalg.norm(A, ord="fro")
        B = B / np.linalg.norm(B, ord="fro")
        
    
    return (
        np.linalg.norm(A - B, ord="fro")
    )





