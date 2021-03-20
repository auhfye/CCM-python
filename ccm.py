import numpy as np
import scipy as sp
from scipy.linalg import block_diag
from utils import searchBestIndicator
def CCM(Xs, Xt, Ys, Yt, args):
    """

    :param Xs: source data matrix, m*ns
    :param Xt: target data matrix, m*nt
    :param Ys: source label, min value = 1
    :param Yt: target label, min value = 1
    :param args: parameters
    :return:
        Zs: transformed source data, d*ns
        Zt: transformed target data, d*nt
    """

    d = args.d
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    T = args.T

    m, ns = Xs.shape[0], Xs.shape[1]
    nt = Xt.shape[1]

    X = np.hstack((Xs,Xt))
    X /= np.linalg.norm(X, axis=0)

    C = len(np.unique(Ys))
    n = ns+nt
    H = np.eye(n) - 1 / n * np.ones((n, n))
    Es = np.zeros((ns,C))
    for c in range(1,C+1):
        Es[Ys==c,c-1] = 1.0/len(Ys==c)

    E = np.vstack((Es,np.zeros((nt,C),dtype='float')))
    V = block_diag(np.zeros((ns,ns)), np.eye(nt))

    Sw = np.zeros((m,m))
    Sb = np.zeros((m,m))
    meanTotal = np.mean(Xs,axis=1)
    meanTotal = np.reshape(meanTotal,[m,1])

    for c in range(1,C+1):
        Xi = Xs[:,Ys==c]
        meanClass = np.mean(Xi,axis=1)
        meanClass = np.reshape(meanClass,[m,1])
        ni = Xi.shape[1]
        Hi = np.eye(ni)-1.0/ns*np.ones((ni,ni))


        Sw = Sw + np.linalg.multi_dot([Xi,Hi,Xi.T])
        Sb = Sb + ns*(meanClass-meanTotal)*(meanClass-meanTotal).T
    Sw = Sw / np.linalg.norm(Sw, 'fro')
    Sb = Sb / np.linalg.norm(Sb, 'fro')

    from sklearn.svm import LinearSVC
    clf = LinearSVC()
    clf.fit(Xs.T,Ys)
    raw_pred = clf.predict(Xt.T)
    print('initial acc:',np.mean(Yt==raw_pred))

    Gt = np.zeros((nt,C))
    for c in range(1,C+1):
        Gt[raw_pred==c,c-1] = 1

    G = np.vstack((np.zeros((ns,C)),Gt))
    Yt0 = np.zeros((nt,1))
    Yt0 = np.reshape(Yt0,[nt,1])

    for i in range(T):
        # Update P
        U = sp.linalg.inv(alpha*np.dot(G.T,G) + np.eye(C))
        temp1 = (E+alpha*np.dot(V,G)).T
        R = np.dot(E,E.T)-np.linalg.multi_dot([E,U,temp1]) + alpha*np.dot(V,V.T) - alpha*np.linalg.multi_dot([V,G,np.dot(U,temp1)])
        R = R / np.linalg.norm(R, 'fro')

        temp2 = np.linalg.multi_dot([X,R,X.T])
        temp3 = np.linalg.multi_dot([X,H,X.T])
        w,Ve = sp.linalg.eig(temp2 + beta*np.eye(m) + gamma*Sw, temp3+gamma*Sb)
        ind = np.argsort(w)
        P = Ve[:, ind[:d]]


        # Update F
        Z = np.dot(P.T,X)
        Z = Z-np.mean(Z,axis=0)
        Z / np.linalg.norm(X, axis=0)
        temp4 = np.dot(Z,E) + alpha*np.linalg.multi_dot([Z,V,G])
        F = np.dot(temp4,U)

        # Update G
        Zs = Z[:,:ns]
        Zt = Z[:,ns:]
        for j in range(nt):
            Yt0[j,0] = searchBestIndicator(Zt[:,j], F, C)

        Gt = np.zeros((nt, C))
        Ytt = np.squeeze(Yt0)
        for c in range(1, C + 1):
            Gt[Ytt == c, c - 1] = 1
        G = np.vstack((np.zeros((ns,C)),Gt))

        acc = np.mean(Yt==Ytt)
        print('iter {} acc:{:.6f}'.format(i,acc))

    return Zs,Zt,acc

