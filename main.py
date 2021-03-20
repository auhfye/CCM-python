from ccm import *
import argparse
import scipy.io as sio
from sklearn.decomposition import PCA

def process():
    datapath = './losocvdata/'


    ACC = []
    for user in range(1,17):
        if user<10:
            Nuser = '0'+str(user)
        else:
            Nuser = str(user)

        print('now target subject is '+Nuser)
        f1 = sio.loadmat(datapath+'feattr_leave_'+Nuser+'_out.mat')
        f2 = sio.loadmat(datapath+'labeltr_leave_'+Nuser+'_out.mat')
        f3 = sio.loadmat(datapath+'featte_leave_'+Nuser+'_out.mat')
        f4 = sio.loadmat(datapath+'labelte_leave_'+Nuser+'_out.mat')

        Xs = f1['feattr']
        Ys = np.squeeze(f2['labelTrain'])+1
        Xt = f3['featte']
        Yt = np.squeeze(f4['labelTest'])+1



        ns = Xs.shape[1]
        X = np.hstack((Xs,Xt))
        X = X/np.linalg.norm(X,axis=0)
        pca = PCA(n_components=750,random_state=0)
        X = pca.fit_transform(X.T)
        Xs = X[:ns,:]
        Xt = X[ns:,:]



        from sklearn.svm import LinearSVC
        LR = LinearSVC(C=1)
        LR.fit(Xs,Ys)
        pred = LR.predict(Xt)
        acc = np.mean(pred==Yt)
        print(acc)

        _,_,acc1 = CCM(Xs.T,Xt.T,Ys,Yt,args)
        print('CCM acc:',acc1)
        ACC.append(acc1)
    print('final result:',ACC)
    ACC = np.array(ACC)
    print('mean acc:',np.mean(ACC))




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--d', '--subspace dimension', type=int, default='500')
    parser.add_argument('--T', '--iteration number', type=int, default='10')
    parser.add_argument('--alpha', '--alpha', type=float, default='1')
    parser.add_argument('--beta', '--beta', type=float, default='0.1')
    parser.add_argument('--gamma', '--gamma', type=float, default='0.5')
    args = parser.parse_args()

    process()


