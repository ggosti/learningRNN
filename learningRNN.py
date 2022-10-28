'''
Created on Giu 20, 2018

@author: Giorgio Gosti

'''

import time
import matplotlib.pyplot as plt
import matplotlib


import sys, os
import pandas as pd
import networkx as nx
import numpy as np

#print('numpy ver', np.__version__)

dataFolder = '' #add folder were dato should be stored if different from current

def stateIndex2stateVec(m,N,typ = 1):
    """
    Returns the binary vector sigma_0 that corresponds to the index m, where m is a int between 0 and 2**N
    typ determins if the neuron activation state is defined in {-1,1} or {0,1} 
    typ=1 --> {-1,1}    typ=0 --> {0,1} 
    """
    sigma_0 = [ (1+typ)* (int(float(m)/2**i) % 2) - typ for i in range(0,N)]    # typ=1 --> [-1,1]    typ=0 --> [0,1]
    sigma_0.reverse()
    sigma_0 = np.array(sigma_0,dtype=np.uint8)
    return sigma_0


# sigma_1 --> decimale = k
def stateVec2stateIndex(sigma,N,typ = 1):
    """
    Returns the index m that corresponds to the binary vector sigma_0, where m is a int between 0 and 2**N
    typ determins if the neuron activation state is defined in {-1,1} or {0,1} 
    typ=1 --> {-1,1}    typ=0 --> {0,1} 
    """
    k=int(0)
    for i in range(0,N):
        k=k+(sigma[i]+typ)/(1+typ)*2**(N-i-1)   # typ=1 --> [-1,1]    typ=0 --> [0,1]
    return int(k)

def stateIndex2stateVecSeq(ms,N,typ = 1):
    """
    Returns a list of binary vectors sigmas that correspond to the list of indexs ms, 
    where m in ms is a int between 0 and 2**N
    typ determins if the neuron activation state is defined in {-1,1} or {0,1} 
    typ=1 --> {-1,1}    typ=0 --> {0,1} 
    """
    # type: (state index sequence, number of neurons, typ) -> state vector sequence
    sigmas = [ stateIndex2stateVec(m,N,typ) for m in ms]
    sigmas = np.array(sigmas)
    return sigmas

def stateVec2stateIndexSeq(sigmas,N,typ = 1):
    """
    Returns a list of bindexes ms that correspond to the list of binary vectors sigmas,
    where m in ms is a int between 0 and 2**N
    typ determins if the neuron activation state is defined in {-1,1} or {0,1} 
    typ=1 --> {-1,1}    typ=0 --> {0,1} 
    """
    ms = [ stateVec2stateIndex(s,N,typ) for s in sigmas]
    ms = np.array(ms)
    return ms

def rowNorm(C):
    """
    normalize with norm 1
    """
    norm = np.float32(np.asmatrix(np.sum(np.abs(C),axis=1))).T
    C = np.float32(np.asarray(C)/np.asarray(norm))
    #print 'norm ',np.sum(np.abs(C),axis=1)
    return C


def generateSmallWorldBase(N,knei,pr,rseed=None):
    """""
    generate small world network with networkx watts_strogatz_graph, 
    and add random weights

    inputs:
    N=numero nodi rete 
    k=numero di primi vicini a cui e legato ciascun nodo in una configurazione ad anello
    p=probabilita di reimpostazione di ogni connessione
    """""
    
    #genero un seme random (numero intero compreso tra 0 e 10000)
    if rseed==None: 
        rseed=np.random.randint(0,10000)
    
    print (rseed)
    

    #genero una rete casuale con seme 'seed' (connessioni casuali tra nodi)
    G=nx.watts_strogatz_graph(N,knei,pr,rseed)

    #genero matrice delle adiacenze di G
    C=np.int32(nx.adjacency_matrix(G).todense())

    #assegnazione dei pesi random ai collegamenti 
    # (per generazione matrice di pesi uso seme 'seed') (?)
    np.random.seed(rseed)
    pesi=np.float32(2*np.random.rand(N,N)-1)

    C=np.float32(np.multiply(pesi,C))

    return rseed, C



def sign(x,signFuncInZero = 1):
    y = np.sign(x)
    if signFuncInZero == 1:
        y[x == 0] = 1
    elif signFuncInZero == 0:
        y[x == 0] = -1
    return y

def transPy(sigma_path0,net1,N,typ = 1, thr = 0,signFuncInZero = 1):
    """
    transiton function. net1 is the network that generates the ttransitions
    
    If sigma_path0 is a binary vector it generates the corresponding transtions.
    
    If sigma_path0 is a list of binary vectors it generates a list with the corresponding transtions.
    
    typ determins if the neuron activation state is defined in {-1,1} or {0,1} 
    typ=1 --> {-1,1}    typ=0 --> {0,1} 
    """
    if not net1 == np.float32:
        net1 = np.float32(net1)
    if not sigma_path0 == np.float32:
        sigma_path0 = np.float32(sigma_path0)
    sigma_path1 = net1.dot(sigma_path0.T)
    #print(sigma_path1)
    #if signFuncInZero == 1:
    #    sigma_path1 [sigma_path1  == 0] = 0.000001

    #print sigma_path1
    sigma_path1 = (1-typ + sign(sigma_path1 +thr,signFuncInZero) )/(2-typ)
    #print sigma_path1
    return sigma_path1.T


def transActiv(sigma_path0, net1, N, typ=1, thr=0, signFuncInZero=1):
    """
    transiton function. net1 is the network that generates the ttransitions

    If sigma_path0 is a binary vector it generates the corresponding transtions.

    If sigma_path0 is a list of binary vectors it generates a list with the corresponding transtions.

    typ determins if the neuron activation state is defined in {-1,1} or {0,1}
    typ=1 --> {-1,1}    typ=0 --> {0,1}
    """
    if not net1 == np.float32:
        net1 = np.float32(net1)
    if not sigma_path0 == np.float32:
        sigma_path0 = np.float32(sigma_path0)
    sumx = net1.dot(sigma_path0.T)
    # print sigma_path1
    # if signFuncInZero == 1:
    #    sigma_path1 [sigma_path1  == 0] = 0.000001

    # print sigma_path1
    sigma_path1 = (1 - typ + sign(sumx + thr, signFuncInZero)) / (2 - typ)
    # print sigma_path1
    return sigma_path1.T,sumx

def getTrajPy(startm,netx,N,typ,thr,trajSize):
    path = []
    sigma_0 = stateIndex2stateVec(startm,N,typ)
    a = startm
    b = 0
    while (not b in path) and (len(path)<trajSize):
        #print 'sigma_0',sigma_0,sigma_0.shape,type(sigma_0)
        #print 'netx',netx,netx.shape,type(netx)
        sigma_1 = transPy(sigma_0, np.float32(netx), N, typ, thr)
        #print 'sigma_1',sigma_1,sigma_1.shape,type(sigma_1)
        b = stateVec2stateIndex(sigma_1,N,typ)
        #print a, '->', b
        path.append(a)
        a = b
        sigma_0 = sigma_1
    if b in path : cycle = path[path.index(b):] 
    else: cycle = []
    #print path[0],path[1]
    #print 'state bit lenght',path[0].bit_length()  
    if int(path[0]).bit_length() <= 64:
        return np.array(cycle,dtype=np.uint64),np.array(path,dtype=np.uint64)
    else: 
        return cycle,path

#get trajectories with neuron activation works on binary vector states not indexes
def getTrajActivation(sigma_0,netx,N,typ,thr,trajSize,signFuncInZero = 0):
    path = []
    pathAct = []
    #sigma_0 = stateIndex2stateVec(startm,N,typ)
    flag = True
    while (flag) and (len(path)<trajSize):
        #print 'netx',netx,netx.shape,type(netx)
        sigma_1,activity = transActiv(sigma_0, np.float32(netx), N, typ, thr,signFuncInZero)
        #print 'sigma_1',sigma_1,sigma_1.shape,type(sigma_1)
        #print a, '->', b
        path.append(sigma_0)
        pathAct.append(activity)
        #print 'sigma_1',sigma_1,sigma_1.shape,type(sigma_1)
        #print path
        for indx,s in enumerate(path):
            if (sigma_1 == s).all():
                flag = False
                loopIndx = indx
        sigma_0 = sigma_1
    if not flag :
        cycle = path[loopIndx:] #+ [sigma_0]
        cycleAct = pathAct[loopIndx:]
    else:
        cycle = []
        cycleAct = []
    #print path[0],path[1]
    #print 'state bit lenght',path[0].bit_length()
    return cycle,path,pathAct,cycleAct

#cost function
def ftr(sigma_path0, sigma_cycleStart, net1, N, typ, thr): # assume path0 is cycle start
    if not len(sigma_cycleStart) > 0:
        if sigma_cycleStart == None:
            sigmas_0 = np.array( list(sigma_path0[1:]))
            sigmas_1 = np.array(transPy(sigma_path0[:-1],net1,N,typ,thr))
    else:
        sigmas_0 = np.array( list(sigma_path0[1:])+[sigma_cycleStart])
        sigmas_1 = np.array(transPy(sigma_path0,net1,N,typ,thr))
    d = sigmas_1 - sigmas_0
    d = np.asfarray(d)
    #if typ == 0: d=0.5*d
    fc = np.sum( d**2) # /float(len(ts)) #np.sqrt(np.sum( d**2))
    return fc

#---------------------------------------------------------------------------------
# gradient descent functions
#----------------------------------------------------------------------------------

def gradientDescentStep(y,X,net0,netPars,autapse = False,signFuncInZero = 1):
    """
    gradient descent step for the linear approximation of the activation function gradient
    """
    N, typ, thr = netPars['N'],netPars['typ'],netPars['thr']
    yhat = transPy(X, net0, N, typ, thr,signFuncInZero)
    #print 'yhat',yhat
    delta = (y-yhat)
    #print delta
    #print X
    #Xp = np.delete(X, (i), axis=1)
    update = np.asmatrix(X).T.dot(np.asmatrix(delta))
    #print update
    if not autapse:
        np.fill_diagonal(update, 0) # important no autapsi!!
    if not np.isfinite(np.sum(delta**2)):
        print('net0')
        print(net0)
        print('yhat')
        print(yhat)
        print('delta')
        print(delta)
    return update,np.sum(delta**2),delta,X

def gradientDescentLogisticStep(y,X,k,net0,netPars):
    """
    gradient descent step  for the logistic approximation of the activation function
    """
    N, typ, thr = netPars['N'],netPars['typ'],netPars['thr']
    yhat = transPy(X, net0, N, typ, thr)
    #print 'yhat',yhat
    delta = (y-yhat)
    #print delta
    #print 'X'
    #print X
    gamma = np.asarray(np.asmatrix(net0).dot(np.asmatrix(X).T))
    logisticDer = k*np.exp(-k*gamma)/(1+np.exp(-k*gamma))**2
    #print 'logisticDer'
    #print logisticDer.T
    #print X*logisticDer.T
    update = np.asmatrix(X*logisticDer.T).T.dot(np.asmatrix(delta))
    #print update
    np.fill_diagonal(update, 0) # important no autapsi!!
    return update,np.sum(delta**2),delta,X,logisticDer,gamma

def gradientDescentStepDeltaRule(y,X,alpha,net0,netPars):
    """
    gradient descent step
    """
    N, typ, thr = netPars['N'],netPars['typ'],netPars['thr']
    yhat = transPy(X, net0, N, typ, thr)
    print('yhat',yhat)
    delta = (y-yhat)
    print(delta)
    print('X',X)
    dR = net0.dot(X.T)
    print('dR',dR)
    deltaRule = (alpha*np.ones((N,N),dtype=np.float32) >= np.abs(dR))
    print('deltaRule',deltaRule.T)
    #Xp = np.delete(X, (i), axis=1)
    update = np.asmatrix(X).T.dot(np.asmatrix(delta)) 
    #print 'update',update.T
    update = update*deltaRule
    np.fill_diagonal(update, 0) # important no autapsi!!
    return update,np.sum(delta**2)

def stochasticGradientDescentStep(y,X,net0,batchSize,netPars):
    N, typ, thr = netPars['N'],netPars['typ'],netPars['thr']
    draws = np.random.choice(y.shape[0],size=batchSize,replace=False)
    ybatch = y[draws,:]
    Xbatch = X[draws,:]
    yhat = transPy(Xbatch, net0, N, typ, thr)
    #yhat = np.array([yhat[:,i]]).T
    delta = (ybatch-yhat)
    #Xp = np.delete(X, (i), axis=1)
    update = Xbatch.T.dot(delta) 
    np.fill_diagonal(update, 0) # important no autapsi!!
    return update,np.sum(delta**2)

def makeTrainXYfromSeqs(seqs,nP,isIndex=True):
    listX = []
    listy = []
    for seq in seqs:
        #print len(seq)#len(list(path1)+list([cicli1[0]]))
        #print seq
        if isIndex == True: seq = stateIndex2stateVecSeq(seq,nP['N'], nP['typ'])
        #print o_sigma_path
        listX.append( np.array(seq[:-1,:]) )
        listy.append( np.array(seq[1:,:]) )
    X = np.vstack(listX)
    y = np.vstack(listy)
    return X,y

#def makeTrainXYfromSeqs(seqs,nP):
#    listX = []
#    listy = []
#    for seq in seqs:
#        #print len(seq)#len(list(path1)+list([cicli1[0]])) 
#        o_sigma_path = stateIndex2stateVecSeq(seq,nP['N'], nP['typ'])
#        #print o_sigma_path
#        listX.append( np.array(o_sigma_path[:-1,:]) )
#        listy.append( np.array(o_sigma_path[1:,:]) )
#    X = np.vstack(listX)
#    y = np.vstack(listy)
#    return X,y



    
def runGradientDescent(X,y,alpha0,N=None,alphaHat=None, nullConstr = None,batchFr = 10.0,passi=10**6,runSeed=3098,gdStrat='SGD',k=1,netPars={'typ':0.0},showGradStep=True, verbose = True, xi = 0.0 ,uniqueRow=False,lbd = 0.0,mexpon=-1.8,normalize = False,Xtest=[],ytest=[], Xval=[], yval=[], autapse=False,signFuncInZero=1):
    if N == None:
        N = netPars['N']
    assert N == X.shape[1] , 'ERROR!: makeTrainXYfromSeqs was made with trasposed input'
    np.random.seed(runSeed)
    net0 = np.float32(2*np.random.rand(N,N)-1) #np.zeros((r, w), dtype=np.float32)  # np.float32(np.random.randint(0, 2, size=(r, w)))  # np.float32(2*np.random.rand(r,w)-1)
    if not autapse: np.fill_diagonal(net0, 0)
    if normalize: net0 = rowNorm(net0)
    if not nullConstr == None: net0[nullConstr==True]=0
     
    #print 'start net0'
    #print net0
    #print np.sum(np.abs(net0),axis=1)
    m = X.shape[0]
    bestErrors = N
    if verbose: print('m ',m)
    if uniqueRow == True:
        new_array = [''.join( str(e) for e in np.uint8(row).tolist() ) for row in X]
        Xunique, index = np.unique(new_array,return_index=True)
        X = X[index,:]
        y = y[index,:]
        m = X.shape[0]
        if verbose: print('m unique ',m)
        plt.figure()
        plt.imshow(X,interpolation='nearest')
        plt.figure()
        plt.imshow(np.corrcoef(X.T),interpolation='nearest')
        plt.colorbar()
    if not gdStrat == 'SGD': batchFr = 1.0
    batchSize = m/batchFr
    if verbose: print('batchSize',batchSize,'fract',batchFr)
    if alpha0 == 0.0: alpha0 =alphaHat *( m **(-1.0) ) *( N **(mexpon) ) #alpha0 =alphaHat /  ( m *  N**2)
    if verbose: print('alphaHat',alphaHat,'alpha0',alpha0)
    
    convStep = np.inf
    deltas = []
    deltasTest = []
    deltasVal = []
    fullDeltas = []
    start = time.time()
    for j in range(passi):
        alpha = alpha0* ( (1+alpha0*lbd*j)**(-1))    
        if gdStrat == 'SGD':
            update,sumSqrDelta = stochasticGradientDescentStep(y,X,net0,batchSize,netPars)
        if gdStrat == 'GD':
            update,sumSqrDelta,delta,X = gradientDescentStep(y,X,net0,netPars,autapse,signFuncInZero)
            if not np.isfinite(sumSqrDelta):
                break
        if gdStrat == 'GDLogistic':
            update,sumSqrDelta,delta,X,logisticDer,gamma = gradientDescentLogisticStep(y,X,k,net0,netPars)
            print(update)
        if j%(passi/200) == 0:
            # if there is test compute score
            if len(Xtest)>0:
                typ, thr = netPars['typ'],netPars['thr']
                ytesthat = transPy(Xtest, net0, N, typ, thr)
                #print 'yhat',yhat
                deltaTest = (ytest-ytesthat)
                deltaTest = np.sum(deltaTest**2)
                deltasTest.append(deltaTest/ytest.shape[0])
            if len(Xval)>0:
                typ, thr = netPars['typ'],netPars['thr']
                yValhat = transPy(Xval, net0, N, typ, thr)
                #print 'yhat',yhat
                deltaVal = (yval-yValhat)
                deltaVal = np.sum(deltaVal**2)
                deltasVal.append(deltaVal/yval.shape[0])
            #print j
            #print 'yhat '
            #print yhat,yhat.shape
            #print 'sumSqrDelta ', sumSqrDelta
            fullSumSqrDelta = sumSqrDelta
            if batchFr < 1.0: updatefull,fullSumSqrDelta,delta,X = gradientDescentStep(y,X,net0,netPars,autapse,signFuncInZero)
            if showGradStep:
                if verbose: print('alpha*update ', (alpha * update.T).mean(), (alpha * update.T).std())
                f, axs = plt.subplots(2,5)
                axs[0,0].set_title('update')
                axs[0,0].set_xlabel('i')
                axs[0,0].set_ylabel('i')
                axs[0,0].imshow(update,interpolation='nearest')
                axs[0,1].set_title('delta')
                axs[0,1].set_xlabel('t')
                axs[0,1].set_ylabel('i')
                axs[0,1].imshow(delta.T,interpolation='nearest')
                axs[0,2].set_title('X')
                axs[0,2].set_xlabel('t')
                axs[0,2].set_ylabel('i')
                axs[0,2].imshow(X.T,interpolation='nearest')
                if gdStrat == 'GDLogistic':
                    axs[0,3].set_title('gamma')
                    axs[0,3].set_xlabel('t')
                    axs[0,3].set_ylabel('i')
                    axs[0,3].imshow(gamma,interpolation='nearest')
                    axs[0,4].set_title('log. Der.')
                    axs[0,4].imshow(logisticDer,interpolation='nearest')
                    axs[0,4].set_xlabel('t')
                    axs[0,4].set_ylabel('i')
                if batchFr < 1.0:
                    if verbose: print('update full', (updatefull.T).mean(), (updatefull.T).std())
                    axs[1,0].set_title('updatefull')
                    axs[1,0].set_xlabel('i')
                    axs[1,0].set_ylabel('i')
                    axs[1,0].imshow(updatefull,interpolation='nearest')
                    axs[1,1].set_title('delta')
                    axs[1,1].set_xlabel('t')
                    axs[1,1].set_ylabel('i')
                    axs[1,1].imshow(delta.T,interpolation='nearest')
                    axs[1,2].set_title('X')
                    axs[1,2].set_xlabel('t')
                    axs[1,2].set_ylabel('i')
                    axs[1,2].imshow(X.T,interpolation='nearest')
            #print 'accuracy ', (sumSqrDelta/batchSize) - (fullSumSqrDelta/y.shape[0]),' alpha ',alpha
            deltas.append(sumSqrDelta/batchSize)
            fullDeltas.append(fullSumSqrDelta/y.shape[0])
        if sumSqrDelta == 0.0:
            fullSumSqrDelta = 0
            if batchFr < 1.0:
                updatefull,fullSumSqrDelta,delta,X = gradientDescentStep(y,X,net0,netPars,autapse,signFuncInZero)
            if fullSumSqrDelta == 0:
                deltas.append(sumSqrDelta/batchSize)
                fullDeltas.append(fullSumSqrDelta/y.shape[0])
                if len(Xtest)>0:
                    deltasTest.append(deltaTest/ytest.shape[0])
                if len(Xval)>0:
                    deltasVal.append(deltaVal/yval.shape[0])
                convStep = j
                if verbose: print('final sumSqrDelta/batchSize ', sumSqrDelta/batchSize)
                break
        #print 'mean update.T', np.mean(np.abs(update.T))
        #print 'mean alpha * update.T', np.mean(np.abs(alpha * update.T))
        #print 'mean net0', np.mean(np.abs(net0))
        net0 += alpha * update.T - xi * net0
        if not nullConstr == None: net0[nullConstr==True]=0
        if sumSqrDelta/y.shape[0]<bestErrors:
            bestErrors = sumSqrDelta/y.shape[0]
            bestNet = net0.copy()
        #net0[net0>1] = 1
        #net0[net0<-1] = -1
        if normalize: net0 = rowNorm(net0)
        #print 'net0',net0.shape
    if verbose: print('final sumSqrDelta ', sumSqrDelta,not np.isfinite(sumSqrDelta))
    if verbose: print('final sumSqrDelta/batchSize ', sumSqrDelta/batchSize)
    if not np.isfinite(sumSqrDelta):
        print(deltas)
    end = time.time()
    exTime = end - start
    if verbose: print('decent time', exTime)
    if (len(Xtest)>0) and (len(Xval)==0):
        return net0,deltas, fullDeltas,exTime,convStep, bestErrors, bestNet, deltasTest
    if (len(Xtest)>0) and (len(Xval)>0):
            return net0,deltas, fullDeltas,exTime,convStep, bestErrors, bestNet, deltasTest, deltasVal
    else:
        return net0,deltas,fullDeltas,exTime,convStep, bestErrors, bestNet

def thFPostiveFNegativeTPFSignRatios(netObj,net0,thRate = 1.0):
    from skimage.filters import threshold_otsu
    th = thRate*threshold_otsu(np.abs(net0))
    thNetObj = np.abs(netObj)>0
    thNet0 = np.abs(net0)>th

    TP = thNetObj * thNet0
    TN = (1 - thNetObj) * (1 - thNet0)
    FP = (1 - thNetObj) * thNet0
    FN = thNetObj * (1 - thNet0)
    print(net0.shape,net0.shape[0]*net0.shape[1])
    FNR = 1.0 - float(np.sum(TP))/np.sum(thNetObj==1)
    FPR = 1.0 - float(np.sum(TN))/np.sum(thNetObj==0)
    #print 'FPR',FPR,'FNR',FNR
    #FNR2 = float(np.sum(FN))/np.sum(thNetObj==1)
    #FPR2 = float(np.sum(FP))/np.sum(thNetObj==0)
    #print 'FPR2',FPR2,'FNR2',FNR2
    TSign = np.sign(netObj)*np.sign(net0)*thNetObj*thNet0
    TPFSign = 1.0 - (np.sum(TSign == 1,dtype = np.float64)/np.sum(TP,dtype = np.float64))
    #print 'True Positive False sign.',TPFSign
    return thNetObj,thNet0,FPR,FNR,TPFSign,TP,TN,TSign

def qualityMeasures(netObj,net0):
    K = np.int8(netObj == 0)
    H = np.int8(netObj != 0)
    neEst = net0[netObj == 0]
    nnE = net0[netObj != 0]
    NEE = np.sum( (neEst)**2,dtype = np.float64 )/np.sum(K,dtype = np.float64)
    ESR = np.sum( (neEst)**2 ,dtype = np.float64)/np.sum( (nnE)**2 ,dtype = np.float64)
    #print 'num. wrong signs',0.5*np.sum(1 - np.sign(netObj[netObj != 0])* np.sign(nnE) ),np.sum(H,dtype = np.float64)
    #plt.figure()
    #plt.imshow(1 - np.sign(netObj[netObj != 0])* np.sign(nnE),interpolation='nearest')
    PESE = 0.5*np.sum(1 - np.sign(netObj[netObj != 0])* np.sign(nnE),dtype = np.float64)/np.sum(H,dtype = np.float64)
    return H,NEE,ESR,PESE

def plotFPFN(ax0,ax,netObj,net0,thNetObj,thNet0,TP,TN,TSign):
    import matplotlib.patches as mpatches
    plotNet(ax0[0],netObj,'Objective net',netObj.max())
    plotNet(ax0[1],net0,'Estimated net',netObj.max())
    ax[0,0].set_title('Objective net thresholded')
    ax[0,0].imshow(np.sign(netObj)*thNetObj,interpolation='nearest')
    ax[0,1].set_title('Estimated net thresholded')
    ax[0,1].imshow(np.sign(net0)*thNet0,interpolation='nearest')
    ax[1,0].set_title('Sensitivity and specificity')
    im=ax[1,0].imshow(2*TP+thNetObj-TN,interpolation='nearest')
    colors = [ im.cmap(im.norm(value)) for value in [3,1,0,-1]]
    labels = [ 'TP','FN','FP','TN']
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label=labels[i] ) for i in range(len(labels)) ] 
    ax[1,0].legend(handles=patches,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax[1,1].set_title('True sign given true positve')
    ax[1,1].imshow(TSign,interpolation='nearest')   

def plotNet(ax,net,title,vabsmax=1):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax.set_title(title)
    im1 = ax.imshow(net,interpolation='nearest',vmin=-vabsmax,vmax=+vabsmax)
    divider1 = make_axes_locatable(ax)
    cax1 = divider1.append_axes("right", size="6%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)

def drawPath(ax,net0,op_list,netPars):
    ax.plot(op_list,label='obj. net. seq.')
    o_sigma_path = stateIndex2stateVecSeq(op_list,netPars['N'], netPars['typ'])
    sigmas = trans(o_sigma_path[:-1],net0, netPars['N'], netPars['typ'], netPars['thr'])
    #print 'estimated net sigmas ', sigmas
    ms = stateVec2stateIndexSeq(sigmas, netPars['N'], netPars['typ'])
    for i,(s,t) in enumerate(zip(op_list[:-1],ms )):
        ax.plot([i,i+1],[s,t])
    ax.set_ylabel('x(t)')
    ax.set_xlabel('t')  

def plotPaths(axs1,axs2,N,netObj,net0,numTests=10,nP={}):
    fs= []
    ls = []
    cycMins = []
    fsCycle = []
    lsCycle = []
    for i in range(numTests):
        sm2 = np.random.randint(0,(2**N) -1 )
        o_cicli1,o_path1 = getTrajPy(sm2,netObj,N,0,0,100000)
        o_sigma_cicli = stateIndex2stateVecSeq(o_cicli1, N,0)
        o_sigma_path = stateIndex2stateVecSeq(o_path1, N,0)
        objseqt1 = list(o_path1)+[o_cicli1[0]]
        #print len(objseqt1)
        cicli1,path1 = getTrajPy(sm2,net0,N,0,0,100000)
        estseqt = list(path1)+[cicli1[0]]
        #print len(estseqt)
        f1 = ftr(o_sigma_path, o_sigma_cicli[0], net0, N, 0, 0, False)
        fs.append(f1)
        ls.append(len(objseqt1))
        drawPath(axs1[i,1],net0,objseqt1,nP)
        axs1[i,0].plot(objseqt1,label='obj. net. seq.')
        axs1[i,0].plot(estseqt,label='est. net. seq.')
        axs1[i,0].set_ylabel('x(t)')
        axs1[i,0].set_xlabel('t') 
        #only over cycles
        cycMin = int(o_cicli1.min())
        if not cycMin in cycMins:
            #print 'cycMin',cycMin
            i2 = len(cycMins)
            cycMins.append(cycMin)
            o_cicli1,o_path1 = getTrajPy(cycMin,netObj,N,0,0,100000)
            o_sigma_cicli = stateIndex2stateVecSeq(o_cicli1, N,0)
            o_sigma_path = stateIndex2stateVecSeq(o_path1, N,0)
            cicli1,path1 = getTrajPy(cycMin,net0,N,0,0,100000)
            f2 = ftr(o_sigma_cicli, o_sigma_cicli[0], net0, N, 0, 0, False)
            fsCycle.append(f2)
            lsCycle.append(len(list(o_cicli1)+[o_cicli1[0]]))
            #print 'len ',len(list(cicli1)+[cicli1[0]]),len(list(o_cicli1)+[o_cicli1[0]])
            #print 'obj ',list(o_cicli1)+[o_cicli1[0]]
            #print 'est ',list(cicli1)+[cicli1[0]]
            drawPath(axs2[i2,1],net0,list(o_cicli1)+[o_cicli1[0]],nP)
            axs2[i2,0].plot(list(o_path1)+[o_cicli1[0]])
            axs2[i2,0].plot(list(path1)+[cicli1[0]])
            axs2[i2,0].set_ylabel('x(t)')
            axs2[i2,0].set_xlabel('t')
    axs1[0,0].legend()
    axs1[0,1].legend()
    axs2[0,0].legend()
    axs2[0,1].legend()
    return fs,ls,cycMins,fsCycle,lsCycle

def prepTests(numTest,N,netObj):
    testSeqs = []
    tls = []
    for i in range(numTest):
        sm2 = np.random.randint(0,(2**N) -1 )
        cicli1,path1 = getTrajPy(sm2,netObj,N,0,0,100000)
        o_sigma_cicli = stateIndex2stateVecSeq(cicli1, N,0)
        o_sigma_path = stateIndex2stateVecSeq(path1, N,0)
        objseqt1 = list(path1)+[cicli1[0]]  
        testSeqs.append([o_sigma_path,o_sigma_cicli[0],objseqt1])
        tls.append(len(objseqt1))
    return testSeqs,np.array(tls)
