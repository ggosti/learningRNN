#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().magic(u'matplotlib notebook')
import learningRNN as lrnn
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual

splits=20
train=int(round(0.7*splits))
n0=20
mlim = -0.0010
#DEL = []
x = np.loadtxt("Run2.txt")
x=x.T
w =np.array_split(x,splits, axis=0)
N= 45
typ = 0 
thr = 0 
nP = {"N":N, "typ":typ, "thr": thr}
alpha=10.


# In[2]:


def NormMatrici (matrice):
    for i in range(len(matrice)): 
        matrice[i]= matrice[i]/(np.dot(matrice[i], matrice[i]))**(0.5) #normalizzo ogni riga di Jij        
    matrice /= (matrice.shape[0])**(0.5)
    return matrice


# In[3]:


matrix_set = []
for j in range(train):
    #medie su 20 condizioni iniziali 
    temp_matrix = []
    for i in range(n0):
        
        #parte di training
        
        T = int(4000./alpha)
        insuccesso = True       #parametro di  non raggiungimento del plateau
        while (insuccesso):
            
            sirvia= np.random.randint(1, 10000)
            X_train, Y_train = lrnn.makeTrainXYfromSeqs([ w[j] ], nP, isIndex= False)

            trained_matrix, deltas, fullDeltas, exTime, convStep, bestErrors, bestNet = lrnn.runGradientDescent(X_train, Y_train, alpha0= 0.0, alphaHat=alpha,
                                         batchFr = 1.0, passi=T, runSeed = sirvia, gdStrat="GD", k=1, netPars=nP,
                                          showGradStep= None, xi= 0.000, mexpon = -1.5, verbose=False,normalize=True)

            #parte di controllo della convergenza dei deltas
        
            b = len(deltas)
            while b>20:
                z = np.polyfit(range(b), deltas[-b:], 1)
                if z[0] > mlim:
                    print "Delta N =", j+1
                    print "Il sistema ha raggiunto l'equilibrio al passo" , len(deltas) - b
                    insuccesso = False
                    #DEL.append(deltas)
                    break 
                b -= 10
            if insuccesso:
                print "Delta N =", j+1
                print "Il sistema non raggiunge l'equilibrio"
                T*=2
        temp_matrix.append(NormMatrici(trained_matrix))
    matrix_set.append(np.mean(np.array(temp_matrix),axis=0))
        


# In[4]:


PE_mean=[]
PE_std=[]
for i in range(len(matrix_set)):
        PE0=[]
        for j in range(splits-train):
            X_test, Y_test = lrnn.makeTrainXYfromSeqs([ w[train+j] ], nP, isIndex= False)
            Y_guess = lrnn.transPy(X_test, matrix_set[i],N,typ,thr)
            PE0.append(np.sum((Y_test-Y_guess)**2/(N*len(Y_test))))
        PE_mean.append(np.mean(PE0))
        PE_std.append(np.std(PE0))


# In[5]:


fig, (ax1)= plt.subplots(1)  
ax1.set_title('media')
plt.errorbar(range(14),PE_mean, yerr=PE_std, fmt='.')



# In[33]:


matrix_set_norm = []
PEt = []
for i in range(train):
    matrix_set_norm.append(NormMatrici(matrix_set[i]))
M=np.mean(np.array(matrix_set_norm), axis=0)
STD=np.std(np.array(matrix_set_norm), axis=0)
for j in range(splits-train):
    X_test, Y_test = lrnn.makeTrainXYfromSeqs([ w[train+j] ], nP, isIndex= False)
    Y_guess = lrnn.transPy(X_test, matrix_set[i],N,typ,thr)
    PEt.append(np.sum((Y_test-Y_guess)**2/(N*len(Y_test))))
print "Media e deviazione standard dell'errore percentuale sulla media dei test delle 14 matrici:\n", np.mean(PEt), np.std(PEt)


# In[30]:


fig, (ax1, ax2)= plt.subplots(2)  
ax1.set_title('media')
c=ax1.imshow(-M/np.max(np.abs(M), axis=0),cmap='seismic')
fig.colorbar(c, ax=ax1)
ax2.set_title('deviazione standard')
c=ax2.imshow(STD, cmap= 'seismic')
fig.colorbar(c, ax=ax2)



# In[ ]:

M_test = np.load('testMatrix.npy')
STD_test = np.load('testMatrixSTD.npy')


print 'M-M_test', np.mean(M-M_test)
print 'M-M_test', np.mean(STD-STD_test)
if np.mean(M-M_test)+np.mean(STD-STD_test)<10**(-4):
    print 'Test: Pass'

# Output should be something like:
# M-M_test 5.05184e-06
# M-M_test 2.7443364e-05
# Test: Pass



plt.show(block=True)
    
