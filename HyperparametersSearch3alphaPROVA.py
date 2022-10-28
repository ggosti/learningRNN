# -*- coding: utf-8 -*-
# %%
#import sys
#sys.path.append('/content/gdrive/My Drive/LaboratorioMEG2021/learningRNN/')

import learningRNN as lrnn
import sogliatura_fun as soglia
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import time
import json

# %%
#importo e sogliatura del file MEG
sub='mllnna85_0203_pow_alpha_no_gcs_45_nodes'
FILENAME = './datiMEG/' +sub+'.csv'
Run1=pd.read_csv(FILENAME, header=None)
#print(Run1)
Sub1=Run1.values
fig, axs = plt.subplots(Sub1.shape[0],1,constrained_layout=True) 
fig.set_size_inches(8,8)
fig.suptitle('MEG BLP signals', fontsize=30)
fig.supxlabel('time', fontsize=20)
fig.supylabel('sources', fontsize=20)
for i in range(Sub1.shape[0]): #Sub1.shape[0]
    axs[i].plot(Sub1[i,:])
    axs[i].set_ylabel(i+1, fontsize=10)
    axs[i].spines[['top','right','bottom','left']].set_visible(False)
    axs[i].set_yticklabels([])
    axs[i].set_xticklabels([])
    
    #axs[i].axis('off')
optimal_thr, x = soglia.optimal_th(Sub1, plot=True)
#print(f'optimal threshold = {optimal_thr}')

# %%
plt.figure(figsize=(20,3))
plt.imshow(Sub1[:,:], aspect='auto')
plt.colorbar()

# %%
# x = np.loadtxt("Run1.txt") #<--segnale già sogliato
# print(x.shape)
# print(x)

x=x.T
print(x.shape)
print(x.dtype,x.shape)

# %%
plt.figure(figsize=(20,3))
plt.imshow(x.T[:, 0:1000],interpolation='nearest', aspect='auto')
plt.colorbar()

plt.figure(figsize=(20,40))
plt.imshow(x.T[:, 2000:3000],interpolation='nearest')

plt.figure(figsize=(20,5))
plt.plot(x.T[1, 2000:3000])


# %%
g=30

# %%
N= 45
typ = 0 
thr = 0 
nP = {"N":N, "typ":typ, "thr": thr}
Xstart, Ystart = lrnn.makeTrainXYfromSeqs([x], nP, isIndex= False)

Xp, X_val, Yp, Y_val = train_test_split(Xstart, Ystart, test_size=0.10)

# %%
print(Ystart)

# %%
wX =np.array_split(Xp,g, axis=0)
wY =np.array_split(Yp,g, axis=0)


print(len(wX),len(wY))
print(wX[0].shape,wY[0].shape)

# %%
alpha=np.array([0.1, 1, 10]) 
T1 = np.array([20000, 4000, 800])  #n° di step per i casi con norm=True oppure con norm=False e xi diverso da 0
T2= np.array([20000, 20000, 4000]) #n° di step per i casi con norm=False e xi=0
xi_GD=np.array([0])
n_alpha=len(alpha) 
n_xi=len(xi_GD)
norm=True


# %%
timestr = time.strftime("%Y%m%d-%H%M%S")

#ad ogni run creo una cartella all'interno di hyperparameters search che contiene i file generati durante il run
#(matrici e figure)
directory_name='sub'+sub+'N'+str(N)+'g'+str(g)+'norm'+str(norm)+'timestr'+timestr+'WithValidationACTfun1'
parent_dir="." 

path = os.path.join(parent_dir, directory_name) #path associato alla cartella creata
os.mkdir(path)

print("Directory '%s' created" %directory_name)

# %%
t_mats = np.zeros((g, n_alpha, n_xi,N,N))      #n°segmenti x n° alpha x n°xi x n°neuroni x n°neuroni x 
t_dels = np.zeros((g,n_alpha,n_xi,200))        #n°segmenti x  n°alpha x n°xi x 200
t_dels_tests = np.zeros((g,n_alpha, n_xi,200))
t_dels_vals = np.zeros((g,n_alpha, n_xi,200))
t_errors=np.zeros((g,n_alpha, n_xi))            #n°segmenti x n°alpha x n°xi
alphaPro=np.zeros((g,n_alpha,n_xi))
xiPro=np.zeros((g,n_alpha,n_xi))
TPro=np.zeros((g,n_alpha,n_xi))
gPro=np.zeros((g,n_alpha,n_xi))

#t_step=np.zeros((g,n_alpha))

for j in range(n_alpha):
  for k in range(n_xi):
    if norm==False and k==0:  
      T=T2
    else: 
      T=T1

    for i in range(g):
      nP = {"N":N, "typ":typ, "thr": thr}
      X, Y = wX[i],wY[i] 
      X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)
      #print(X_train)
      print('X_train shape',X_train.shape)
      #print(X_test)
      print('X_test shape',X_test.shape)
      trained_matrix, deltas, fullDeltas, exTime, convStep, bestErrors, bestNet,deltas_test, deltas_val =\
      lrnn.runGradientDescent(X_train, Y_train, alpha0= 0.0, alphaHat=alpha[j],
                             batchFr = 1.0, passi=T[j], runSeed = 3098, gdStrat="GD", k=1, netPars=nP,
                              showGradStep= False, xi= xi_GD[k], mexpon = -1.5, normalize = norm,
                                Xtest=X_test, ytest=Y_test,Xval=X_val, yval=Y_val, signFuncInZero=0)
    
      t_mats[i,j,k,:,:]=trained_matrix 
      t_dels[i,j,k,:]=deltas           
      t_dels_tests[i,j,k,:]=deltas_test
      t_dels_vals[i,j,k,:]=deltas_val
      t_errors[i,j,k]=bestErrors    
      alphaPro[i,j,k]=alpha[j]
      xiPro[i,j,k]=xi_GD[k]
      TPro[i,j,k]=T[j]
      gPro[i,j,k]=i

# %%
print(t_mats.shape)
#print(t_mats[:,:,:,0])

# %%
#voglio salvare t_mats, t_dels, t_dels_tests, t_errors nella cartella generata in precedenza
with open(path+'/matrici_iperparametri (t_mat,t_dels,t_dels_test,t_dels_vals,t_errors,alphaPro,xiPro,Tpro,gPro)', 'wb') as f:
    np.save(f, np.array(t_mats))
    np.save(f,np.array(t_dels))
    np.save(f,np.array(t_dels_tests))
    np.save(f,np.array(t_dels_vals))
    np.save(f,np.array(t_errors))
    np.save(f,np.array(alphaPro))
    np.save(f,np.array(xiPro))
    np.save(f,np.array(TPro))
    np.save(f,np.array(gPro))
f.close

#per aprire le matrici in un altro script:
#path = "C:/Users/martina/Desktop/IIT/hyperparameters search/N45g10timestr20220512-200145"
# with open(path+'/matrici_iperparametri', 'rb') as f:
#     t_mats=np.load(f)
#     t_dels=np.load(f)
#     t_dels_tests=np.load(f)
#     t_dels_vals=np.load(f)
#     t_errors=np.load(f)
#     alphaPro=np.load(f)
#     xiPro=np.load(f)
#     TPro=np.load(f)
#     gPro=np.load(f)
# f.close

# %%
val_min=np.min([np.min(t_dels[:,:,:,-1]), np.min(t_dels_tests[:,:,:,-1]), np.min(t_dels_vals[:,:,:,-1]), np.min(t_errors[:,:,:])])
val_max=np.max([np.max(t_dels[:,:,:,-1]), np.max(t_dels_tests[:,:,:,-1]), np.max(t_dels_vals[:,:,:,-1]), np.max(t_errors[:,:,:])])
print(val_min, val_max)

# %%
# --------- ISTOGRAMMI -------------

fig, axs = plt.subplots(n_alpha, 1, constrained_layout=True)
fig.set_size_inches(20, 22)
#fig.suptitle('histograms', fontsize=30, y=1.03)
    
# # clear subplots
# for ax in axs:
#     ax.remove()

# add subfigure per subplot
gridspec = axs[0].get_subplotspec().get_gridspec()
subfigs = [fig.add_subfigure(gs) for gs in gridspec]

for j, subfig in enumerate(subfigs):
    if norm==False and k==0:  
        T=T2
    else: 
        T=T1

    subfig.suptitle('alpha='+str(alpha[j])+', T='+str(T[j]), fontsize=40)

    # create 1x3 subplots per subfig
    axs = subfig.subplots(1, 3)
    axs[0].hist(np.array(t_dels)[:,j,:,-1],bins=40,range=(val_min,val_max)) 
    axs[0].set(title='deltasTrain')
    axs[1].hist(np.array(t_dels_tests)[:,j,:,-1], bins=40,range=(val_min,val_max),label='deltasTest')
    axs[1].hist(np.array(t_dels_vals)[:,j,:,-1], bins=40,range=(val_min,val_max),label='deltasVal')   
    axs[1].set(title='deltasTest and deltasVal')
    axs[1].legend()
    axs[2].hist(t_errors[:,j,:], bins=40,range=(val_min,val_max))          
    axs[2].set(title='bestErrors')

fig.savefig(path+'/histograms')
    

# %%
#ISTOGRAMMI (Vecchia versione)

# for j in range(n_alpha):
#     fig, axs = plt.subplots(n_xi, 3)
#     fig.set_size_inches(20, 22)
#     fig.suptitle('alpha = '+ str(alpha[j]) +', T ='+str(T[j]), fontsize=20)
#     for k in range(n_xi):
        
#         axs[k,0].hist(np.array(t_dels)[:,-1,j,k],bins=40,range=(val_min,val_max)) 
#         axs[k,0].set(title='deltasTrain (xi='+str(xi_GD[k])+')')
#         axs[k,1].hist(np.array(t_dels_tests)[:,-1,j,k], bins=40,range=(val_min,val_max))
#         axs[k,1].hist(np.array(t_dels_vals)[:,-1,j,k], bins=40,range=(val_min,val_max))   
#         axs[k,1].set(title='deltasTest and deltasVal (xi='+str(xi_GD[k])+')')
#         axs[k,2].hist(t_errors[:,j,k], bins=40,range=(val_min,val_max))          
#         axs[k,2].set(title='bestErrors (xi='+str(xi_GD[k])+')')

    #fig.savefig(path+'/histograms (alpha='+str(alpha[j])+')')

# %%
#----------- ERRORI ---------------

# create 2x1 subplots
fig, axs = plt.subplots(1, n_alpha, constrained_layout=True)
fig.set_size_inches(25,10)
#fig.suptitle('Figure title')

for j in range(n_alpha):
    if norm==False and k==0:  
        T=T2
    else: 
        T=T1
            
    for deltasTest,deltasTrain, deltasVal in zip(t_dels_tests[:,j,:,:],t_dels[:,j,:,:], t_dels_vals[:,j,:,:]):
        
        axs[j].plot(range(0,T[j],int(T[j]/200)),deltasTest.T,'-b',label='test') #range(0,T[j],int(T[j]/200)),
        axs[j].plot(range(0,T[j],int(T[j]/200)),deltasTrain.T,'-r',label='train')
        axs[j].plot(range(0,T[j],int(T[j]/200)), deltasVal.T,'-g',label='validation')
        axs[j].legend(['test','train','validation'])
        axs[j].set(title='alpha ='+str(alpha[j])+', T ='+str(T[j]))
        

plt.savefig(path+'/deltas Train and Tests')

# %%
#------------ MATRICI ------------- ?????????????????

#faccio la media e lo standard error sui segmenti --> ottengo due matrici medie (una per ogni alpha)
# mean_mats=np.zeros((N,N,n_alpha,n_xi))
# se_mats=np.zeros((N,N,n_alpha,n_xi))
# for j in range(n_alpha):
#     for k in range(n_xi):
#         mean_mats[:,:,j,k]=np.mean(t_mats[:,:,:,j,k],axis=2)
#         se_mats[:,:,j,k]=t_mats[:,:,:,j,k].std(axis=2)/np.sqrt(g)

# titoli=['MEDIA', 'STANDARD ERROR']
# for j in range(n_alpha):
    
    # create 2x1 subplots
    # fig, big_axs= plt.subplots(2,1)
    # fig.set_size_inches(30,13)
    # fig.suptitle('alpha = '+ str(alpha[j]) , fontsize=30, y=1.03)

    # # clear subplots
    # for ax in axs:
    #     ax.remove()

    # # add subfigure per subplot
    # gridspec = axs[0].get_subplotspec().get_gridspec()
    # subfigs = [fig.add_subfigure(gs) for gs in gridspec]

    # for row, subfig in enumerate(subfigs):
    #     subfig.suptitle(titoli[row], fontsize=25)

    #     # create 1x4 subplots per subfig
    #     axs = subfig.subplots(1,n_xi)
     
    #     for k, ax in enumerate(axs):
    #         vabs_max=np.max(np.abs(mean_mats[:,:,j,k]))
    #         im1=axs[0,k].imshow(mean_mats[:,:,j,k],cmap='seismic',vmin=-vabs_max, vmax=vabs_max)
    #         plt.colorbar(im1, ax=ax[0,k], fraction=0.046)
    #         axs[0,k].set_title('Media(alpha='+str(alpha[j])+', xi='+str(xi_GD[k])+')')
    #         im2=axs[1,k].imshow(se_mats[:,:,j,k],cmap='seismic')
            # plt.colorbar(im2, ax=ax[1,k],fraction=0.046)
            # axs[1,k].set_title('Standard error (alpha='+str(alpha[j])+',xi='+str(xi_GD[k])+')')
        

# %%
#MATRICI (vecchia versione)

#faccio la media e lo standard error sui segmenti --> ottengo due matrici medie (una per ogni alpha)
mean_mats=np.zeros((n_alpha,n_xi,N,N))
se_mats=np.zeros((n_alpha,n_xi,N,N))
for j in range(n_alpha):
    for k in range(n_xi):
        mean_mats[j,k,:,:]=np.mean(t_mats[:,j,k,:,:],axis=0)
        se_mats[j,k,:,:]=t_mats[:,j,k,:,:].std(axis=0)/np.sqrt(g)



fig, axs = plt.subplots(n_alpha,1, constrained_layout=True)
fig.set_size_inches(10, 15)

# add subfigure per subplot
gridspec = axs[0].get_subplotspec().get_gridspec()
subfigs = [fig.add_subfigure(gs) for gs in gridspec]

for j, subfig in enumerate(subfigs):
    
    subfig.suptitle('alpha='+str(alpha[j])+', T='+str(T[j]), fontsize=40)

    # create 1x3 subplots per subfig
    axs = subfig.subplots(1, 2)
    vabs_max=np.max(np.abs(mean_mats[j,k,:,:]))
    im1=axs[0].imshow(mean_mats[j,k,:,:],cmap='seismic',vmin=-vabs_max, vmax=vabs_max)
    plt.colorbar(im1, ax=axs[0], fraction=0.046)
    axs[0].set_title('Media')
    im2=axs[1].imshow(se_mats[j,k,:,:],cmap='seismic')
    plt.colorbar(im2, ax=axs[1],fraction=0.046)
    axs[1].set_title('Standard error')

fig.savefig(path+'/mean and se matrix')




