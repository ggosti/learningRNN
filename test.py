import learningRNN as lrnn
import numpy as np
import hrnn
import time
import matplotlib.pyplot as plt

N= 12 # Number of neurons
typ = 0 # typ = 0 neurons with binary activation states {0,1}, typ = 0  neurons with states {-1,1}
thr = 0.0 # activation function threshold
nP = {"N":N, "typ":typ, "thr": thr}

# generate objective network
num_genr, objective =lrnn.generateSmallWorldBase(N,3,0.3,rseed=3219)

# generate inital state
initial_state = lrnn.stateIndex2stateVec(17,N,typ)
out=hrnn.stateIndex2stateVec(17,N)
print('out',out, (out==initial_state).all())
initial_state_index = lrnn.stateVec2stateIndex(initial_state, N, typ)
print('--> Test state index to state vec and back',initial_state_index==17)
initial_state_index_cpp = hrnn.stateVec2stateIndex(out)
initial_state_index_cpp2 = hrnn.stateVec2stateIndex(initial_state)
print('--> Test state index to state vec and back cpp implementation',initial_state_index_cpp,initial_state_index_cpp==17,initial_state_index_cpp2==17)

# generate the corresponding transitions
print('initial_state',initial_state)
transition = lrnn.transPy(initial_state,objective, N, typ, thr)
print('transition',transition.T[0])
print('--objective--')
print(objective)
transitionCpp = hrnn.tranCpp(initial_state,objective, typ, thr,1)
print('transitionCpp',transitionCpp)

print('-->Test transition Cpp',(transition.T[0]==transitionCpp).all())


# generate list of inital states and the corresponding transitions
initial_state_list = lrnn.stateIndex2stateVecSeq([19,2001,377], N, typ)
#print('initial_state_list')
#print(np.array(initial_state_list))
transition_list = lrnn.transPy(initial_state_list,objective, N, typ, thr)
#print('transition_list')
#print(transition_list)
transition_listCpp = hrnn.transManyStatesCpp(np.array(initial_state_list),objective, typ, thr,1)
#print('transitionCpp')
#print(transition_listCpp)
transition_listCBlas = hrnn.transManyStatesCBlas(np.array(initial_state_list),objective, typ, thr,1)
#print('transitionCBlass')
#print(transition_listCBlas)

print('-->Test transition Cpp more states',(transition_list ==  transition_listCpp).all())
print('-->Test transition CBlas more states',(transition_list ==  transition_listCBlas).all())



# generate trajectory
cycle, trajectory = lrnn.getTrajPy(initial_state_index, objective, N,
                                   typ, thr, trajSize = 1000)
print('cycle, trajectory',cycle, trajectory)


# generate 1400 trajectories
seqs = []
seeds = np.random.choice((2**N) , size=1400, replace=False)
for i,sm in enumerate(seeds):
       cicli1,path1 = lrnn.getTrajPy(sm,objective,N,0,0,100000)
       seq1 = list(path1)+[cicli1[0]]
       seqs.append(seq1)

X_train, Y_train = lrnn.makeTrainXYfromSeqs(seqs, nP, isIndex= True)
print('Y_train',Y_train.dtype)

t1= time.time()
transition_list = lrnn.transPy(X_train,objective, N, typ, thr)
t2= time.time()
transition_listCpp = hrnn.transManyStatesCpp(X_train,objective, typ, thr,1)
t3= time.time()
transition_listCBlas = hrnn.transManyStatesCBlas(X_train,objective, typ, thr,1)
t4= time.time()
print('pyhton prediction time',t2-t1)
print('cpp prediction time',t3-t2)
print('CBlas prediction time',t4-t3)
print('-->Test many transitions 2 Cpp',(transition_list ==  transition_listCpp).all())
print('-->Test many transitions 2 CBlas',(transition_list ==  transition_listCBlas).all())
#print(transition_list)
#print(transition_listCBlas)



net0 = np.float64(2*np.random.rand(N,N)-1) #np.zeros((r, w), dtype=np.float32)  # np.float32(np.random.randint(0, 2, size=(r, w)))  # np.float32(2*np.random.rand(r,w)-1)
net0 = lrnn.rowNorm(net0)
obj=net0.copy()
t1= time.time()
update,sumSqdelta,delta,_ = lrnn.gradientDescentStep(Y_train,X_train,net0,netPars=nP,autapse = True,signFuncInZero = 1)
t2= time.time()

results = hrnn.gradientDescentStepCpp(Y_train,X_train,net0,typ,thr, 1)
t3= time.time()

resultsCblas = hrnn.gradientDescentStepCblas(Y_train,X_train,net0,typ,thr, 1)
t4= time.time()

print('gradeint descent step pyhton',t2-t1)
print('gradeint descent step cpp',t3-t2)
print('gradeint descent step Cblas cpp',t4-t3)
print('gradeint descent step results',results.keys())


print('deltas Cblas shape',resultsCblas['deltas'].shape)
print('deltas shape',delta.shape)
print('deltas Cblas',resultsCblas['deltas'])
print('deltas',delta)
print('--> Test deltas', (results['deltas']==delta).all() )
print('--> Test deltas Cblas', (resultsCblas['deltas']==delta).all() )
#print('update',results['update'])
#print('update',update)
print('--> Test update',np.sum((results['update']-update)**2))

"""

alpha = 0.00001
NSteps=100

t1= time.time()
net1,delta = lrnn.gradientDescentNSteps(Y_train,X_train,net0,alpha,NSteps,netPars=nP,autapse = True,signFuncInZero = 1)
t2= time.time()

results = hrnn.gradientDescentNStepsCpp(Y_train,X_train,obj,alpha,NSteps,typ,thr, 1)
t3= time.time()

print('gradeint descent Nsteps pyhton',t2-t1)
print('gradeint descent Nsteps cpp',t3-t2)
print('gradeint descent Nsteps results',results.keys())

#print('deltas',results['deltas'][:2,:])
#print('deltas',delta[:2,:])
print('--> Test deltas', (results['deltas']==delta).all() )
#print('net1',results['net'].shape)
#print('net1',net1.shape)
#print('net1',results['net'][:2,:])
#print('net1',net1[:2,:])
print('--> Test net0',np.sum((results['net']-net1)**2))

# set the gradient descent hyperparameters
T = 3000#3000 # Number of gradient descent steps
alpha = 10

# run gradient descent
trained_matrix, deltas, fullDeltas, exTime, convStep, bestErrors, bestNet=\
     lrnn.runGradientDescent(X_train, Y_train, alpha0= 0.0, alphaHat=alpha,
                             batchFr = 1.0, passi=T, runSeed = 3198, gdStrat="GD", k=1, netPars=nP,
                             showGradStep= False, xi= 0.000, mexpon = -1.5,normalize=True)

# see results

plt.figure()
#if np.isinf(convStep):
#    plt.plot(range(0,T,int(T/200)),deltas,label='Train alpha '+str(alpha))
#    plt.semilogy()
#else:
#    plt.plot(range(0,convStep+int(T/200),int(T/200)),deltas,label='Train alpha '+str(alpha))
#    plt.semilogy()
#plt.legend()
plt.plot(deltas)




# run gradient descent
trained_matrix1, deltas, exTime, convStep, bestErrors, bestNet =\
     lrnn.runGradientDescentCpp(X_train, Y_train, alpha0= 0.0, alphaHat=alpha,
                             passi=T, runSeed = 3198, netPars=nP,
                             xi= 0.000, mexpon = -1.5,normalize=True)

print('--> Test trained_matrix',np.sum((trained_matrix-trained_matrix1)**2))
# see results
#if np.isinf(convStep):
#    plt.plot(range(0,T,int(T/200)),deltas,label='Train alpha '+str(alpha))
#    plt.semilogy()
#else:
#    plt.plot(range(0,convStep+int(T/200),int(T/200)),deltas,label='Train alpha '+str(alpha))
#    plt.semilogy()
#plt.legend()
plt.plot(deltas)



fig, (ax1,ax2)= plt.subplots(2)
ax1.set_title('objective')
ax1.imshow(objective)
ax2.set_title('trained_matrix')
ax2.imshow(trained_matrix)

plt.show(block=True)
"""
