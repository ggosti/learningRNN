import learningRNN as lrnn
import numpy as np
import hrnn

N= 16 # Number of neurons
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
print('transition',transition.T)
print('transition',transition.T[0])
#print('--objective--')
#print(objective[3:5,:])
transitionCpp = hrnn.tranCpp(initial_state,objective, typ, thr,1)
print('transitionCpp',transitionCpp)

print('-->Test transition Cpp',(transition.T[0]==transitionCpp).all())


# generate list of inital states and the corresponding transitions
initial_state_list = lrnn.stateIndex2stateVecSeq([19,2001,377], N, typ)
print('initial_state_list')
print(initial_state_list)
transition_list = lrnn.transPy(initial_state_list,objective, N, typ, thr)
print('transition_list')
print(transition_list)

# generate trajectory
cycle, trajectory = lrnn.getTrajPy(initial_state_index, objective, N,
                                   typ, thr, trajSize = 1000)


# set the gradient descent hyperparameters
T = 3000 # Number of gradient descent steps
alpha = 10


# generate 1400 trajectories
seqs = []
seeds = np.random.choice((2**N) , size=1400, replace=False)
for i,sm in enumerate(seeds):
       cicli1,path1 = lrnn.getTrajPy(sm,objective,N,0,0,100000)
       seq1 = list(path1)+[cicli1[0]]
       seqs.append(seq1)

X_train, Y_train = lrnn.makeTrainXYfromSeqs(seqs, nP, isIndex= True)

# run gradient descent
trained_matrix, deltas, fullDeltas, exTime, convStep, bestErrors, bestNet =\
     lrnn.runGradientDescent(X_train, Y_train, alpha0= 0.0, alphaHat=alpha,
                             batchFr = 1.0, passi=T, runSeed = 3198, gdStrat="GD", k=1, netPars=nP,
                             showGradStep= False, xi= 0.000, mexpon = -1.5,normalize=True)

# see results
import matplotlib.pyplot as plt
plt.figure()
if np.isinf(convStep):
    plt.plot(range(0,T,int(T/200)),deltas,label='Train alpha '+str(alpha))
    plt.semilogy()
else:
    plt.plot(range(0,convStep+int(T/200),int(T/200)),deltas,label='Train alpha '+str(alpha))
    plt.semilogy()
plt.legend()

fig, (ax1,ax2)= plt.subplots(2)
ax1.set_title('objective')
ax1.imshow(objective)
ax2.set_title('trained_matrix')
ax2.imshow(trained_matrix)
plt.show(block=True)
