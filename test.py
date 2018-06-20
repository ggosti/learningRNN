import learningRNN as lrnn
import numpy as np

N= 16 # Number of neurons
typ = 0 # typ = 0 neurons with binary activation states {0,1}, typ = 0  neurons with states {-1,1}
thr = 0 # activation function threshold
nP = {"N":N, "typ":typ, "thr": thr}

# generate objective network
num_genr, objective =lrnn.generateSmallWorldBase(N,3,0.3,rseed=2219)

# generate inital state
initial_state = lrnn.stateIndex2stateVec(17,N,typ)
initial_state_index = lrnn.stateVec2stateIndex(initial_state, N, typ)

# generate the corresponding transitions
transition = lrnn.transPy(initial_state,objective, N, typ, thr)

# generate list of inital states and the corresponding transitions
initial_state_list = lrnn.stateIndex2stateVecSeq([19,2001,377], N, typ)
transition_list = lrnn.transPy(initial_state_list,objective, N, typ, thr)

# generate trajectory
cycle, trajectory = lrnn.getTrajPy(initial_state_index, objective, N,
                                   typ, thr, trajSize = 1000)


# set the gradient descent hyperparameters
T = 1000 # Number of gradient descent steps
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
trained_matrix, deltas, fullDeltas, exTime, convStep =\
     lrnn.runGradientDescent(X_train, Y_train, alpha0= 0.0, alphaHat=alpha,
                         batchFr = 1.0, passi=T, runSeed = 3098, gdStrat="GD", k=1, netPars=nP,
                          showGradStep= None, xi= 0.000, mexpon = -1.5)

# see results
import matplotlib.pyplot as plt
fig, (ax1,ax2)= plt.subplots(2)
ax1.set_title('objective')
ax1.pcolor(objective)
ax2.set_title('trained_matrix')
ax2.pcolor(trained_matrix)
plt.show(block=True)
