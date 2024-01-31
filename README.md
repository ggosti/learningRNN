BPTT for Discrete-time binary recurrent neural network
=======================================

Implementation of a discrete-time recurrent neural network with binary neurons which learns to predict the behavior of a role-model neural network with the backpropagation through time (BPTT) method.
In principle this learning neural network can be trained to predict any binary discrete-time dynamical system.


Requirements
------------

The code uses Python2.7 and the following Python open source packages:
- NumPy
- Matplotlib
- Pandas
- NetworkX

You may install the Python requirements with `pip install -r requirements.txt` or `conda install --yes --file requirements.txt`, or is you want to create a new conda enviroment `conda create --name myenv --file requirements.txt`.


Test
------------
Run `python test.py` in the folder generated from the unziped repository to test if everytihing is setup. The code `test.py`
runs the most important functions, which are explained in the rest of this file.

Generate Trajectories from the Model Network
--------------------------------------------

This implementation considers a discrete-time recurrent neural network with binary neurons.
The number of neurons `N` can be set, and the neurons can have activation state either {0,1} if `typ = 0`,
or {-1,1} if `typ = 1`. The activation function can only be a step function with a certain threshold thr.

In the following text it is assumed that the following packages are loaded:
```python
import learningRNN as lrnn
import numpy as np
import matplotlib.pyplot as plt
```

It is useful to set the following parameters:
```python
N= 16 # Number of neurons
typ = 0 # typ = 0 neurons with binary activation states {0,1}, typ = 1  neurons with states {-1,1}.
        # typ=1 --> {-1,1}    typ=0 --> {0,1} 
thr = 0 # activation function threshold
nP = {"N":N, "typ":typ, "thr": thr}
```

Generate a random small-world network with the NetworkX implementation of Watts-Strogatz algorithm
and add random weights. This network will be called objective: 

```python
num_genr, objective =lrnn.generateSmallWorldBase(N,3,0.3,rseed=6219)
```

`num_genr` is the Watts-Strogatz algorithm seed.
Get a int from 0 to 2^N (17 in this example),
this int is an index that labels the neuron's activation profile.
The corresponding binary vector, which will
be used as a neuron activation profile, can be obtained as:

```python
initial_state = lrnn.stateIndex2stateVec(17,N,typ)
```

We can get back to the index:
```python
initial_state_index = lrnn.stateVec2stateIndex(initial_state, N, typ)
```

Get the next state given the objective network and an `initial_state`
```python
transition = lrnn.transPy(initial_state,objective, N, typ, thr)
```

Produce the trajectory of the discrete-time recurrent neural network given the objective network and an `initial_state`.
`trajectory` is the full trajectory, and `cycle` is the limit cycle or attractor.
```python
cycle, trajectory = lrnn.getTrajPy(initial_state_index, objective, N,
                                   typ, thr, trajSize = 1000)
```

Given a list of initial states `initial_state_list`, get a list with the corresponding transition
```python
initial_state_list = lrnn.stateIndex2stateVecSeq([19,2001,377], N, typ)
transition_list = lrnn.transPy(initial_state_list,objective, N, typ, thr)
```

Train the Learner network
---------------------------

First we have to set the number of gradient descent steps `T`, and the learning rate `alpha`:
```python
T = 3000 # Number of gradient descent steps
alpha = 10.
```

To make the training set I generate several trajectories
with different initial states. All these trajectories are put in a list `seqs`:
```python
seqs = []
seeds = np.random.choice((2**N) , size=1400, replace=False)
for i,sm in enumerate(seeds):
    cycle1,path1 = lrnn.getTrajPy(sm,objective,N,0,0,100000)
    seq1 = list(path1)
    seqs.append(seq1)
```
From the list `seqs` I may generate the training set:
```python
X_train, Y_train = lrnn.makeTrainXYfromSeqs(seqs, nP, isIndex= True)
```
Similarly, I can generate the test set.
Now, I can train the learner network:
```python  
trained_matrix, deltas, fullDeltas, exTime, convStep, bestErrors, bestNet =\
     lrnn.runGradientDescent(X_train, Y_train, alpha0= 0.0, alphaHat=alpha,
                             batchFr = 1.0, passi=T, runSeed = 3098, gdStrat="GD", k=1, netPars=nP,
                             showGradStep= False, xi= 0.000, mexpon = -1.5,normalize=True)
```
Visualize how the cost function changed through the gradient descent iteration:
```python
plt.figure()
if np.isfinite(convStep):
  plt.plot(range(0,convStep+int(T/200),int(T/200)),deltas,label='Train alpha '+str(alpha))
else:
  plt.plot(range(0,T,int(T/200)),deltas,label='Train alpha '+str(alpha))
plt.legend()
```

Visualize the result:
```python
fig, (ax1,ax2)= plt.subplots(2)
ax1.set_title('objective')
ax1.imshow(objective,cmap='seismic',vmin=-1.,vmax=1.)
ax2.set_title('trained_matrix')
ax2.imshow(trained_matrix,cmap='seismic',vmin=-1.,vmax=1.)
plt.show(block=True)
```

Simultaneously Train and Test the Learner network
---------------------------

First we have to set the number of gradient descent steps `T`, and the learning rate `alpha`:
```python
T = 3000 # Number of gradient descent steps
alpha = 10.
```

To make the training set and the test set I generate several trajectories
with different initial states. All these trajectories are put in a list `seqs`:
```python
seqs = []
seeds = np.random.choice((2**N) , size=1400, replace=False)
for i,sm in enumerate(seeds):
    cycle1,path1 = lrnn.getTrajPy(sm,objective,N,0,0,100000)
    seq1 = list(path1)+[cycle1[0]]
    seqs.append(seq1)

seqsTest = []
seedsTest = np.random.choice((2**N) , size=400, replace=False)
for i,sm in enumerate(seedsTest):
    cycle1,path1 = lrnn.getTrajPy(sm,objective,N,0,0,100000)
    seq1 = list(path1)+[cycle1[0]]
    seqsTest.append(seq1)
```
From the lists `seqs` and `seqsTest` I may generate the training and the test sets:
```python
X_train, Y_train = lrnn.makeTrainXYfromSeqs(seqs, nP, isIndex= True)
X_test, Y_test = lrnn.makeTrainXYfromSeqs(seqsTest, nP, isIndex= True)
```
Similarly, I can generate the test set.
Now, I can train the learner network:
```python  
trained_matrix, deltas, fullDeltas,exTime,convStep, bestErrors, bestNet, deltasTest=\
     lrnn.runGradientDescent(X_train, Y_train, alpha0= 0.0, alphaHat=alpha,
                             batchFr = 1.0, passi=T, runSeed = 3098, gdStrat="GD", k=1, netPars=nP,
                             showGradStep= False, xi= 0.000, mexpon = -1.5,normalize=True,
                             Xtest=X_test, ytest=Y_test)
```

Visualize the test and training score evolution:
```python
plt.figure()
if np.isfinite(convStep):
  Ts = range(0,convStep+int(T/200),int(T/200))
else:
  Ts = range(0,T,int(T/200))
  
plt.plot(Ts,deltas,label='Train alpha '+str(alpha))
plt.plot(Ts,deltasTest,label='Test alpha '+str(alpha))
plt.legend()
```


Visualize the result:
```python
fig, (ax1,ax2)= plt.subplots(2)
ax1.set_title('objective')
ax1.imshow(objective,cmap='seismic',vmin=-1.,vmax=1.)
ax2.set_title('trained_matrix')
ax2.imshow(trained_matrix,cmap='seismic',vmin=-1.,vmax=1.)
plt.show(block=True)
```
