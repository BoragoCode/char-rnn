#coding:utf-8
import theano
import theano.tensor as T
import sys
import numpy as np
import cPickle
from collections import OrderedDict

file = sys.argv[1]
data = open(file,'r').read()
data = data.decode('utf-8')
chars = list(set(data)) #char vocabulary

data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 0.005
iter = 40
save_freq = 5 #The step (counted by the number of iterations) at which the model is saved to hard disk.

print 'Compile the model...'
Wxh = theano.shared(value=np.float32(np.random.randn(hidden_size, vocab_size)*0.01), name='Wxh')
Whh = theano.shared(value=np.float32(np.random.randn(hidden_size, hidden_size)*0.01), name='Whh')
Why = theano.shared(value=np.float32(np.random.randn(vocab_size, hidden_size)*0.01), name='Why')
bh = theano.shared(value=np.zeros((hidden_size, 1),dtype='float32'), name='bh')
by = theano.shared(value=np.zeros((vocab_size, 1),dtype='float32'), name='by')

def RNN(idx, ht):
    xi = theano.shared(np.zeros((vocab_size,1), dtype='float32'),name='xi')
    #print type(idx)
    x = T.set_subtensor(xi[idx], 1.0)
    #x = theano.shared(value=xi, name='x')
    hs = T.tanh(T.dot(Wxh, x)+T.dot(Whh, ht) + bh)
    ys = T.dot(Why, hs) + by
    ps = T.exp(ys)/T.sum(T.exp(ys))
    ps = ps.flatten()
    
    return hs,ps

training_x = T.ivector('x_data')
training_y = T.ivector('y_data')

h0 = theano.shared(value=np.zeros((hidden_size, 1),dtype='float32'), name='h0')

_res, _ = theano.scan(RNN, sequences=[training_x], outputs_info=[h0, None])

classProbs = _res[1]
#print type(classProbs)
target_probs = T.diag(classProbs.T[training_y])
softmax_cost = -T.log(target_probs)
training_cost = T.sum(softmax_cost)

def sharedX(value, name=None, borrow=False, dtype=None):
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(theano._asarray(value, dtype=dtype),
                         name=name,
                         borrow=borrow)

def Adam(grads, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
    updates = []
    i = sharedX(0.)
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in grads.items():
        m = sharedX(p.get_value() * 0.)
        v = sharedX(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates

def compute_updates(training_cost, params):
    updates = []
    
    grads = T.grad(training_cost, params)
    grads = OrderedDict(zip(params, grads))
    
    for p, g in grads.items():
        m = sharedX(p.get_value() * 0.)
        m_t = m + g * g
        p_t = p - learning_rate * g / T.sqrt(m_t + 1e-8)
        updates.append((m, m_t))
        updates.append((p, p_t))
    
    return updates

params = [Wxh, Whh, Why, bh, by]
updates = compute_updates(training_cost, params)

train_model = theano.function(inputs=[training_x, training_y], outputs=[training_cost], updates=updates, on_unused_input='ignore', name="train_fn")
print 'Done!'
    

def dumpModel(filename):
    save_file = open(filename, 'wb')  # this will overwrite current contents
    for param in [Wxh, Whh, Why, bh, by]:
        cPickle.dump(param.get_value(borrow = True),save_file,-1)
    save_file.close()

def loadModel(filename):
    load_file = open(filename,'rb')
    param_list = [Wxh, Whh, Why, bh, by]
    for i in range(len(param_list)): 
        param_list[i].set_value(cPickle.load(load_file), borrow = True)
    load_file.close()

def sample(seed_ix, n):
    #loadModel('model20')

    Wxh_ = Wxh.get_value()
    Whh_ = Whh.get_value()
    Why_ = Why.get_value()
    bh_ = bh.get_value()
    by_ = by.get_value()
    
    hs = np.zeros((hidden_size,1))
    output = []
    for j in range(n):
        xs = np.zeros((vocab_size,1)) # encode in 1-of-k representation
        xs[seed_ix] = 1
        hs = np.tanh(np.dot(Wxh_, xs) + np.dot(Whh_, hs) + bh_) # hidden state
        ys = np.dot(Why_, hs) + by_ # unnormalized log probabilities for next chars
        ps = np.exp(ys) / np.sum(np.exp(ys)) # probabilities for next chars
        seed_ix = np.argmax(ps)
        
        output.append(seed_ix)
    return output

p = 0
n = 0
loss = 0
i = 1
idx_of_begin = chars.index(u'^')

print 'Begin training...'
while(i<=iter):
    if p+seq_length+1 >= len(data): 
        h0.set_value(np.zeros((hidden_size, 1),dtype='float32'))
        p = 0 # go from start of data
        print 'the iter is:',i
        print 'the loss is:',loss
        print 'average loss: ',loss/n
        loss = 0
        n = 0
        i += 1
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
    
    loss_ = train_model(inputs,targets)
    loss += loss_[0]
    if i%save_freq == 0 and n == 0:
        print('dump model:iter = %i' % i)
        dumpModel('model'+str(i))
    n += 1
    p += seq_length 

out = sample(0,50)
print 'sample:',''.join([ix_to_char[i] for i in out])
