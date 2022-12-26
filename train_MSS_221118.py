
import tensorflow as tf
from os import walk
import os
import numpy as np
import librosa
from util import to_spec
from model import multi_band_multi_stack as infer
from config import NetConfig_pansori, ModelConfig
from random import *
import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


traindrum = []
trainsong = []
trainmix = []
trainNum = 0
batchSize = 1
 
       
for (root, dirs, files) in walk(NetConfig_pansori.DATA_PATH+'janggu/'):
    for filename in files:
        print("the filename for janggu", filename)
        fullpath_drum = os.path.join(root, filename)
        drum_wav = librosa.load(fullpath_drum, sr=ModelConfig.SR, mono=True)[0]
        drum_spec = to_spec(drum_wav)
        #print("the size of drum_spec", drum_spec.size)
        drum_spec_mag = np.abs(drum_spec)
        maxVal = np.max(drum_spec_mag)
        traindrum.append(drum_spec_mag/maxVal)
        
for (root, dirs, files) in walk(NetConfig_pansori.DATA_PATH+'song/'):
    for filename in files:
        print("the filename for song", filename)
        fullpath_song = os.path.join(root, filename)
        song_wav = librosa.load(fullpath_song, sr=ModelConfig.SR, mono=True)[0]
        song_spec = to_spec(song_wav)
        #print("the size of song_spec", song_spec.size)
        song_spec_mag = np.abs(song_spec)
        maxVal = np.max(song_spec_mag)
        trainsong.append(song_spec_mag/maxVal)
        
for (root, dirs, files) in walk(NetConfig_pansori.DATA_PATH+'mix/'):
    for filename in files:
        print("the filename for mix", filename)
        fullpath_mix = os.path.join(root, filename)
        mix_wav = librosa.load(fullpath_mix, sr=ModelConfig.SR, mono=True)[0]
        mix_spec = to_spec(mix_wav)
        #print("the size of mix_spec", mix_spec.size)
        mix_spec_mag = np.abs(mix_spec)
        maxVal = np.max(mix_spec_mag)
        trainmix.append(mix_spec_mag/maxVal)
        
        trainNum = trainNum+1
        
print('Number of training examples : {}'.format(trainNum))

start_time = time.time()

def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"

# Model
print('Initialize network')

y_output=[]
x_mixed = tf.placeholder(tf.float32, shape=(batchSize, 512, 64, 1), name='x_mixed') # the input is mixed spectrogram
y_mixed = tf.placeholder(tf.float32, shape=(batchSize, 512, 64, 2), name='y_mixed') # the label has 4 spectrogram for each source
y_pred = infer(x_mixed,2)
#net = tf.make_template('net',y_pred)
y_output.append(tf.multiply(x_mixed,y_pred[0]))
loss_0 = tf.reduce_mean(tf.abs(y_mixed - y_output[0]) , name='loss0')
    
y_output.append(tf.multiply(x_mixed,y_pred[1]))
loss_1 = tf.reduce_mean(tf.abs(y_mixed - y_output[1]) , name='loss1')
    
y_output.append(tf.multiply(x_mixed,y_pred[2]))
loss_2 = tf.reduce_mean(tf.abs(y_mixed - y_output[2]) , name='loss2')
    
y_output.append(tf.multiply(x_mixed,y_pred[3]))
loss_3 = tf.reduce_mean(tf.abs(y_mixed - y_output[3]) , name='loss3')
    
loss_fn = loss_0+loss_1+loss_2+loss_3
# Loss, Optimizer
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
learning_rate = tf.train.exponential_decay(NetConfig_pansori.LR, global_step,NetConfig_pansori.DECAY_STEP, NetConfig_pansori.DECAY_RATE, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_fn, global_step=global_step)

x_input = np.zeros((batchSize, 512, 64, 1),dtype=np.float32)
y_input = np.zeros((batchSize, 512, 64, 2),dtype=np.float32)

displayIter = 50
lossAcc = 0
randperm = np.random.permutation(trainNum)
curIndex = 0

def count_number_trainable_params():
    
    #Counts the number of trainable variables.
    
    tot_nb_params = 0
    for trainable_variable in tf.trainable_variables():
        shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
        current_nb_params = get_nb_params_shape(shape)
        tot_nb_params = tot_nb_params + current_nb_params
    return tot_nb_params

def get_nb_params_shape(shape):
    
    #Computes the total number of params for a given shap.
    #Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
    
    nb_params = 1
    for dim in shape:
        nb_params = nb_params*int(dim)
    return nb_params 


with tf.Session(config=NetConfig_pansori.session_conf) as sess:

    print("Number of trainable parameters: %d" % count_number_trainable_params())

    # Initialized, Load state
    sess.run(tf.global_variables_initializer())

    for step in range(global_step.eval(), NetConfig_pansori.FINAL_STEP):

        for i in range(batchSize):
            seq = randperm[curIndex]
            start = randint(0,trainmix[seq].shape[-1]-64)
            x_input[i,:,:,:] = np.expand_dims(trainmix[seq][0:512,start:start+64],2)
            y_input[i,:,:,0] = traindrum[seq][0:512,start:start+64]
            y_input[i,:,:,1] = trainsong[seq][0:512,start:start+64]
            curIndex = curIndex+1
            if curIndex == trainNum:
                curIndex = 0
                randperm = np.random.permutation(trainNum)

        l = sess.run([loss_fn, optimizer],
                                 feed_dict={x_mixed: x_input, y_mixed: y_input})

        lossAcc = lossAcc+l[0]
        if step%displayIter==0:
            print('step-{}\tloss={}'.format(step, lossAcc/displayIter))
            print("Elapsed time: ", elapsed(time.time() - start_time))
            lossAcc = 0

        # Save state
        if step % NetConfig_pansori.CKPT_STEP == 0:
            tf.train.Saver().save(sess, NetConfig_pansori.CKPT_PATH + '/checkpoint', global_step=step)
