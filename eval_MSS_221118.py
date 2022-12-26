import numpy as np
import tensorflow as tf
from os import walk
import os
from model import multi_band_multi_stack as infer
#from config import NetConfig_DSD_100, ModelConfig
from config import NetConfig_pansori, ModelConfig
from util import to_spec, to_wav_file, bss_eval_sdr
import librosa
from statistics import median
import time
import librosa.display
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batchSize = 1

start_time = time.time()

def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"

# Model
print('Initialize network model')

x_mixed = tf.placeholder(tf.float32, shape=(batchSize, 512, 64, 1), name='x_mixed')
y_mixed = tf.placeholder(tf.float32, shape=(batchSize, 512, 64, 2), name='y_mixed')
y_pred = infer(x_mixed,2)
print("the y_pred is",y_pred)
print("the y_pred 0 is",y_pred[0])
print("the y_pred 1 is",y_pred[1])
print("the y_pred 2 is",y_pred[2])
print("the y_pred 3 is",y_pred[3])


#y_output = tf.multiply(x_mixed,y_pred)
net = tf.make_template('net',y_pred)

x_input = np.zeros((batchSize, 512, 64, 1),dtype=np.float32)
#y_input = np.zeros((batchSize, 512, 64, 2),dtype=np.float32)


sdr_janggu = []
sdr_song  = []

with tf.Session(config=NetConfig_pansori.session_conf) as sess:

    # Initialized, Load state
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(NetConfig_pansori.CKPT_PATH + '/checkpoint-50000'))
    if ckpt and ckpt.model_checkpoint_path:
        print('Load weights')
        tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
        print("what is NetConfig_pansori.DATA_PATH", NetConfig_pansori.DATA_PATH)
        print("what is full path", NetConfig_pansori.DATA_PATH+'/mixtures/Test/')

        for (root, dirs, files) in walk(NetConfig_pansori.DATA_PATH+'/mixtures/Test/'):
            for d in dirs:
                print(d)

                #filenameBass = NetConfig_DSD_100.DATA_PATH+'/Sources/Test/'+d+'/bass.wav'
                filenamejanggu = NetConfig_pansori.DATA_PATH+'/sources/Test/'+d+'/janggu.wav'
                filenamesong = NetConfig_pansori.DATA_PATH+'/sources/Test/'+d+'/song.wav'
                #filenameOther = NetConfig_DSD_100.DATA_PATH+'/Sources/Test/'+d+'/other.wav'
                filenameMix = NetConfig_pansori.DATA_PATH+'/mixtures/Test/'+d+'/mix.wav'

                mixed_wav  = librosa.load(filenameMix, sr=ModelConfig.SR, mono=True)[0]
                gt_wav_janggu = librosa.load(filenamejanggu, sr=ModelConfig.SR, mono=True)[0]
                #gt_wav_drum = librosa.load(filenameDrums, sr=ModelConfig.SR, mono=True)[0]
                #gt_wav_other = librosa.load(filenameOther, sr=ModelConfig.SR, mono=True)[0]
                gt_wav_song = librosa.load(filenamesong, sr=ModelConfig.SR, mono=True)[0]
                
                the_spec = to_spec(gt_wav_janggu)
                the_spec_mag = np.abs(the_spec)
                the_maxVal = np.max(the_spec_mag)
                final_data = the_spec_mag/the_maxVal
                '''
                librosa.display.specshow(librosa.power_to_db(final_data,ref=np.max),y_axis='mel', fmax=8000,x_axis='time')
                plt.title('ground truth for janggu')
                plt.colorbar(format='%+2.0f dB')
                plt.tight_layout()
                plt.show()
                '''
                
                the_spec_song = to_spec(gt_wav_song)
                the_spec_mag_song = np.abs(the_spec_song)
                the_maxVal_song = np.max(the_spec_mag_song)
                final_data_song = the_spec_mag_song/the_maxVal_song
                '''
                librosa.display.specshow(librosa.power_to_db(final_data_song,ref=np.max),y_axis='mel', fmax=8000,x_axis='time')
                plt.title('ground truth for song')
                plt.colorbar(format='%+2.0f dB')
                plt.tight_layout()
                plt.show()
                '''
                
                mixed_spec = to_spec(mixed_wav)
                mixed_spec_mag = np.abs(mixed_spec)
                mixed_spec_phase = np.angle(mixed_spec)
                maxTemp = np.max(mixed_spec_mag)
                mixed_spec_mag = mixed_spec_mag/maxTemp
                
                '''
                librosa.display.specshow(librosa.power_to_db(mixed_spec_mag,ref=np.max),y_axis='mel', fmax=8000,x_axis='time')
                plt.title('mix source of janggu and song')
                plt.colorbar(format='%+2.0f dB')
                plt.tight_layout()
                plt.show()
                '''

                mixed_wav_orig,sr_orig = librosa.load(filenameMix, sr=None, mono=True)
                #gt_wav_dialogue_orig = librosa.load(filenamedialogue, sr=None, mono=True)[0]
                gt_wav_janggu_orig = librosa.load(filenamejanggu, sr=None, mono=True)[0]
                #gt_wav_other_orig = librosa.load(filenameOther, sr=None, mono=True)[0]
                gt_wav_song_orig = librosa.load(filenamesong, sr=None, mono=True)[0]

                srcLen = mixed_spec_mag.shape[-1]
                startIndex = 0
                #y_est_dialogue = np.zeros((512,srcLen),dtype=np.float32)
                y_est_janggu = np.zeros((512,srcLen),dtype=np.float32)
                #y_est_other = np.zeros((512,srcLen),dtype=np.float32)
                y_est_song = np.zeros((512,srcLen),dtype=np.float32)
                while startIndex+64<srcLen:
                    x_input[0,:,:,0] = mixed_spec_mag[0:512,startIndex:startIndex+64]
                    #print("the shape.............", y_pred.shape)
                    y_output = sess.run(y_pred, feed_dict={x_mixed: x_input})
                    #print("the shape.............", y_output)
                    #print("the y_output is............", y_output[1])
                    y_output = y_output[2]
                    if startIndex==0:
                        #y_est_dialogue[:,startIndex:startIndex+64] = y_output[0,:,:,0]
                        y_est_janggu[:,startIndex:startIndex+64] = y_output[0,:,:,0]
                        #y_est_other[:,startIndex:startIndex+64] = y_output[0,:,:,2]
                        y_est_song[:,startIndex:startIndex+64] = y_output[0,:,:,1]
                    else:
                        #y_est_dialogue[:,startIndex+16:startIndex+48] = y_output[0,:,16:48,0]
                        y_est_janggu[:,startIndex+16:startIndex+48] = y_output[0,:,16:48,0]
                        #y_est_other[:,startIndex+16:startIndex+48] = y_output[0,:,16:48,2]
                        y_est_song[:,startIndex+16:startIndex+48] = y_output[0,:,16:48,1]
                    startIndex = startIndex+32

                x_input[0,:,:,0] = mixed_spec_mag[0:512,srcLen-64:srcLen]
                y_output = sess.run(y_pred, feed_dict={x_mixed: x_input})
                #print("the shape second.............", y_output)
                #print("the y_output is............", y_output[1])
                y_output = y_output[2]
                srcStart = srcLen-startIndex-16
                #y_est_dialogue[:,startIndex+16:srcLen] = y_output[0,:,64-srcStart:64,0]
                y_est_janggu[:,startIndex+16:srcLen] = y_output[0,:,64-srcStart:64,0]
                #y_est_other[:,startIndex+16:srcLen] = y_output[0,:,64-srcStart:64,2]
                y_est_song[:,startIndex+16:srcLen] = y_output[0,:,64-srcStart:64,1]

                #y_est_dialogue[np.where(y_est_dialogue<0)] = 0
                y_est_janggu[np.where(y_est_janggu<0)] = 0
                #y_est_other[np.where(y_est_other<0)] = 0
                '''
                librosa.display.specshow(librosa.power_to_db(y_est_janggu,ref=np.max),y_axis='mel', fmax=8000 ,x_axis='time')
                plt.title('predicted janggu mask')
                plt.colorbar(format='%+2.0f dB')
                plt.tight_layout()
                plt.show()
                '''
                
                
                y_est_song[np.where(y_est_song<0)] = 0
                #y_est_dialogue = y_est_dialogue * mixed_spec_mag[0:512,:] * maxTemp
                y_est_janggu = y_est_janggu * mixed_spec_mag[0:512,:] * maxTemp
                #y_est_other = y_est_other * mixed_spec_mag[0:512,:] * maxTemp
                y_est_song = y_est_song * mixed_spec_mag[0:512,:] * maxTemp
                
                '''
                librosa.display.specshow(librosa.power_to_db(y_est_janggu,ref=np.max),y_axis='mel', fmax=8000 ,x_axis='time')
                plt.title('predicted janggu')
                plt.colorbar(format='%+2.0f dB')
                plt.tight_layout()
                plt.show()
                
                
                librosa.display.specshow(librosa.power_to_db(y_est_song,ref=np.max),y_axis='mel', fmax=8000 ,x_axis='time')
                plt.title('predicted song')
                plt.colorbar(format='%+2.0f dB')
                plt.tight_layout()
                plt.show()
                '''
                
                #y_wav_dialogue = to_wav_file(y_est_dialogue,mixed_spec_phase[0:512,:])
                y_wav_janggu = to_wav_file(y_est_janggu,mixed_spec_phase[0:512,:])
                #y_wav_other = to_wav_file(y_est_other,mixed_spec_phase[0:512,:])
                y_wav_song = to_wav_file(y_est_song,mixed_spec_phase[0:512,:])


                #upsample to original SR
                #y_wav_dialogue_orig = librosa.resample(y_wav_dialogue,ModelConfig.SR,sr_orig) # this is the audio generate by network after resampling to original SR
                y_wav_janggu_orig = librosa.resample(y_wav_janggu,ModelConfig.SR,sr_orig)
                #y_wav_other_orig = librosa.resample(y_wav_other,ModelConfig.SR,sr_orig)
                y_wav_song_orig = librosa.resample(y_wav_song,ModelConfig.SR,sr_orig)

                sdr = bss_eval_sdr(gt_wav_janggu_orig,y_wav_janggu_orig)
                printstr = "janggu SDR : "
                printstr = printstr+str(sdr)+" song SDR : "
                sdr_janggu.append(sdr)
                
                sdr = bss_eval_sdr(gt_wav_song_orig,y_wav_song_orig)
                printstr = printstr+str(sdr)+" "
                sdr_song.append(sdr)
                #sdr = bss_eval_sdr(gt_wav_other_orig,y_wav_other_orig)
                #printstr = printstr+str(sdr)+" Vocal SDR : "
                #sdr_other.append(sdr)



                print(printstr)
                print("Elapsed time: ", elapsed(time.time() - start_time))

print('Median SDR')
print('janggu : ' + str(median(sdr_janggu)) + ' song : ' + str(median(sdr_song)))




