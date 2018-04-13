import tensorflow as tf
### Only needed for me, not to block the whole GPU, you don't need this stuff
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))
### ---end of weird stuff


import sys
import argparse
import data

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Activation, Conv1D, TimeDistributed, Flatten
from keras.layers import Bidirectional, Concatenate,Flatten,Reshape
from keras.optimizers import SGD, Adam
from keras.initializers import Constant
from keras.layers import CuDNNLSTM as LSTM  #massive speedup on graphics cards
#from keras.layers import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences

import numpy
import json
import re
import h5py

import random

def normname(n):
    return re.sub("[^a-zA-Z0-9]","_",n)

class Predictor:

    def build_model(self,dicts_filename,word_seq_len):

        char_emb_dim=100
        pos_emb_dim=100
        deprel_emb_dim=100
        rnn_dim=200
        
        self.dicts_filename=dicts_filename
        with open(self.dicts_filename,"rt") as f:
            self.char_dict,self.pos_dict,self.deprel_dict,self.feat_val_dict=json.load(f)
        #vectorized train is list of words
        #each word is (input,output)
        #input is [[...char sequence...], pos, deprel]
        #output is [ classnum, classnum, classnum ] with as many classes as there are features
        
        inp_chars=Input(shape=(word_seq_len,)) #this is a sequence
        inp_pos=Input(shape=(1,)) #one POS
        inp_deprel=Input(shape=(1,)) #one DEPREL

        chars_emb=Embedding(len(self.char_dict),char_emb_dim,mask_zero=False,embeddings_initializer=Constant(value=0.01))(inp_chars)
        pos_emb=Flatten()(Embedding(len(self.pos_dict),pos_emb_dim,embeddings_initializer=Constant(value=0.01))(inp_pos))
        drel_emb=Flatten()(Embedding(len(self.deprel_dict),deprel_emb_dim,embeddings_initializer=Constant(value=0.01))(inp_deprel))

        rnn_out=Bidirectional(LSTM(rnn_dim))(chars_emb)

        cc=Concatenate()([rnn_out,pos_emb,drel_emb])
        hidden=Dense(201,activation="tanh")(cc)
        outputs=[]
        for feat_name in sorted(self.feat_val_dict.keys()):
            outputs.append(Dense(len(self.feat_val_dict[feat_name]),name=normname(feat_name+"_out"),activation="softmax")(hidden))
        
        self.model=Model(inputs=[inp_chars,inp_pos,inp_deprel], outputs=outputs)
        self.model.compile(optimizer="adam",loss="sparse_categorical_crossentropy")

    def save_model(self,file_name):
        model_json = self.model.to_json()
        with open(file_name+".model.json", "w") as f:
            print(model_json,file=f)
        

# from sklearn.metrics import accuracy_score
# def accuracy(predictions, gold, lengths):
#     pred_tags = numpy.concatenate([labels[:lengths[i]] for i, labels in enumerate(predictions)]).ravel()
    
#     gold_tags = numpy.concatenate([labels[:lengths[i], 0] for i, labels in enumerate(gold)]).ravel()
    
#     print('Accuracy:', accuracy_score(gold_tags, pred_tags))

# class EvaluateFeats(keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         pred = numpy.argmax(self.model.predict(validation_vectorized_data_padded), axis=-1)
#         accuracy(pred, validation_vectorized_labels_padded, validation_lengths) # FIXME: Using global variables here, not good!

            
