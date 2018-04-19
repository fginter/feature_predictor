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
from keras.models import model_from_json

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
        rnn_dim=500
        
        self.dicts_filename=dicts_filename
        with open(self.dicts_filename,"rt") as f:
            self.char_dict,self.pos_dict,self.deprel_dict,self.feat_val_dict=json.load(f)
        #vectorized train is list of words
        #each word is (input,output)
        #input is [[...char sequence...], pos, deprel]
        #output is [ classnum, classnum, classnum ] with as many classes as there are features
        
        inp_chars=Input(name="inp_char_seq",shape=(word_seq_len,)) #this is a sequence
        inp_pos=Input(name="inp_pos",shape=(1,)) #one POS
        inp_deprel=Input(name="inp_deprel",shape=(1,)) #one DEPREL

        chars_emb=Embedding(len(self.char_dict),char_emb_dim,mask_zero=False,embeddings_initializer=Constant(value=0.01))(inp_chars)
        pos_emb=Flatten()(Embedding(len(self.pos_dict),pos_emb_dim,embeddings_initializer=Constant(value=0.01))(inp_pos))
        drel_emb=Flatten()(Embedding(len(self.deprel_dict),deprel_emb_dim,embeddings_initializer=Constant(value=0.01))(inp_deprel))

        rnn_out=Bidirectional(LSTM(rnn_dim))(chars_emb)

        cc=Concatenate()([rnn_out,pos_emb,drel_emb])
        hidden=Dense(501,activation="tanh")(cc)
        outputs=[]
        for feat_name in sorted(self.feat_val_dict.keys()):
            outputs.append(Dense(len(self.feat_val_dict[feat_name]),name="out_"+normname(feat_name),activation="softmax")(hidden))
        
        self.model=Model(inputs=[inp_chars,inp_pos,inp_deprel], outputs=outputs)
        self.model.compile(optimizer="adam",loss="sparse_categorical_crossentropy")

    def load_model(self,model_name):
        with open(model_name+".model.json", "rt") as f:
            self.model=model_from_json(f.read())
            self.model.load_weights(model_name+".weights.h5")
        with open(model_name+".dicts.json","rt") as f:
            self.char_dict,self.pos_dict,self.deprel_dict,self.feat_val_dict=json.load(f)
        
    def save_model(self,file_name):
        model_json = self.model.to_json()
        with open(file_name+".model.json","w") as f:
            print(model_json,file=f)
        with open(file_name+".dicts.json","w") as f:
            json.dump((self.char_dict,self.pos_dict,self.deprel_dict,self.feat_val_dict),f)
            
    def word_seq_len(self):
        try:
            return self.model.get_layer("inp_char_seq").get_config()["batch_input_shape"][1]
        except ValueError:
            return self.model.get_layer("input_1").get_config()["batch_input_shape"][1]




        
        
def acc(out,out_gold):
    out_pred=numpy.vstack([numpy.argmax(p,axis=-1) for p in out]).T  #examples by output
    out_gold=numpy.vstack(out_gold).T                               #examples by output
    matching=numpy.where(out_pred==out_gold,1,0)  #examples by output
    row_match_count=numpy.sum(matching,axis=1)   #examples, count of matching outputs
    correct=len(numpy.where(row_match_count>=24)[0])
    total=matching.shape[0]
    print("ACC=",correct/total,"(",correct,"/",total,")")

            
