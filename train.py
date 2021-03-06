import tensorflow as tf
from keras.backend.tensorflow_backend import set_session, clear_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = -1
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import model as model
import data
import random
import numpy
import json
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.sequence import pad_sequences
import sys
import rbfopt
import gc
import time
import os.path

def params_from_name(other_model):
    other_model=os.path.basename(other_model)
    name,params=other_model.split("__",1)
    param_dict={}
    for param_val in params.split("__"):
        param,val=param_val.rsplit("_",1)
        val=float(val)
        param_dict[param]=val
    return param_dict
        

def train(args,batch_size=1000,epochs=13,patience=2,params_in_name=True,**kwargs): #kwargs is parameters
    param_string="__".join("{}_{}".format(k,v) for k,v in sorted(kwargs.items()))
    model_class=getattr(model,args.classname) #Pick the right class
    model_name=args.model_file
    if params_in_name:
        model_name+="__"+param_string
    m=model_class() #instantiate the model
    m.build_model(args.dicts_file,word_seq_len,word_embeddings,**kwargs)
    m.save_model(model_name)
    save_cb=ModelCheckpoint(filepath=model_name+".weights.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    es_cb=EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto')
    tensorboard_log_dir="{}.tensorboard.log/{}".format(args.model_file,param_string)
    tb_cb=TensorBoard(tensorboard_log_dir)
    print("Tensorboard logs in", tensorboard_log_dir, file=sys.stderr)
    hist=m.model.fit(x=inputs_train_dict, y=outputs_train_dict, validation_data=(inputs_devel_dict,outputs_devel_dict), verbose=1, batch_size=batch_size, epochs=epochs, callbacks=[save_cb,es_cb,tb_cb])
    with open(model_name+".history.json","w") as f:
        json.dump((hist.epoch,hist.history),f)
    retval=float(min(hist.history["val_loss"]))

    del m.model
    del m
    del hist
    clear_session()
    for _ in range(10):
        gc.collect()
        time.sleep(0.1)

    return retval

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--train-file', help='.conllu')
    parser.add_argument('--devel-file', help='.conllu')
    parser.add_argument('--dicts-file', help='.json')
    parser.add_argument('--embeddings', help='.vector or .bin')
    parser.add_argument('--maxrank-emb', type=int, default=100000, help='Max rank of the embedding')
    parser.add_argument('--classname', help='Name of class in model.py')
    parser.add_argument('--like', help='train a new model like this one, pick parameters from the name')
    parser.add_argument('--model-file', help='file-name-prefix to save to')
    args = parser.parse_args()

    word_embeddings=data.read_embeddings(args.embeddings,args.maxrank_emb)
    
    with open(args.train_file) as f:
        train_conllu=data.read_conll(f)
        inputs_train_dict,outputs_train_dict,output_features=data.prep_data(train_conllu,args.dicts_file,word_embeddings.vocab,word_seq_len=None,shuffle=True)
    word_seq_len=inputs_train_dict["inp_char_seq"].shape[1]
    with open(args.devel_file) as f:
        devel_conllu=data.read_conll(f)
        inputs_devel_dict,outputs_devel_dict,output_features_dev=data.prep_data(devel_conllu,args.dicts_file,word_embeddings.vocab,word_seq_len=word_seq_len,shuffle=False)
        assert output_features==output_features_dev

    if args.like:
        params=params_from_name(args.like)
        print("Training with parameters:",params,file=sys.stderr)
        sys.stderr.flush()
        train(args,batch_size=250,epochs=30,patience=4,params_in_name=False,**params)
    else:
        def black_box(hyperparameters):
            (lr,do,kern_l2,act_l2)=hyperparameters
            return train(args,lr=lr,do=do,kern_l2=kern_l2,act_l2=act_l2)
                                                   #  lr    do    k_l2  a_l2
        bb=rbfopt.RbfoptUserBlackBox(4,numpy.array([0.001,  0.0,  0.0,   0.0]),\
                                       numpy.array([0.009,  0.3,  0.0001, 0.0001]),numpy.array(['R','R','R','R']),black_box)
        settings = rbfopt.RbfoptSettings(max_clock_time=8*60*60,target_objval=0.0,num_cpus=1)
        alg = rbfopt.RbfoptAlgorithm(settings, bb)
        val, x, itercount, evalcount, fast_evalcount = alg.optimize()
        with open(args.model_file+".rbfopt.log.json","wt") as f:
            json.dump((val, list(x), itercount, evalcount, fast_evalcount),f)
        print("FINAL",val, x, itercount, evalcount, fast_evalcount,file=sys.stderr)
