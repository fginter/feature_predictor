import model as model
import data
import random
import numpy
import json
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--train-file', help='.conllu')
    parser.add_argument('--devel-file', help='.conllu')
    args = parser.parse_args()

    data_train=data.vectorize_data(open(args.train_file),"dicts_fi.json")
    data_devel=data.vectorize_data(open(args.devel_file),"dicts_fi.json")

    random.shuffle(data_train)
    inputs_train=numpy.array([item[0] for item in data_train])
    inputs_train_lst=[pad_sequences(inputs_train[:,0],padding="pre"),inputs_train[:,1],inputs_train[:,2]]
    word_seq_len=inputs_train_lst[0].shape[1]
    
    outputs_train=numpy.array([item[1] for item in data_train])
    outputs_train_lst=[outputs_train[:,i] for i in range(outputs_train.shape[1])]

    inputs_devel=numpy.array([item[0] for item in data_devel])
    inputs_devel_lst=[pad_sequences(inputs_devel[:,0],padding="pre",maxlen=word_seq_len),inputs_devel[:,1],inputs_devel[:,2]]
    outputs_devel=numpy.array([item[1] for item in data_devel])
    outputs_devel_lst=[outputs_devel[:,i] for i in range(outputs_devel.shape[1])]

    m=model.Predictor()
    m.build_model("dicts_fi.json",word_seq_len)

    model_name="second_try"
    m.save_model(model_name)
    save_cb=ModelCheckpoint(filepath=model_name+".weights.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    hist=m.model.fit(x=inputs_train_lst, y=outputs_train_lst, validation_data=(inputs_devel_lst,outputs_devel_lst), verbose=1, batch_size=200, epochs=15, callbacks=[save_cb])
    with open(model_name+".history.json","w") as f:
        json.dump((hist.epoch,hist.history),f)

