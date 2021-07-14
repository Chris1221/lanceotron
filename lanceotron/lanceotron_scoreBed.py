#!/package/python3-base/3.8.3/bin/python3.8

import lanceotron
import lanceotron_classic as LoT
import argparse
import numpy as np
import os
import pyBigWig
from keras.models import load_model
from keras import backend as K
from joblib import Parallel, delayed
import time
import csv
import pickle
from sklearn.preprocessing import StandardScaler
import keras
import tensorflow as tf

parser = argparse.ArgumentParser(description='Sort significantly enriched regions of ChIP-seq singnals using a CNN')
parser.add_argument('file', help='bigwig file')
parser.add_argument('-b', '--bed', type=str, help='bed file used')
parser.add_argument('-m', '--model', type=str, help='Deep learning model to classify candidate peaks')
parser.add_argument('-f', '--folder', type=str, help='folder in public directory for writing')
args = parser.parse_args()

bins = 1000
bed_file = args.bed
bigWig_file = args.file
base_model_file = args.model
out_folder = args.folder
read_coverage_factor = 10**9

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

def build_model():
    deep_dense_size = 10
    dropout_rate = 0.5
    first_filter_num = 70
    first_filter_size = 9
    hidden_filter_num = 120
    hidden_filter_size = 6
    learning_rate = 0.0001
    wide_and_deep_dense_size = 70


    input_wide = keras.layers.Input(shape=(12, 1))
    wide_model = keras.layers.Flatten()(input_wide)
    input_deep = keras.layers.Input((2000, 1))
    deep_model = input_deep
    # deep model first conv layer 
    deep_model = keras.layers.Convolution1D(first_filter_num, kernel_size=first_filter_size, padding='same')(deep_model)
    deep_model = keras.layers.BatchNormalization()(deep_model)
    #deep_model = keras.layers.Activation('relu')(deep_model)
    deep_model = keras.layers.LeakyReLU()(deep_model)
    # deep model - 4 conv blocks
    deep_model = keras.layers.Convolution1D(hidden_filter_num, kernel_size=hidden_filter_size, padding='same')(deep_model)
    deep_model = keras.layers.BatchNormalization()(deep_model)
    #deep_model = keras.layers.Activation('relu')(deep_model)
    deep_model = keras.layers.LeakyReLU()(deep_model)
    deep_model = keras.layers.MaxPool1D(pool_size=2)(deep_model)

    deep_model = keras.layers.Convolution1D(hidden_filter_num, kernel_size=hidden_filter_size, padding='same')(deep_model)
    deep_model = keras.layers.BatchNormalization()(deep_model)
    #deep_model = keras.layers.Activation('relu')(deep_model)
    deep_model = keras.layers.LeakyReLU()(deep_model)
    deep_model = keras.layers.MaxPool1D(pool_size=2)(deep_model)

    deep_model = keras.layers.Convolution1D(hidden_filter_num, kernel_size=hidden_filter_size, padding='same')(deep_model)
    deep_model = keras.layers.BatchNormalization()(deep_model)
    #deep_model = keras.layers.Activation('relu')(deep_model)
    deep_model = keras.layers.LeakyReLU()(deep_model)
    deep_model = keras.layers.MaxPool1D(pool_size=2)(deep_model)

    deep_model = keras.layers.Convolution1D(hidden_filter_num, kernel_size=hidden_filter_size, padding='same')(deep_model)
    deep_model = keras.layers.BatchNormalization()(deep_model)
    #deep_model = keras.layers.Activation('relu')(deep_model)
    deep_model = keras.layers.LeakyReLU()(deep_model)
    deep_model = keras.layers.MaxPool1D(pool_size=2)(deep_model)

     # deep model - dense layer with dropout 
    deep_model = keras.layers.Dense(deep_dense_size)(deep_model)
    deep_model = keras.layers.BatchNormalization()(deep_model)
    #deep_model = keras.layers.Activation('relu')(deep_model)
    deep_model = keras.layers.LeakyReLU()(deep_model)
    deep_model = keras.layers.Dropout(dropout_rate)(deep_model)
    deep_model = keras.layers.Flatten()(deep_model)

    # shape output only dense layer
    shape_output = keras.layers.Dense(2, activation='softmax', name='shape_classification')(deep_model)

    # p-value output only dense layer
    pvalue_output = keras.layers.Dense(2, activation='softmax', name='pvalue_classification')(wide_model)

    # combine wide and deep paths
    concat = keras.layers.concatenate([wide_model, deep_model, pvalue_output])
    wide_and_deep = keras.layers.Dense(wide_and_deep_dense_size)(concat)
    wide_and_deep = keras.layers.BatchNormalization()(wide_and_deep)
    wide_and_deep = keras.layers.LeakyReLU()(wide_and_deep)

    wide_and_deep = keras.layers.Dense(wide_and_deep_dense_size)(wide_and_deep)
    wide_and_deep = keras.layers.BatchNormalization()(wide_and_deep)
    wide_and_deep = keras.layers.LeakyReLU()(wide_and_deep)
    output = keras.layers.Dense(2, activation='softmax', name='overall_classification')(wide_and_deep)
    model = keras.models.Model(inputs=[input_deep, input_wide], outputs=[output, shape_output, pvalue_output])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], loss_weights=[0.7, 0.3, 0], metrics=['accuracy'])
    model.load_weights('/package/lanceotron/20210519/models/wide_and_deep_fully_trained_v5_03.h5')
    return model


def classify_chrom_bed_list_DONOTUSE(chrom, bigWig_file, bed_list, base_model_file, out_folder):
    pyBigWig_object=pyBigWig.open(bigWig_file)
    if pyBigWig_object.intervals('{}'.format(chrom[0]))!=None:
        pileup_array=np.zeros(chrom[1])
        pyBigWig_object=pyBigWig.open(bigWig_file)
        for reads in pyBigWig_object.intervals('{}'.format(chrom[0])):
            pileup_array[reads[0]:reads[1]] = reads[2]
        chrom_mean = pyBigWig_object.stats(chrom[0], type='mean', exact=True)
        chrom_std = pyBigWig_object.stats(chrom[0], type='std', exact=True)
        pyBigWig_object.close()
        coord_list = []
        for bed_entry in bed_list:
            if bed_entry[0] == chrom[0]:
                coord_list.append([bed_entry[1], bed_entry[2], bed_entry[3]])
        if base_model_file.endswith('baseModel_categories.h5') or base_model_file.endswith('custom.h5'):
            signal_array = lanceotron.extract_signal_legacy(pileup_array, coord_list, chrom_mean, chrom_std)
        elif base_model_file == 'newExtractSignalOldTrainingSet_trichannel.h5':
            signal_array = lanceotron.extract_signal_2knew(pileup_array, coord_list, chom_mean, chrom_std)
        elif base_model_file.endswith('trichannel.h5'):
                bins = 2000
                signal_array = lanceotron.extract_signal_2k(pileup_array, coord_list, chrom_mean, chrom_std)
        else:
            signal_array = lanceotron.extract_signal(pileup_array, coord_list, chrom_mean, chrom_std)
        model = load_model(base_model_file)
        chrom_classifications = model.predict(signal_array, verbose=0)
        if type(chrom_classifications)==np.ndarray:  
            region_count, class_count = chrom_classifications.shape
            with open('{}{}.bed'.format(out_folder, chrom[0]), 'w') as f:
                for i, coordinate in enumerate(coord_list):
                    f.write('{}\t{}\t{}\t{}'.format(chrom[0], coordinate[0], coordinate[1], coordinate[2]))
                    for j in range(class_count):
                        f.write('\t{:.5f}'.format(chrom_classifications[i][j]))
                    f.write('\n')
        K.clear_session()

def classify_chrom_bed_list(chrom, bigWig_file, bed_list, base_model_file, out_folder):
    pileup_array=np.zeros(chrom[1])
    pyBigWig_object=pyBigWig.open(bigWig_file)
    if pyBigWig_object.intervals('{}'.format(chrom[0])) == None:
        pileup_array=np.zeros(chrom[1])
        chrom_mean = 0.0
        chrom_std = 1.0
    else:
        for reads in pyBigWig_object.intervals('{}'.format(chrom[0])):
            pileup_array[reads[0]:reads[1]] = reads[2]
        chrom_mean = pyBigWig_object.stats(chrom[0], type='mean', exact=True)
        chrom_std = pyBigWig_object.stats(chrom[0], type='std', exact=True)
    pyBigWig_object.close()
    coord_list = []
    for bed_entry in bed_list:
        if bed_entry[0] == chrom[0]:
            coord_list.append([bed_entry[1], bed_entry[2], bed_entry[3]])
    if base_model_file.endswith('baseModel_categories.h5') or base_model_file.endswith('custom.h5'):
        signal_array = lanceotron.extract_signal_legacy(pileup_array, coord_list, chrom_mean, chrom_std)
    elif base_model_file == 'newExtractSignalOldTrainingSet_trichannel.h5':
        signal_array = lanceotron.extract_signal_2knew(pileup_array, coord_list, chom_mean, chrom_std)
    elif base_model_file.endswith('trichannel.h5'):
            bins = 2000
            signal_array = lanceotron.extract_signal_2k(pileup_array, coord_list, chrom_mean, chrom_std)
    else:
        signal_array = lanceotron.extract_signal(pileup_array, coord_list, chrom_mean, chrom_std)
    model = load_model(base_model_file)
    chrom_classifications = model.predict(signal_array, verbose=0)
    if type(chrom_classifications)==np.ndarray:  
        region_count, class_count = chrom_classifications.shape
        with open('{}{}.bed'.format(out_folder, chrom[0]), 'w') as f:
            for i, coordinate in enumerate(coord_list):
                f.write('{}\t{}\t{}\t{}'.format(chrom[0], coordinate[0], coordinate[1], coordinate[2]))
                for j in range(class_count):
                    f.write('\t{:.5f}'.format(chrom_classifications[i][j]))
                f.write('\n')
    K.clear_session()
    
def classify_chrom_bed_list_alt(chrom, bigWig_file, bed_list, base_model_file, out_folder):
    pyBigWig_object=pyBigWig.open(bigWig_file)
    read_coverage_total = pyBigWig_object.header()['sumData']
    read_coverage_rphm = read_coverage_total/read_coverage_factor
    if pyBigWig_object.intervals('{}'.format(chrom[0])) == None:
        pyBigWig_object.close()
        coverage_array=np.zeros(chrom[1])
    else:
        pyBigWig_object.close()
        bigwig_data = LoT.Bigwig_data(bigWig_file)
        chrom_stats_dict = bigwig_data.get_chrom_info(chrom[0])
        coverage_array = bigwig_data.make_chrom_coverage_map(chrom_stats_dict)/read_coverage_rphm
    coord_list = []
    id_list = []
    for bed_entry in bed_list:
        if bed_entry[0] == chrom[0]:
            coord_list.append([bed_entry[1], bed_entry[2]])
            id_list.append(bed_entry[3])
    X_wide_array, X_deep_array = LoT.extract_signal_wide_and_deep_chrom(coverage_array, coord_list, read_coverage_rphm)
    standard_scaler_wide = pickle.load(open('/package/lanceotron/20210519/standard_scaler/standard_scaler_wide_v5_03.p', 'rb'))
    X_wide_array_norm = standard_scaler_wide.transform(X_wide_array)
    X_wide_array_norm = np.expand_dims(X_wide_array_norm, axis=2)
    standard_scaler = StandardScaler()
    X_deep_array_norm_T = standard_scaler.fit_transform(X_deep_array.T)
    standard_scaler_deep = pickle.load(open('/package/lanceotron/20210519/standard_scaler/standard_scaler_deep_v5_03.p', 'rb'))
    X_deep_array_norm = standard_scaler_deep.transform(X_deep_array_norm_T.T)
    X_deep_array_norm = np.expand_dims(X_deep_array_norm, axis=2)
    model = build_model()
    model_classifications = model.predict([X_deep_array_norm, X_wide_array_norm], verbose=0)
    K.clear_session()
    chrom_file_out = []
    for i, coord_pair in enumerate(coord_list):
        out_list = [chrom[0], coord_pair[0], coord_pair[1], id_list[i], model_classifications[0][i][0], model_classifications[1][i][0], model_classifications[2][i][0], model_classifications[0][i][1]]
        X_wide_list = X_wide_array[i][:-1].tolist()
        out_list+=X_wide_list
        chrom_file_out.append(out_list)
    with open('{}{}.bed'.format(out_folder, chrom[0]), 'w', newline='') as f:
        bed_writer = csv.writer(f, delimiter='\t')
        bed_writer.writerows(chrom_file_out)

    
    
bed_list = lanceotron.bed_file_to_list(bed_file)

chroms_in_bed = []
for bed_entry in bed_list:
    if bed_entry[0] not in chroms_in_bed:
        chroms_in_bed.append(bed_entry[0])
chrom_list = []
pyBigWig_object = pyBigWig.open(bigWig_file)
bigWig_chroms = pyBigWig_object.chroms()
for chrom in chroms_in_bed:
    if chrom in bigWig_chroms:
        chrom_list.append([chrom, bigWig_chroms[chrom]])
chrom_list = sorted(chrom_list, key=lambda chrom: chrom[1], reverse=True)

merge_folder = '{}merge/'.format(out_folder)
if not os.path.exists(merge_folder):
    os.makedirs(merge_folder)
        
cores = 1

if base_model_file.startswith('wide_and_deep_fully_trained_v5'):
    Parallel(n_jobs=cores)(delayed(classify_chrom_bed_list_alt)(chrom, bigWig_file, bed_list, base_model_file, merge_folder) for chrom in chrom_list)
else:
    Parallel(n_jobs=cores)(delayed(classify_chrom_bed_list)(chrom, bigWig_file, bed_list, base_model_file, merge_folder) for chrom in chrom_list)

with open('{}complete.txt'.format(out_folder), 'w') as f:
    f.write('')
