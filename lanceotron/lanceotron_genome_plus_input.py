#!/package/python3-base/3.8.3/bin/python3.8

import lanceotron_classic as LoT
import os
import numpy as np
import pyBigWig
import pickle
from sklearn.preprocessing import StandardScaler
import keras
from keras import backend as K
import csv
from scipy.stats import poisson
import argparse
import tensorflow as tf
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='Sort significantly enriched regions of ChIP-seq singnals using a CNN')
parser.add_argument('file', help='bigwig file')
parser.add_argument('-i', '--input', type=str, help='input file used to calculate Poisson-based significance')
parser.add_argument('-t', '--threshold', type=float, help='threshold used for selecting candidate peaks')
parser.add_argument('-w', '--window', type=int, help='window size for rolling mean to use for selecting candidate peaks')
parser.add_argument('-m', '--model', type=str, help='Deep learning model to classify candidate peaks')
parser.add_argument('-f', '--folder', type=str, help='folder to write results to')

args = parser.parse_args()

bigwig_file = args.file
control_file = args.input
out_folder = args.folder
window = args.window
initial_threshold = args.threshold
min_peak_width = 50
max_peak_width = 2000
read_coverage_factor = 10**9
base_model_file = args.model

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

def calculate_pvalue_from_input(chrom, start, end, seq_depth_test, seq_depth_control, pyBigWig_object, max_height, length):
    if start<0:
        start = 0
    if end>length:
        end = length
    cov_input = pyBigWig_object.stats(chrom, start, end, type='mean', exact=True)[0]
    if cov_input:
        ave_coverage_input = cov_input*(seq_depth_test/seq_depth_control)
    else:
        ave_coverage_input = 0
    #ave_coverage_input = pyBigWig_object.stats(chrom, start, end, type='mean', exact=True)[0]*(seq_depth_test/seq_depth_control)
    with np.errstate(divide='ignore'):
        pvalue = -1*np.log10(1-poisson.cdf(max_height, ave_coverage_input))
    if np.isinf(pvalue):
        pvalue = 100.
    if pvalue>100:
        pvalue = 100.    
    return pvalue

def classify_chrom(chrom, chrom_stats_dict, read_coverage_rphm, read_coverage_rphm_input, bigwig_file, control_file, window, initial_threshold, min_peak_width, base_model_file, out_folder):
    bigwig_data = LoT.Bigwig_data(bigwig_file)

    if base_model_file == 'wide_and_deep_fully_trained_v5_03_grid_candidate.h5':
        window_list = [100, 200, 400, 800, 1600]
        threshold_list = [2, 4, 8, 16, 32]
        passed_tests_req = 5
        min_peak_width = 50
        max_peak_width = 800
        test_counter_array = np.zeros(genome_stats_dict[chrom]['chrom_len'])
        for window in window_list:
            coverage_array_smooth = bigwig_data.make_chrom_coverage_map(genome_stats_dict[chrom], smoothing=window)
            for threshold in threshold_list:
                test_passed_coord_list = LoT.label_enriched_regions_threshold(coverage_array_smooth, genome_stats_dict[chrom]['chrom_mean']*threshold)
                for coord_list in test_passed_coord_list:
                    test_counter_array[coord_list[0]:coord_list[1]]+=1
        initial_enriched_region_coord_list = LoT.label_enriched_regions_threshold(test_counter_array, passed_tests_req, min_peak_width)

        enriched_region_coord_list = []
        retest_enriched_region_coord_list = []
        if initial_enriched_region_coord_list:
            for coord_list in initial_enriched_region_coord_list:
                if coord_list[1]-coord_list[0]<=max_peak_width:
                    enriched_region_coord_list.append(coord_list)
                else:
                    retest_enriched_region_coord_list.append(coord_list)
            test_level = passed_tests_req+1
            while retest_enriched_region_coord_list:
                retest_enriched_region_coord_list_temp = []
                for coord_list in retest_enriched_region_coord_list:
                    higher_enrichment_coord_list = LoT.label_enriched_regions_threshold(test_counter_array[coord_list[0]:coord_list[1]], test_level, min_peak_width)
                    if higher_enrichment_coord_list:
                        for new_coord_list in higher_enrichment_coord_list:
                            if new_coord_list[1]-new_coord_list[0]<=max_peak_width:
                                enriched_region_coord_list.append([new_coord_list[0]+coord_list[0], new_coord_list[1]+coord_list[0]])
                            else:
                                retest_enriched_region_coord_list_temp.append([new_coord_list[0]+coord_list[0], new_coord_list[1]+coord_list[0]])
                    else:
                        enriched_region_coord_list.append(coord_list)
                retest_enriched_region_coord_list = retest_enriched_region_coord_list_temp
                test_level+=1
                
    else:
        max_peak_width = 2000
        coverage_array_smooth = bigwig_data.make_chrom_coverage_map(genome_stats_dict[chrom], smoothing=window)
        enriched_region_coord_list = LoT.label_enriched_regions_dynamic_threshold_width(coverage_array_smooth, genome_stats_dict[chrom]['chrom_mean']*initial_threshold, genome_stats_dict[chrom]['chrom_mean'], max_peak_width, min_region_size=min_peak_width)
        
    chrom_file_out = []
    if enriched_region_coord_list:
        coverage_array = bigwig_data.make_chrom_coverage_map(genome_stats_dict[chrom])/read_coverage_rphm
        X_wide_array, X_deep_array = LoT.extract_signal_wide_and_deep_chrom(coverage_array, enriched_region_coord_list, read_coverage_rphm)
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
        pyBigWig_input = pyBigWig.open(control_file)
        for i, coord_pair in enumerate(enriched_region_coord_list):
            average_cov = coverage_array[coord_pair[0]:coord_pair[1]].mean()*read_coverage_rphm
            pvalue_input_1 = calculate_pvalue_from_input(chrom, coord_pair[0], coord_pair[1], read_coverage_total, read_coverage_total_input, pyBigWig_input, average_cov, genome_stats_dict[chrom]['chrom_len'])
            pvalue_input_2 = calculate_pvalue_from_input(chrom, coord_pair[0]-500, coord_pair[1]+500, read_coverage_total, read_coverage_total_input, pyBigWig_input, average_cov, genome_stats_dict[chrom]['chrom_len'])
            if pvalue_input_1<=pvalue_input_2:
                pvalue_input = pvalue_input_1
            else:
                pvalue_input = pvalue_input_2
            out_list = [chrom, coord_pair[0], coord_pair[1], model_classifications[0][i][0], model_classifications[1][i][0], model_classifications[2][i][0], pvalue_input]
            X_wide_list = X_wide_array[i][:-1].tolist()
            out_list+=X_wide_list
            chrom_file_out.append(out_list)
        pyBigWig_input.close()
        with open(merge_folder+chrom+'.bed', 'w', newline='') as f:
            #f.write('chrom\tstart\tend\toverall_peak_score\tshape_score\tpvalue_score\tpvalue_input\tpvalue_chrom\tpvalue_10kb\tpvalue_20kb\tpvalue_30kb\tpvalue_40kb\tpvalue_50kb\tpvalue_60kb\tpvalue_70kb\tpvalue_80kb\tpvalue_90kb\tpvalue_100kb\n')
            bed_writer = csv.writer(f, delimiter='\t')
            bed_writer.writerows(chrom_file_out)


pyBigWig_object = pyBigWig.open(bigwig_file)
read_coverage_total = pyBigWig_object.header()['sumData']
read_coverage_rphm = read_coverage_total/read_coverage_factor
pyBigWig_object.close()

pyBigWig_input = pyBigWig.open(control_file)
read_coverage_total_input = pyBigWig_input.header()['sumData']
read_coverage_rphm_input = read_coverage_total_input/read_coverage_factor
pyBigWig_input.close()

bigwig_data = LoT.Bigwig_data(bigwig_file)
genome_stats_dict = bigwig_data.get_genome_info(include_special_chromosomes=False)

chrom_list = []
for chrom in genome_stats_dict:
    chrom_list.append([chrom, genome_stats_dict[chrom]['chrom_len']])
chrom_list = sorted(chrom_list, key=lambda chrom: chrom[1], reverse=True)

merge_folder = '{}merge/'.format(out_folder)
if not os.path.exists(merge_folder):
    os.makedirs(merge_folder)

cores = 1 
    
Parallel(n_jobs=cores)(delayed(classify_chrom)(chrom[0], genome_stats_dict[chrom[0]], read_coverage_rphm, read_coverage_rphm_input, bigwig_file, control_file, window, initial_threshold, min_peak_width, base_model_file, merge_folder) for chrom in chrom_list)
            
with open(out_folder+'complete.txt', 'w') as f:
    f.write('')

