#!/package/python3-base/3.8.3/bin/python3.8

from . import lanceotron
from . import classic as LoT
import numpy as np
import os
import pyBigWig
from joblib import Parallel, delayed
import csv
import pickle
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from tqdm import tqdm

# Pickle warnings are unnecessary. If it breaks, it breaks.
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# Running TF in CPU mode will generate warnings that I don't care about.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.config.threading import set_intra_op_parallelism_threads, set_inter_op_parallelism_threads

import pkg_resources

# TODO: Load time could be improved by explicitly naming imports.
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

def classify_chrom(chrom, bigWig_file, window, threshold, min_peak_width, base_model_file, out_folder):
    print(chrom)
    pileup_array=np.zeros(chrom[1])
    pyBigWig_object=pyBigWig.open(bigWig_file)
    for reads in pyBigWig_object.intervals('{}'.format(chrom[0])):
        pileup_array[reads[0]:reads[1]] = reads[2]
    chrom_mean = pyBigWig_object.stats(chrom[0], type='mean', exact=True)
    chrom_std = pyBigWig_object.stats(chrom[0], type='std', exact=True)
    pyBigWig_object.close()
    pileup_array_smooth = lanceotron.average_array(pileup_array, window)
    #mean_signal_10k = lanceotron.average_array(pileup_array, 10000)
    coord_list = lanceotron.label_enriched_regions(pileup_array_smooth, (chrom_mean[0]*threshold), min_peak_width)
    if base_model_file.endswith('baseModel_categories.h5') or base_model_file.endswith('custom.h5'):
        signal_array = lanceotron.extract_signal_legacy(pileup_array, coord_list, chrom_mean, chrom_std)
        model = load_model(base_model_file)
    elif base_model_file.endswith('trichannel.h5'):
        bins = 2000
        signal_array = lanceotron.extract_signal_2k(pileup_array, coord_list, chrom_mean, chrom_std)
        model = load_model(base_model_file)
    elif base_model_file.endswith('model_weights.h5'):
        from keras.models import Sequential
        from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
        from keras.optimizers import Adam
        bins = 2000
        signal_array = lanceotron.extract_signal_2k(pileup_array, coord_list, chrom_mean, chrom_std)
        model = Sequential()
        model.add(Conv1D(filters=4, kernel_size=2, activation='relu', input_shape=(bins, 3), padding='valid'))
        model.add(BatchNormalization(axis=1))
        model.add(Conv1D(filters=8, kernel_size=2, activation='relu', padding='valid'))
        model.add(BatchNormalization(axis=1))
        model.add(Conv1D(filters=16, kernel_size=2, activation='relu', padding='valid'))
        model.add(BatchNormalization(axis=1))
        model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding='valid'))
        model.add(BatchNormalization(axis=1))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='valid'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.25))
        model.add(Conv1D(filters=128, kernel_size=32, activation='relu', padding='valid'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(1000, activation="relu"))
        model.add(Dropout(0.50))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer=Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),loss='categorical_crossentropy', metrics=['accuracy'])
        model.load_weights('/t1-data/user/lhentges/trainingData/models/v3/ATAC_20/ATAC_20_model_weights.h5')
    else:
        signal_array = lanceotron.extract_signal(pileup_array, coord_list, chrom_mean, chrom_std)
        model = load_model(base_model_file)
    chrom_classifications = model.predict(signal_array, verbose=0)
    if type(chrom_classifications)==np.ndarray:  
        region_count, class_count = chrom_classifications.shape
        with open('{}{}.bed'.format(out_folder, chrom[0]), 'w') as f:
            for i, coordinate in enumerate(coord_list):
                f.write('{}\t{}\t{}'.format(chrom[0], coordinate[0], coordinate[1]))
                for j in range(class_count):
                    f.write('\t{:.5f}'.format(chrom_classifications[i][j]))
                f.write('\n')
    K.clear_session()

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



    model.load_weights(pkg_resources.resource_filename("lanceotron.static", "wide_and_deep_fully_trained_v5_03.h5"))
    return model

    
def classify_chrom_alt(chrom, chrom_stats_dict, read_coverage_factor, bigWig_file, window, threshold, min_peak_width, base_model_file, out_folder):
    
    pyBigWig_object = pyBigWig.open(bigWig_file)
    read_coverage_total = pyBigWig_object.header()['sumData']
    read_coverage_rphm = read_coverage_total/read_coverage_factor
    pyBigWig_object.close()
    bigwig_data = LoT.Bigwig_data(bigWig_file)
    
    if base_model_file == 'wide_and_deep_fully_trained_v5_03_grid_candidate.h5':
        window_list = [100, 200, 400, 800, 1600]
        threshold_list = [2, 4, 8, 16, 32]
        passed_tests_req = 5
        min_peak_width = 50
        max_peak_width = 800
        test_counter_array = np.zeros(chrom_stats_dict['chrom_len'])
        for window in window_list:
            coverage_array_smooth = bigwig_data.make_chrom_coverage_map(chrom_stats_dict, smoothing=window)
            for threshold in threshold_list:
                test_passed_coord_list = LoT.label_enriched_regions_threshold(coverage_array_smooth, chrom_stats_dict['chrom_mean']*threshold)
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
        coverage_array_smooth = bigwig_data.make_chrom_coverage_map(chrom_stats_dict, smoothing=window)
        enriched_region_coord_list = LoT.label_enriched_regions_dynamic_threshold_width(coverage_array_smooth, chrom_stats_dict['chrom_mean']*threshold, chrom_stats_dict['chrom_mean'], max_peak_width, min_region_size=min_peak_width)
    
    if enriched_region_coord_list:
        # Retrieve the scaler values from local static directory.

        wide_path = pkg_resources.resource_filename('lanceotron.static', 'standard_scaler_wide_v5_03.p')
        deep_path = pkg_resources.resource_filename('lanceotron.static', 'standard_scaler_deep_v5_03.p')


        standard_scaler_wide = pickle.load(open(wide_path, 'rb'))
        standard_scaler_deep = pickle.load(open(deep_path, 'rb'))

        coverage_array = bigwig_data.make_chrom_coverage_map(chrom_stats_dict)/read_coverage_rphm
        X_wide_array, X_deep_array = LoT.extract_signal_wide_and_deep_chrom(coverage_array, enriched_region_coord_list, read_coverage_rphm)
        X_wide_array_norm = standard_scaler_wide.transform(X_wide_array)
        X_wide_array_norm = np.expand_dims(X_wide_array_norm, axis=2)
        standard_scaler = StandardScaler()
        X_deep_array_norm_T = standard_scaler.fit_transform(X_deep_array.T)
        X_deep_array_norm = standard_scaler_deep.transform(X_deep_array_norm_T.T)
        X_deep_array_norm = np.expand_dims(X_deep_array_norm, axis=2)
        model = build_model()
        model_classifications = model.predict([X_deep_array_norm, X_wide_array_norm], verbose=0)
        K.clear_session()
    chrom_file_out = []
    for i, coord_pair in enumerate(enriched_region_coord_list):
        out_list = [chrom[0], coord_pair[0], coord_pair[1], model_classifications[0][i][0], model_classifications[1][i][0], model_classifications[2][i][0], model_classifications[0][i][1]]
        X_wide_list = X_wide_array[i][:-1].tolist()
        out_list+=X_wide_list
        chrom_file_out.append(out_list)
    with open('{}{}.bed'.format(out_folder, chrom[0]), 'w', newline='') as f:
        bed_writer = csv.writer(f, delimiter='\t')
        bed_writer.writerows(chrom_file_out)
    
def classify_chrom_grid_search(chrom, bigWig_file, window_list, threshold_list, min_peak_width, base_model_file, out_folder):
    print(chrom)
    model = load_model(base_model_file)
    pileup_array=np.zeros(chrom[1])
    pyBigWig_object=pyBigWig.open(bigWig_file)
    if pyBigWig_object.intervals('{}'.format(chrom[0]))!=None:
        for reads in pyBigWig_object.intervals('{}'.format(chrom[0])):
            pileup_array[reads[0]:reads[1]] = reads[2]
        chrom_mean = pyBigWig_object.stats(chrom[0], type='mean', exact=True)
        chrom_std = pyBigWig_object.stats(chrom[0], type='std', exact=True)
        pyBigWig_object.close()
        for window in window_list:
            pileup_array_smooth = lanceotron.average_array(pileup_array, window)
            for threshold in threshold_list:
                print('{} {} {}: finding enriched regions'.format(chrom[0], window, threshold))
                coord_list = lanceotron.label_enriched_regions(pileup_array_smooth, (chrom_mean[0]*threshold), min_peak_width)
                print('{} {} {}: {} regions found'.format(chrom[0], window, threshold, len(coord_list)))
                for i in range(0, len(coord_list), batch_size):
                    signal_array = lanceotron.extract_signal(pileup_array, coord_list[i:i+batch_size], chrom_mean, chrom_std)
                    chrom_classifications = model.predict(signal_array, verbose=0)
                    if type(chrom_classifications)==np.ndarray:  
                        region_count, class_count = chrom_classifications.shape
                        with open('{}{}.bed'.format(out_folder, chrom[0]), 'a') as f:
                            for j, coordinate in enumerate(coord_list[i:i+batch_size]):
                                f.write('{}\t{}\t{}'.format(chrom[0], coordinate[0], coordinate[1]))
                                for k in range(class_count):
                                    f.write('\t{:.5f}'.format(chrom_classifications[j][k]))
                                f.write('\n')
        print('{} done.'.format(chrom[0]))
    else:
        pyBigWig_object.close()
    K.clear_session()


def run_genome(args):
    """Runs the LanceOTron peak caller without a reference input track.

    Args:
        args (Namespace): Results from argparse.ArgumentParser.parse_args (alternatively a dictionary of keywork arguments will work for testing.)
    """

    bigWig_file = args.file
    threshold = args.threshold
    window = args.window
    base_model_file = args.model
    out_folder = args.folder
    grid_search = args.grid
    min_peak_width = 50
    batch_size = 10000
    bins = 1000
    read_coverage_factor = 10**9
    cores = args.cores
    pipeline = args.pipeline

    # This is now an option on the command line
    # so setting it is explicit.
    #print('reserving {} core(s) on cluster'.format(cores))

    window_list = [100, 200, 400, 800, 1600]
    threshold_list = [1, 2, 4, 8, 16]

    set_intra_op_parallelism_threads(1)
    set_inter_op_parallelism_threads(1)

    chrom_list = []
    threshold_list = []
    pyBigWig_object = pyBigWig.open(bigWig_file)
    for chrom_name in pyBigWig_object.chroms():
        if (not chrom_name.startswith('chrUn')) and ('_' not in chrom_name) and (chrom_name!='chrM') and (chrom_name!='chrEBV'):
            chrom_mean = pyBigWig_object.stats(chrom_name, type='mean', exact=True)[0]
            if chrom_mean!=None:
                chrom_list.append([chrom_name, pyBigWig_object.chroms(chrom_name)])
                threshold_list.append([chrom_name, threshold*chrom_mean])
    pyBigWig_object.close()
    chrom_list = sorted(chrom_list, key=lambda chrom: chrom[1], reverse=True)
    # print(chrom_list)

    merge_folder = '{}merge/'.format(out_folder)
    if not os.path.exists(merge_folder):
        os.makedirs(merge_folder)

    #cores = 16
    cores = 1 

    chrom_iterator = tqdm(chrom_list, desc = "Calling chromosomes")
        
    if grid_search=='True':
        Parallel(n_jobs=cores)(delayed(classify_chrom_grid_search)(chrom, bigWig_file, window_list, threshold_list, min_peak_width, base_model_file, merge_folder) for chrom in chrom_iterator)
    # This string will never start with the name, it's sufficient that it should contain the model name.
    elif 'wide_and_deep_fully_trained_v5' in base_model_file:
        # Also now a command line option, no need for output.
        #print('alt signal extraction and model')
        bigwig_data = LoT.Bigwig_data(bigWig_file)
        genome_stats_dict = bigwig_data.get_genome_info(include_special_chromosomes=False)
        Parallel(n_jobs=cores)(delayed(classify_chrom_alt)(chrom, genome_stats_dict[chrom[0]], read_coverage_factor, bigWig_file, window, threshold, min_peak_width, base_model_file, merge_folder) for chrom in chrom_iterator)
    else:
        Parallel(n_jobs=cores)(delayed(classify_chrom)(chrom, bigWig_file, window, threshold, min_peak_width, base_model_file, merge_folder) for chrom in chrom_iterator)


    with open('{}chrom_enrichment_thresholds.txt'.format(out_folder), 'w') as f:
        f.write('chromosome\tsignal_threshold\n')
        for chrom_name, threshold in threshold_list:
            f.write('{}\t{}\n'.format(chrom_name, threshold))

    with open('{}complete.txt'.format(out_folder), 'w') as f:
        f.write('')

    if pipeline:
        from glob import glob
        import natsort
        # This doesn't really work with paths
        #filename_split = bigWig_file.split('.')
        # Take the file name portion of the bigWig file and use it to name the bed.
        filename = Path(bigWig_file).stem
        with open('{}{}.bed'.format(out_folder, filename), 'w') as singleFile:
            singleFile.write('chrom\tstart\tend\tH3K4me1_score\tnoise_score\tATAC_score\tH3K4me3_score\tTF_score\tH3K27ac_score\n')
        with open('{}{}.bed'.format(out_folder, filename), 'a') as singleFile:
            for csvFile in natsort.natsorted(glob('{}*.bed'.format(merge_folder))):
                for line in open(csvFile, 'r'):
                    singleFile.write(line)