from . import lanceotron as Ltron
from .utils import make_directory_name, build_model, calculate_pvalue_from_input
import os, sys
import numpy as np
import pyBigWig
import pickle
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import csv
import pkg_resources


def find_and_score_peaks(args):
    bigwig_file = args.file
    out_folder = make_directory_name(args.folder)
    initial_threshold = args.threshold
    window = args.window
    skip_header = args.skipheader

    min_peak_width = 50
    max_peak_width = 2000
    read_coverage_factor = 10**9
    out_file_name = bigwig_file.split('/')[-1].split('.')[0]+'_L-tron.bed'

    pyBigWig_object = pyBigWig.open(bigwig_file)
    read_coverage_total = pyBigWig_object.header()['sumData']
    read_coverage_rphm = read_coverage_total/read_coverage_factor
    pyBigWig_object.close()

    bigwig_data = Ltron.Bigwig_data(bigwig_file)
    genome_stats_dict = bigwig_data.get_genome_info()
    bed_file_out = []

    for chrom in genome_stats_dict:
        print(chrom)
        coverage_array_smooth = bigwig_data.make_chrom_coverage_map(genome_stats_dict[chrom], smoothing=window)
        enriched_region_coord_list = Ltron.label_enriched_regions_dynamic_threshold_width(coverage_array_smooth, genome_stats_dict[chrom]['chrom_mean']*initial_threshold, genome_stats_dict[chrom]['chrom_mean'], max_peak_width, min_region_size=min_peak_width)
        chrom_file_out = []
        if enriched_region_coord_list:
            wide_path = pkg_resources.resource_filename('lanceotron.static', 'standard_scaler_wide_v5_03.p')
            deep_path = pkg_resources.resource_filename('lanceotron.static', 'standard_scaler_deep_v5_03.p')

            coverage_array = bigwig_data.make_chrom_coverage_map(genome_stats_dict[chrom])/read_coverage_rphm
            X_wide_array, X_deep_array = Ltron.extract_signal_wide_and_deep_chrom(coverage_array, enriched_region_coord_list, read_coverage_rphm)
            standard_scaler_wide = pickle.load(open(wide_path, 'rb'))
            X_wide_array_norm = standard_scaler_wide.transform(X_wide_array)
            X_wide_array_norm = np.expand_dims(X_wide_array_norm, axis=2)
            standard_scaler = StandardScaler()
            X_deep_array_norm_T = standard_scaler.fit_transform(X_deep_array.T)
            standard_scaler_deep = pickle.load(open(deep_path, 'rb'))
            X_deep_array_norm = standard_scaler_deep.transform(X_deep_array_norm_T.T)
            X_deep_array_norm = np.expand_dims(X_deep_array_norm, axis=2)
            model = build_model()
            model_classifications = model.predict([X_deep_array_norm, X_wide_array_norm], verbose=1)
            K.clear_session()
            for i, coord_pair in enumerate(enriched_region_coord_list):
                out_list = [chrom, coord_pair[0], coord_pair[1], model_classifications[0][i][0], model_classifications[1][i][0], model_classifications[2][i][0]]
                X_wide_list = X_wide_array[i][:-1].tolist()
                X_wide_list = [100. if x>10 else x for x in X_wide_list]
                out_list+=X_wide_list
                chrom_file_out.append(out_list)
            bed_file_out+=chrom_file_out

    with open(out_folder+out_file_name, 'w', newline='') as f:
        if not skip_header:
            f.write('chrom\tstart\tend\toverall_peak_score\tshape_score\tenrichment_score\tpvalue_chrom\tpvalue_10kb\tpvalue_20kb\tpvalue_30kb\tpvalue_40kb\tpvalue_50kb\tpvalue_60kb\tpvalue_70kb\tpvalue_80kb\tpvalue_90kb\tpvalue_100kb\n')
        bed_writer = csv.writer(f, delimiter='\t')
        bed_writer.writerows(bed_file_out)

def call_peaks_with_input(args):
    bigwig_file = args.file
    control_file = args.input
    out_folder = make_directory_name(args.folder)
    initial_threshold = args.threshold
    window = args.window
    skip_header = args.skipheader

    min_peak_width = 50
    max_peak_width = 2000
    read_coverage_factor = 10**9
    out_file_name = bigwig_file.split('/')[-1].split('.')[0]+'_L-tron.bed'


    pyBigWig_object = pyBigWig.open(bigwig_file)
    read_coverage_total = pyBigWig_object.header()['sumData']
    read_coverage_rphm = read_coverage_total/read_coverage_factor
    pyBigWig_object.close()

    bigwig_data = Ltron.Bigwig_data(bigwig_file)
    genome_stats_dict = bigwig_data.get_genome_info()
    bed_file_out = []

    for chrom in genome_stats_dict:
        print(chrom)
        coverage_array_smooth = bigwig_data.make_chrom_coverage_map(genome_stats_dict[chrom], smoothing=window)
        enriched_region_coord_list = Ltron.label_enriched_regions_dynamic_threshold_width(coverage_array_smooth, genome_stats_dict[chrom]['chrom_mean']*initial_threshold, genome_stats_dict[chrom]['chrom_mean'], max_peak_width, min_region_size=min_peak_width)
        chrom_file_out = []
        if enriched_region_coord_list:

            wide_path = pkg_resources.resource_filename('lanceotron.static', 'standard_scaler_wide_v5_03.p')
            deep_path = pkg_resources.resource_filename('lanceotron.static', 'standard_scaler_deep_v5_03.p')

            coverage_array = bigwig_data.make_chrom_coverage_map(genome_stats_dict[chrom])/read_coverage_rphm
            X_wide_array, X_deep_array = Ltron.extract_signal_wide_and_deep_chrom(coverage_array, enriched_region_coord_list, read_coverage_rphm)
            standard_scaler_wide = pickle.load(open(wide_path, 'rb'))
            X_wide_array_norm = standard_scaler_wide.transform(X_wide_array)
            X_wide_array_norm = np.expand_dims(X_wide_array_norm, axis=2)
            standard_scaler = StandardScaler()
            X_deep_array_norm_T = standard_scaler.fit_transform(X_deep_array.T)
            standard_scaler_deep = pickle.load(open(deep_path, 'rb'))
            X_deep_array_norm = standard_scaler_deep.transform(X_deep_array_norm_T.T)
            X_deep_array_norm = np.expand_dims(X_deep_array_norm, axis=2)
            model = build_model()
            model_classifications = model.predict([X_deep_array_norm, X_wide_array_norm], verbose=1)
            pyBigWig_input = pyBigWig.open(control_file)
            read_coverage_total_input = pyBigWig_input.header()['sumData']
            read_coverage_rphm_input = read_coverage_total_input/read_coverage_factor
            K.clear_session()
            for i, coord_pair in enumerate(enriched_region_coord_list):
                average_cov = coverage_array[coord_pair[0]:coord_pair[1]].mean()*read_coverage_rphm
                pvalue_input = calculate_pvalue_from_input(chrom, coord_pair[0], coord_pair[1], read_coverage_total, read_coverage_total_input, pyBigWig_input, average_cov)
                out_list = [chrom, coord_pair[0], coord_pair[1], model_classifications[0][i][0], model_classifications[1][i][0], model_classifications[2][i][0], pvalue_input]
                X_wide_list = X_wide_array[i][:-1].tolist()
                X_wide_list = [100. if x>10 else x for x in X_wide_list]
                out_list+=X_wide_list
                chrom_file_out.append(out_list)
            pyBigWig_input.close()
            bed_file_out+=chrom_file_out

    with open(out_folder+out_file_name, 'w', newline='') as f:
        if not skip_header:
            f.write('chrom\tstart\tend\toverall_peak_score\tshape_score\tenrichment_score\tpvalue_input\tpvalue_chrom\tpvalue_10kb\tpvalue_20kb\tpvalue_30kb\tpvalue_40kb\tpvalue_50kb\tpvalue_60kb\tpvalue_70kb\tpvalue_80kb\tpvalue_90kb\tpvalue_100kb\n')
        bed_writer = csv.writer(f, delimiter='\t')
        bed_writer.writerows(bed_file_out)

def score_bed(args):
    """Score an existing BED file of peaks from a coverage track using Lanceotron's model.
    """
    bigwig_file = args.file
    out_folder = make_directory_name(args.folder)
    bed_file = args.bed
    skip_header = args.skipheader

    read_coverage_factor = 10**9
    out_file_name = bigwig_file.split('/')[-1].split('.')[0]+'_L-tron.bed'

    pyBigWig_object = pyBigWig.open(bigwig_file)
    read_coverage_total = pyBigWig_object.header()['sumData']
    read_coverage_rphm = read_coverage_total/read_coverage_factor
    pyBigWig_object.close()

    bed_list = Ltron.bed_file_to_list(bed_file)
    chroms_in_bed = []
    for bed_entry in bed_list:
        if bed_entry[0] not in chroms_in_bed:
            chroms_in_bed.append(bed_entry[0])

    bigwig_data = Ltron.Bigwig_data(bigwig_file)
    genome_stats_dict = bigwig_data.get_genome_info(include_special_chromosomes=True)
    bed_file_out = []

    for chrom in chroms_in_bed:
        print(chrom)
        enriched_region_coord_list = []
        for bed_entry in bed_list:
            if bed_entry[0]==chrom:
                enriched_region_coord_list.append([bed_entry[1], bed_entry[2]])
        chrom_file_out = []
        if enriched_region_coord_list:
            wide_path = pkg_resources.resource_filename('lanceotron.static', 'standard_scaler_wide_v5_03.p')
            deep_path = pkg_resources.resource_filename('lanceotron.static', 'standard_scaler_deep_v5_03.p')

            coverage_array = bigwig_data.make_chrom_coverage_map(genome_stats_dict[chrom])/read_coverage_rphm
            X_wide_array, X_deep_array = Ltron.extract_signal_wide_and_deep_chrom(coverage_array, enriched_region_coord_list, read_coverage_rphm)
            standard_scaler_wide = pickle.load(open(wide_path, 'rb'))
            X_wide_array_norm = standard_scaler_wide.transform(X_wide_array)
            X_wide_array_norm = np.expand_dims(X_wide_array_norm, axis=2)
            standard_scaler = StandardScaler()
            X_deep_array_norm_T = standard_scaler.fit_transform(X_deep_array.T)
            standard_scaler_deep = pickle.load(open(deep_path, 'rb'))
            X_deep_array_norm = standard_scaler_deep.transform(X_deep_array_norm_T.T)
            X_deep_array_norm = np.expand_dims(X_deep_array_norm, axis=2)
            model = build_model()
            model_classifications = model.predict([X_deep_array_norm, X_wide_array_norm], verbose=1)
            K.clear_session()
            for i, coord_pair in enumerate(enriched_region_coord_list):
                out_list = [chrom, coord_pair[0], coord_pair[1], model_classifications[0][i][0], model_classifications[1][i][0], model_classifications[2][i][0]]
                X_wide_list = X_wide_array[i][:-1].tolist()
                X_wide_list = [100. if x>10 else x for x in X_wide_list]
                out_list+=X_wide_list
                chrom_file_out.append(out_list)
            bed_file_out+=chrom_file_out

    with open(out_folder+out_file_name, 'w', newline='') as f:
        if not skip_header:
            f.write('chrom\tstart\tend\toverall_peak_score\tshape_score\tenrichment_score\tpvalue_chrom\tpvalue_10kb\tpvalue_20kb\tpvalue_30kb\tpvalue_40kb\tpvalue_50kb\tpvalue_60kb\tpvalue_70kb\tpvalue_80kb\tpvalue_90kb\tpvalue_100kb\n')
        bed_writer = csv.writer(f, delimiter='\t')
        bed_writer.writerows(bed_file_out)