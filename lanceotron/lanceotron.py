import pyBigWig
import numpy as np
import csv
import math

def bed_file_to_list(bed_file):
    bed_list = []
    with open(bed_file, 'r') as f:
        bed_reader = csv.reader(f, delimiter='\t')
        for row in bed_reader:
            # Using a merged pipelined bed file
            if row[0].lower() in ["chrom", "chr"]:
                continue
            bed_list.append([row[0], int(row[1]), int(row[2]), row[3]])
    return bed_list

def average_array(array, window):
    window = int(window)
    if window > len(array):
        window = int(len(array)/10)
        print('averaging window larger than array, window adjusted to size {}'.format(window))
    return_array = np.cumsum(array,dtype=float)
    return_array[window:] = return_array[window:] - return_array[:-window]
    return_array /= window
    left_pad = int(window/2)
    right_pad = window-left_pad
    return_array[left_pad:-right_pad] = return_array[window:]
    return_array[:left_pad] = np.mean(array[:left_pad])
    return_array[-right_pad:] = np.mean(array[-right_pad:])
    return return_array

# check_endpoints looks for special cases where the start or end of a region are enriched
def check_endpoints(coord_array,first_element_enriched,start_pos,end_pos):
    coord_array = coord_array+1
    if len(coord_array)%2==0:
        if first_element_enriched==True:
            coord_array=np.insert(coord_array,0,start_pos)
            coord_array=np.append(coord_array,end_pos)
    else:
        if first_element_enriched==True:
            coord_array=np.insert(coord_array,0,start_pos)
        else:
            coord_array=np.append(coord_array,end_pos)
    return coord_array.reshape(-1,2)

# label_enriched_signal returns the coordinates of regions greater than some threshold;
def label_enriched_regions(array, threshold, min_peak_width=0):
    truth_array = array>threshold
    enriched_region_list = []
    if truth_array.any:
        coord_array = np.where(truth_array[:-1]!=truth_array[1:])[0]
        coord_array_checked = check_endpoints(coord_array, truth_array[0], 0, len(array))
        enriched_region_list=coord_array_checked.tolist()
    filtered_region_list = []
    for region_coordinates in enriched_region_list:
        if (region_coordinates[1]-region_coordinates[0])>=min_peak_width:
            filtered_region_list.append(region_coordinates)
    return filtered_region_list

def scale_up_entry(pileup_array, start_pos, end_pos, bins):
    bin_array = np.zeros(bins)
    if start_pos==end_pos:
        end_pos+=1
    region_size = end_pos-start_pos
    total_pad_len = 0
    while bins%(region_size+total_pad_len)!=0:
        total_pad_len+=1
    scale_up_factor = int(bins/(region_size+total_pad_len))
    left_pad_len = int(total_pad_len/2)
    right_pad_len = total_pad_len-left_pad_len
    signal_start = start_pos-left_pad_len
    signal_end = end_pos+right_pad_len
    if signal_start<0 or signal_end>len(pileup_array):
        print('entry too near edge of chromosome to extract signal:{}-{}'.format(start_pos, end_pos))
        return bin_array
    else:
        bin_array = np.repeat(pileup_array[signal_start:signal_end], scale_up_factor)
        return bin_array

def scale_down_entry(pileup_array, start_pos, end_pos, bins):
    bin_array = np.zeros(bins)
    region_size = end_pos-start_pos
    region_mod = region_size%bins
    if region_mod == 0:
        total_pad_len = 0
    else:
        total_pad_len = bins-region_mod
    scale_factor = int((region_size+total_pad_len)/bins)
    left_pad_len = int(total_pad_len/2)
    right_pad_len = total_pad_len-left_pad_len
    signal_start = start_pos-left_pad_len
    signal_end = end_pos+right_pad_len
    if signal_start<0 or signal_end>len(pileup_array):
        print('entry too near edge of chromosome to extract signal: {}-{}'.format(start_pos, end_pos))
    else:
        bin_array = pileup_array[signal_start:signal_end].reshape(bins, scale_factor).mean(axis=1)
    return bin_array
    
def extract_signal_legacy(pileup_array, coord_list, signal_mean, signal_std, bins=1000):
    signal_array = np.zeros((len(coord_list), bins, 3))
    for i, coord in enumerate(coord_list):
        start = coord[0]
        end = coord[1]
        region_size = end-start
        mid = int(region_size/2)+start
        mid_range = int(bins/2)
        long_range = mid_range*10
        if region_size<bins:
            signal_array[i,:,0] = scale_up_entry(pileup_array, start, end, bins)
        elif region_size==bins:
            if start<0 or end>len(pileup_array):
                print('entry too near edge of chromosome to extract signal: {}-{}'.format(start, end))
            else:
                signal_array[i,:,0] = pileup_array[start:end]
        else:
            signal_array[i,:,0] = scale_down_entry(pileup_array, start, end, bins)
        if (mid-mid_range)<0 or (mid+mid_range)>len(pileup_array):
                print('entry too near edge of chromosome to extract signal: {}-{}'.format(start, end))
        else:
            signal_array[i,:,1] = pileup_array[mid-mid_range:mid+mid_range]
        signal_array[i,:,2] = scale_down_entry(pileup_array, (mid-long_range), (mid+long_range), bins)
    signal_array = (signal_array-signal_mean)/signal_std
    return signal_array       

def find_lcm(a, b):
    return abs(a * b) / math.gcd(a,b) if a and b else 0

def extract_signal_2k(pileup_array, coord_list, signal_mean, signal_std, bins=2000):
    signal_array = np.zeros((len(coord_list), bins, 3))
    for i, coord in enumerate(coord_list):
        start = coord[0]-100
        end = coord[1]+100
        region_size = end-start
        mid = int(region_size/2)+start
        mid_range = int(bins/2)
        long_range = mid_range*10
        if (start-long_range)<0 or (end+long_range)>len(pileup_array):
            print('entry too near edge of chromosome to extract signal: {}-{}'.format(start, end))
        else:
            lcm = find_lcm(region_size, bins)
            repeat_factor = int(lcm/region_size)
            reshape_factor = int(lcm/bins)
            signal_array[i,:,0] = np.repeat(pileup_array[start:end], repeat_factor).reshape(bins, reshape_factor).mean(axis=1)
            signal_array[i,:,1] = pileup_array[mid-mid_range:mid+mid_range]
            signal_array[i,:,2] = pileup_array[mid-long_range:mid+long_range].reshape(bins, -1).mean(axis=1)
    signal_array = (signal_array-signal_mean)/signal_std
    return signal_array

def extract_signal_2knew(pileup_array, coord_list, signal_mean, signal_std, bins=2000):
    signal_array = np.zeros((len(coord_list), bins, 3))
    for i, coord in enumerate(coord_list):
        start = coord[0]
        end = coord[1]
        region_size = end-start
        mid = int(region_size/2)+start
        mid_range = int(bins/2)
        long_range = mid_range*10
        if (mid-long_range)<0 or (mid+long_range)>len(pileup_array):
            print('entry too near edge of chromosome to extract signal: {}-{}'.format(coord[0], coord[1]))
        else:
            if region_size>bins:
                start = mid-int(bins/2)
                end = start+bins
                signal_array[i,:,0] = np.nan_to_num(np.array(pyBigWig_object.values(chrom_name, mid-mid_range, mid+mid_range)))
            else:
                padding = bins-region_size
                left_pad = int(padding/2)
                signal_array[i,left_pad:left_pad+region_size,0] = np.nan_to_num(np.array(pyBigWig_object.values(chrom_name, start, end)))
            signal_array[i,:,1] = np.nan_to_num(np.array(pyBigWig_object.values(chrom_name, mid-mid_range, mid+mid_range)))
            signal_array[i,:,2] = np.nan_to_num(np.array(pyBigWig_object.values(chrom_name, mid-long_range, mid+long_range))).reshape(bins, -1).mean(axis=1)
    signal_array = (signal_array-signal_mean)/signal_std
    return signal_array

def extract_signal_single_2knew(pyBigWig_object, chrom_name, start, end, signal_length, signal_mean, signal_std, bins=2000):
    signal_array = np.zeros((1, bins, 3))
    region_size = end-start
    mid = int(region_size/2)+start
    mid_range = int(bins/2)
    long_range = mid_range*10
    if (mid-long_range)<0 or (mid+long_range)>signal_length:
        print('entry too near edge of chromosome to extract signal: {}-{}'.format(start, end))
    else:
        if region_size>bins:
            start = mid-int(bins/2)
            end = start+bins
            signal_array[0,:,0] = np.nan_to_num(np.array(pyBigWig_object.values(chrom_name, mid-mid_range, mid+mid_range)))
        else:
            padding = bins-region_size
            left_pad = int(padding/2)
            signal_array[0,left_pad:left_pad+region_size,0] = np.nan_to_num(np.array(pyBigWig_object.values(chrom_name, start, end)))
        signal_array[0,:,1] = np.nan_to_num(np.array(pyBigWig_object.values(chrom_name, mid-mid_range, mid+mid_range)))
        signal_array[0,:,2] = np.nan_to_num(np.array(pyBigWig_object.values(chrom_name, mid-long_range, mid+long_range))).reshape(bins, -1).mean(axis=1)
    signal_array = (signal_array-signal_mean)/signal_std
    return signal_array


def extract_signal(pileup_array, coord_list, signal_mean, signal_std, bins=1000):
    signal_array = np.zeros((len(coord_list), bins, 2))
    for i, coord in enumerate(coord_list):
        start = coord[0]
        end = coord[1]
        region_size = end-start
        mid = int(region_size/2)+start
        mid_range = int(bins/2)
        long_range = mid_range*10
        if (start-long_range)<0 or (end+long_range)>len(pileup_array):
            print('entry too near edge of chromosome to extract signal: {}-{}'.format(start, end))
        else:
            lcm = find_lcm(region_size, bins)
            repeat_factor = int(lcm/region_size)
            reshape_factor = int(lcm/bins)
            signal_array[i,:,0] = np.repeat(pileup_array[start:end], repeat_factor).reshape(bins, reshape_factor).mean(axis=1)
            signal_array[i,:,1] = pileup_array[mid-long_range:mid+long_range].reshape(bins, -1).mean(axis=1)
    signal_array = (signal_array-signal_mean)/signal_std
    return signal_array


def extract_signal_clustering(pyBigWig_object, chrom_name, start, end, signal_length, signal_mean, signal_std, bins=2000):
    signal_array = np.zeros(bins)
    region_size = end-start
    if region_size == 0:
        end+=1
        region_size+=1
    mid = int(region_size/2)+start
    mid_range = int(bins/2)
    if (mid-mid_range)<0 or (mid+mid_range)>signal_length:
        print('entry too near edge of chromosome to extract signal: {}-{}'.format(start, end))
    else:
        if region_size>bins:
            start = mid-int(bins/2)
            end = start+bins
            signal_array = np.nan_to_num(np.array(pyBigWig_object.values(chrom_name, mid-mid_range, mid+mid_range)))
        else:
            padding = bins-region_size
            left_pad = int(padding/2)
            print(chrom_name, start, end, left_pad, left_pad+region_size)
            signal_array[left_pad:left_pad+region_size] = np.nan_to_num(np.array(pyBigWig_object.values(chrom_name, start, end)))
    signal_array = (signal_array-signal_mean)/signal_std
    return signal_array
