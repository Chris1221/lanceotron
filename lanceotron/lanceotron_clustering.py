#!/package/python3-base/3.8.3/bin/python3.8

import argparse
import os

parser = argparse.ArgumentParser(description = 'Perform clustering on bigWig signal at coordinates given by a bed file')
parser.add_argument('file', help='bigwig file')
parser.add_argument('-b', '--bed', type=str, help='bed file used')
parser.add_argument('-f', '--folder', type=str, help='folder in public directory for writing')
parser.add_argument('-c', '--cluster', type=str, help='comma separated list of clustering techniques to calculate')
args = parser.parse_args()

features = 2000
bigwig_file = args.file
bed_file = args.bed
out_folder = args.folder
cluster_type_str = args.cluster

os.environ['NUMBA_CACHE_DIR'] = out_folder

import lanceotron_classic as LoTron
import numpy as np
import pyBigWig
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

# tsne and umap work better with raw signal, while pca performs better with standardized signal
def create_clusters(signal_array, cluster_type):
    print(cluster_type)
    if cluster_type == 'pca':
        standard_scaler = StandardScaler()
        signal_array_norm = standard_scaler.fit_transform(signal_array)
        cluster_array = PCA(n_components=2, random_state=42).fit_transform(signal_array_norm)
    elif cluster_type == 'tsne':
        cluster_array = TSNE(n_components=2, random_state=42, n_jobs=1).fit_transform(signal_array)
    elif cluster_type == 'umap':
        cluster_array = UMAP(n_components=2, random_state=42, low_memory=False).fit_transform(signal_array)
    return cluster_array

# extract signal
bed_list = LoTron.bed_file_to_list(bed_file)
coverage_array = np.zeros((len(bed_list), features))
bigwig_data = LoTron.Bigwig_data(bigwig_file)
genome_stats_dict = bigwig_data.get_genome_info()
pyBigWig_object = pyBigWig.open(bigwig_file)
for i, bed_entry in enumerate(bed_list):
    if bed_entry[0] in genome_stats_dict:
        region_length = bed_entry[2]-bed_entry[1]
        region_start = bed_entry[1]+int(region_length/2)-int(features/2)
        if (region_start>=0) and (region_start+features<=genome_stats_dict[bed_entry[0]]['chrom_len']):
            coverage_array[i] = np.nan_to_num(np.array(pyBigWig_object.values(bed_entry[0], region_start, region_start+features)))
        else:
            print('region too near to chromosome start/end to extract signal')
pyBigWig_object.close()

# reduce dimensions
cluster_type_list = cluster_type_str.lower().split(',')
cluster_dict = {}
for cluster_type in cluster_type_list:
    cluster_dict[cluster_type] = create_clusters(coverage_array, cluster_type)

# write results
with open('{}clustering.tsv'.format(out_folder), 'w') as f:
    for i, r in enumerate(bed_list):
        for cluster_type in cluster_type_list:
            for dim in cluster_dict[cluster_type][i]:
                f.write('{}\t'.format(dim))
        f.write('\n')
