import argparse
from .genome import run_genome

import pkg_resources

def genome():

    # Model file path to set as a default
    model_path = pkg_resources.resource_filename('lanceotron.static', 'wide_and_deep_fully_trained_v5_03.h5')

    """CLI Interface for Lance-o-Tron genome peak caller without an input track for reference. See lanceotron-genome 
    """
    parser = argparse.ArgumentParser(description='Sort significantly enriched regions of ChIP-seq singnals using a CNN')
    
    parser.add_argument('file', help='bigwig file')
    parser.add_argument('-f', '--folder', type=str, help='folder to write results to')
    parser.add_argument('-t', '--threshold', type=float, default = 4, help='threshold used for selecting candidate peaks')
    parser.add_argument('-w', '--window', type=int, default = 400, help='window size for rolling mean to use for selecting candidate peaks')
    parser.add_argument('-m', '--model', type=str, default = model_path, help='Deep learning model to classify candidate peaks')
    parser.add_argument('-g', '--grid', type=str, default = False, help='grid search across all parameters')
    parser.add_argument('-c', '--cores', type=int, default=1, help='number of cores to use in parallel')
    parser.add_argument('-p', '--pipeline', type=bool, default=False, help='True/False - if true will merge output into single bedfile')

    args = parser.parse_args()
    run_genome(args)