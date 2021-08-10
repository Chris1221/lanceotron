import argparse
import sys

from .find_and_score_peaks import find_and_score_peaks
from .find_peaks_with_input import call_peaks_with_input

import pkg_resources

def genome():

    # Model file path to set as a default
    model_path = pkg_resources.resource_filename('lanceotron.static', 'wide_and_deep_fully_trained_v5_03.h5')

    """CLI Interface for Lance-o-Tron genome peak caller without an input track for reference. See lanceotron-genome 
    """
    parser = argparse.ArgumentParser(description='Sort significantly enriched regions of ChIP-seq singnals using a CNN')
    subparsers = parser.add_subparsers(help='sub-command help')

    """ Old version 
    callpeaks = subparsers.add_parser("callPeaks")
    callpeaks.add_argument('file', help='bigwig file')
    callpeaks.add_argument('-f', '--folder', type=str, help='folder to write results to')
    callpeaks.add_argument('-t', '--threshold', type=float, default = 4, help='threshold used for selecting candidate peaks')
    callpeaks.add_argument('-w', '--window', type=int, default = 400, help='window size for rolling mean to use for selecting candidate peaks')
    callpeaks.add_argument('-m', '--model', type=str, default = model_path, help='Deep learning model to classify candidate peaks')
    callpeaks.add_argument('-g', '--grid', type=str, default = False, help='grid search across all parameters')
    callpeaks.add_argument('-c', '--cores', type=int, default=1, help='number of cores to use in parallel')
    callpeaks.add_argument('-p', '--pipeline', type=bool, default=True, help='True/False - if true will merge output into single bedfile')
    callpeaks.set_defaults(func = run_genome)

    scorebed = subparsers.add_parser("scoreBed")
    scorebed.add_argument('file', help='bigwig file')
    scorebed.add_argument('-b', '--bed', type=str, help='bed file used')
    scorebed.add_argument('-f', '--folder', type=str, help='folder in public directory for writing')
    scorebed.add_argument('-m', '--model', type=str, default = model_path, help='Deep learning model to classify candidate peaks')
    scorebed.set_defaults(func = scoreBed)
    """

    findandscore = subparsers.add_parser("callPeaks")
    findandscore.add_argument('file', help='bigwig file')
    findandscore.add_argument('-t', '--threshold', type=float, default=4, help='initial threshold used for selecting candidate peaks; default=4')
    findandscore.add_argument('-w', '--window', type=int, default=400, help='window size for rolling mean to use for selecting candidate peaks; default=400')
    findandscore.add_argument('-f', '--folder', type=str, default='./', help='folder to write results to; default=current directory')
    findandscore.add_argument('--skipheader', default=False, action='store_true', help='skip writing header')
    findandscore.set_defaults(func = find_and_score_peaks)

    parser = argparse.ArgumentParser(description='Sort significantly enriched regions of ChIP-seq singnals using a CNN')
    fas_input = subparsers.add_parser("callPeaksInput")
    fas_input.add_argument('file', help='bigwig file')
    fas_input.add_argument('-i', '--input', type=str, help='control input track used to calculate Poisson-based significance of peaks')
    fas_input.add_argument('-t', '--threshold', type=float, default=4, help='initial threshold used for selecting candidate peaks; default=4')
    fas_input.add_argument('-w', '--window', type=int, default=400, help='window size for rolling mean to use for selecting candidate peaks; default=400')
    fas_input.add_argument('-f', '--folder', type=str, default='./', help='folder to write results to; default=current directory')
    fas_input.add_argument('--skipheader', default=False, action='store_true', help='skip writing header')
    fas_input.set_defaults(func = call_peaks_with_input)


    # Parse the arguments and quit if no file specified.
    # This is mainly to catch lanceotron being called with no additional args.
    args = parser.parse_args()
    if 'file' not in args:
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    args.func(args)