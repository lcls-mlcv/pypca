#!/usr/bin/env python

import argparse
from edit_config import editConfig
import subprocess
import sys

def setupParserOptions():
    # TODO: Add help messages.
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config_file',
                        required=True,
                        help='Path to config file.')
    parser.add_argument('-t', '--task', type=str, help='Task to run.')

    ## shell script options
    parser.add_argument('-f',
                        '--facility',
                        type=str,
                        help='Facility where we are running at')
    parser.add_argument('-q',
                        '--queue',
                        type=str,
                        help='Queue to use on SLURM')
    parser.add_argument('-n', '--ncores', type=str, help='Number of cores')
    parser.add_argument('-e',
                        '--experiment_name',
                        type=str,
                        help='Experiment Name')
    parser.add_argument('-r', '--run_number', type=str, help='Run Number')

    ################ Optional below ####################
    ## setup
    parser.add_argument('--setup+queue', type=str, help='')
    parser.add_argument('--setup+root_dir', type=str, help='')
    parser.add_argument('--setup+exp', type=str, help='')
    parser.add_argument('--setup+run', type=int, help='')
    parser.add_argument('--setup+det_type', type=str, help='')
    parser.add_argument('--setup+cell', type=str, help='')
    parser.add_argument('--setup+event_receiver', type=str, help='')
    parser.add_argument('--setup+event_code', type=int, help='')
    parser.add_argument('--setup+event_logic', type=str, help='')

    ## fetch_mask
    parser.add_argument('--fetch_mask+dataset', type=str, help='')

    ## fetch_geom

    ## build_mask
    parser.add_argument('--build_mask+thresholds', type=str, help='')
    parser.add_argument('--build_mask+n_images', type=int, help='')
    parser.add_argument('--build_mask+n_edge', type=int, help='')
    parser.add_argument('--build_mask+combine',
                        action=argparse.BooleanOptionalAction,
                        help='')

    ## run_analysis
    parser.add_argument('--run_analysis+max_events', type=int, help='')

    ## opt_geom
    parser.add_argument('--opt_geom+n_iterations', type=int, help='')
    parser.add_argument('--opt_geom+n_peaks', type=int, help='')
    parser.add_argument('--opt_geom+threshold', type=int, help='')
    parser.add_argument('--opt_geom+center', type=str, help='')

    ## find_peaks
    parser.add_argument('--find_peaks+tag', type=str, help='')
    parser.add_argument('--find_peaks+psana_mask', type=int, help='')
    parser.add_argument('--find_peaks+min_peaks', type=int, help='')
    parser.add_argument('--find_peaks+max_peaks', type=int, help='')
    parser.add_argument('--find_peaks+npix_min', type=int, help='')
    parser.add_argument('--find_peaks+npix_max', type=int, help='')
    parser.add_argument('--find_peaks+amax_thr', type=float, help='')
    parser.add_argument('--find_peaks+atot_thr', type=float, help='')
    parser.add_argument('--find_peaks+son_min', type=float, help='')
    parser.add_argument('--find_peaks+peak_rank', type=int, help='')
    parser.add_argument('--find_peaks+r0', type=float, help='')
    parser.add_argument('--find_peaks+dr', type=float, help='')
    parser.add_argument('--find_peaks+nsigm', type=float, help='')

    ## index
    parser.add_argument('--index+time', type=str, help='')
    parser.add_argument('--index+ncores', type=int, help='')
    parser.add_argument('--index+tag_cxi', type=str, help='')
    parser.add_argument('--index+tag', type=str, help='')
    parser.add_argument('--index+int_radius', type=str, help='')
    parser.add_argument('--index+methods', type=str, help='')
    parser.add_argument('--index+cell', type=str, help='')
    parser.add_argument('--index+tolerance', type=str, help='')
    parser.add_argument('--index+no_revalidate',
                        action=argparse.BooleanOptionalAction,
                        help='')
    parser.add_argument('--index+multi',
                        action=argparse.BooleanOptionalAction,
                        help='')
    parser.add_argument('--index+profile',
                        action=argparse.BooleanOptionalAction,
                        help='')

    ## stream_analysis
    parser.add_argument('--stream_analysis+tag', type=str, help='')

    ## determine_cell
    parser.add_argument('--determine_cell+tag', type=str, help='')

    ## merge
    parser.add_argument('--merge+tag', type=str, help='')
    parser.add_argument('--merge+symmetry', type=str, help='')
    parser.add_argument('--merge+iterations', type=int, help='')
    parser.add_argument('--merge+model', type=str, help='')
    parser.add_argument('--merge+foms', type=str, help='')
    parser.add_argument('--merge+nshells', type=int, help='')
    parser.add_argument('--merge+highres', type=float, help='')

    ## solve
    parser.add_argument('--solve+tag', type=str, help='')
    parser.add_argument('--solve+pdb', type=str, help='')

    ## refine_center
    parser.add_argument('--refine_center+runs', type=str, help='')
    parser.add_argument('--refine_center+dx', type=str, help='')
    parser.add_argument('--refine_center+dy', type=str, help='')

    ## refine_distance
    parser.add_argument('--refine_distance+runs', type=str, help='')
    parser.add_argument('--refine_distance+dz', type=str, help='')

    ## elog_display

    args = vars(parser.parse_args())
    return args


def parseShellOptions(args):
    s = ''
    for key in args:
        if args[key] == None:
            pass        
        elif key == 'task' or key == 'config_file' or key == 'facility' or key == 'queue' \
            or key == 'experiment_name' or key == 'n_cores' or key == 'run_number':
            s += ' --%s %s' %(key, args[key])
        else:
            pass
    return s



if __name__ == '__main__':
    args = setupParserOptions()
    editConfig(args)
    args['config_file'] = args['config_file'][:-5] + '-tmp' + '.yaml'
    cmd = 'elog_submit.sh' + parseShellOptions(args)
    print(cmd)
    subprocess.run(cmd, shell=True)
