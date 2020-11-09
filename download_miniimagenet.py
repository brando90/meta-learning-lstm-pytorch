#!/home/miranda9/.conda/envs/automl-meta-learning_wmlce-v1.7.0-py3.7/bin/python3.7
#SBATCH --job-name="miranda9job"
#SBATCH --output="demo.%j.%N.out"
#SBATCH --error="demo.%j.%N.err"
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=8
#SBATCH --threads-per-core=4
#SBATCH --mem-per-cpu=1200
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --mail-user=brando.science@gmail.com
#SBATCH --mail-type=ALL

import time
import os

from pathlib import Path


def report_times(start, verbose=False):
    '''
    How much time has passed since the time "start"
    :param float start: the number representing start (usually time.time())
    '''
    meta_str = ''
    ## REPORT TIMES
    start_time = start
    seconds = (time.time() - start_time)
    minutes = seconds / 60
    hours = minutes / 60
    if verbose:
        print(f"--- {seconds} {'seconds ' + meta_str} ---")
        print(f"--- {minutes} {'minutes ' + meta_str} ---")
        print(f"--- {hours} {'hours ' + meta_str} ---")
        print('\a')
    ##
    msg = f'time passed: hours:{hours}, minutes={minutes}, seconds={seconds}'
    return msg, seconds, minutes, hours


def download_and_extract_miniimagenet(path):
    """ Function to download miniImagent from google drive link.
    sources:
    - https://drive.google.com/file/d/1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR
    - https://github.com/markdtw/meta-learning-lstm-pytorch
    """
    import os
    from torchvision.datasets.utils import download_file_from_google_drive, extract_archive

    path = path.expanduser()

    file_id = '1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR'
    filename_zip = 'miniImagenet.tgz'
    # if zip not there re-download it
    path_2_zip = path / filename_zip
    if not path_2_zip.exists():
        download_file_from_google_drive(file_id, path, filename_zip)
    # if actual data is not in appriopriate location extract it from zip to location
    if not (path / 'miniImagenet').exists():
        os.system(f'tar -xvzf {path_2_zip} -C {path}/')  # extract data set in above location


if __name__ == "__main__":
    start = time.time()
    print('-> starting Downlooad')

    # dir to place mini-imagenet
    path = Path('~/data/miniimagenet_meta_lstm/').expanduser()
    download_and_extract_miniimagenet(path)

    print('--> DONE')
    time_passed_msg, _, _, _ = report_times(start)
    print(f'--> {time_passed_msg}')