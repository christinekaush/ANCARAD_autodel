import subprocess
from tqdm import tqdm
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('log_folder')
parser.add_argument('--store_outputs', type=bool, help=('set equal to True if you want to store images, default: False'), default=False)
args = parser.parse_args()


log_folder = Path(args.log_folder)

for log in tqdm(log_folder.iterdir()):
    foldername = log.name
    experimentname = '_'.join(foldername.split('_')[:-1])
    experimentname = Path(r'experiment_folders\all_experiments')/experimentname
    experimentnum = foldername.split('_')[-1]
    try:
        if args.store_outputs:
            subprocess.run(['python', 'store_outputs.py', str(experimentname), experimentnum, 'dice', '--dataset', 'val', '--storefile', 'outputs'])
        else:
            subprocess.run(['python', 'store_outputs.py', str(experimentname), experimentnum, 'dice'])
        # python store_outputs.py experiment_folders\all_experiments\ADC_adc_aug_basic_f1_adam 00 dice --dataset val --storefile outputs
    except Exception:
        print('failed at ', log)
    