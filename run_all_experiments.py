import subprocess


experiments = [
    'experiment_folders/all_experiments/' + s for s in [
        'PETCT_petct_windowing_c32_w220_aug_basic_f1_adam', 
        'PETCT_petct_windowing_c32_w220_aug_basic_f1_adam_wo_drop', 
        'PETCT_petct_windowing_c32_w220_aug_basic_f1_adam_wo_rmd',
        'PETCT_petct_windowing_c32_w220_basic_f1_adam_woaug'
    ]
]

for experiment in experiments:
    subprocess.run(['python', 'run_experiment.py', experiment, '5000', '--eval', 'dice'])