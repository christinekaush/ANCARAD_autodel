import subprocess


experiments = [
    'experiment_folders\paper\\cv1\\' + s for s in [

        ## Experiments to evaluate modalities on PET/CT/MRI dataset (36 patients):
        #'ADC_adc_basic_f1_adam',
        #'CT_ct_windowing_c32_w220_basic_f1_adam',
        #'Perf_perf_basic_f1_adam',
        #'PETCT_petct_windowing_c32_w220_basic_f1_adam',
        #'PET_pet_basic_f1_adam',
        #'DPCT_dpct_windowing_c70_w300_basic_f1_adam',
        #'PETDPCT_petdpct_windowing_c70_w300_basic_f1_adam',
        #'T2W_t2w_basic_f1_adam',
        #'T2WADC_t2wadc_basic_f1_adam',
        #'T2WADCPerf_t2wadcperf_basic_f1_adam'

        ## Augmentation experiments:
        #'T2W_t2w_basic_f1_adam_aug',
        #'DPCT_dpct_windowing_c70_w300_basic_f1_adam_aug',
        #'PETCT_petct_windowing_c32_w220_basic_f1_adam_aug',
        #'PETDPCT_petdpct_windowing_c70_w300_basic_f1_adam_aug',
        #'CT_ct_windowing_c32_w220_basic_f1_adam_aug',

        ## Experiments on PET/CT dataset (85 patients):
        #'PET_pet_basic_f1_adam_85',
        #'CT_ct_windowing_c32_w220_basic_f1_adam_85',
        #'DPCT_dpct_windowing_c70_w300_basic_f1_adam_85',
        #'PETCT_petct_windowing_c32_w220_basic_f1_adam_85',
        #'PETDPCT_petdpct_windowing_c70_w300_basic_f1_adam_85',

        ## Experiments on PET/CT dataset (85 patients) with augmentation:
        #'PET_pet_basic_f1_adam_85_aug',
        #'CT_ct_windowing_c32_w220_basic_f1_adam_85_aug',
        #'DPCT_dpct_windowing_c70_w300_basic_f1_adam_85_aug'
        #'PETCT_petct_windowing_c32_w220_basic_f1_adam_85_aug',
        #'PETDPCT_petdpct_windowing_c70_w300_basic_f1_adam_85_aug'

        # Additional experiments for paper:
        #'PETDPCTT2W_petdpctt2w_windowing_c70_w300_basic_f1_adam',
        #'PETCTT2W_petctt2w_windowing_c32_w220_basic_f1_adam',
        'CTT2W_ctt2w_windowing_c32_w220_basic_f1_adam',
        #'PETT2W_pett2w_basic_f1_adam'
    ]
]


from pathlib import Path

base_path = Path('.\experiment_folders\paper\\cv1\\')
all_experiments = [str(i) for i in base_path.glob('*')]

for experiment in experiments:
    subprocess.run(['python', 'run_experiment.py', experiment, '5000', '--eval', 'dice'])

subprocess.run(['python', 'run_cv.py'])

