__author__ = "Christine Kiran Kaushal"
__email__ = "christine.kiran@gmail.com"

from pathlib import Path
import subprocess


experiment_folders = Path('.\\experiment_folders\\paper\\')
experiments = ['PETCT_petct_windowing_c32_w220_basic_f1_adam',
'PETDPCT_petdpct_windowing_c70_w300_basic_f1_adam'
]

cv_folders = [str(split_folder) for split_folder in experiment_folders.glob('cv*')]

for exp in experiments:
    for cv in cv_folders:
        print(cv)
        experiment = f'.\\{cv}\\{exp}'
        base_path = Path(f'.\\{cv}\\{exp}')
        name = f'{exp}_{cv[-3:]}'
        subprocess.run(['python', 'run_experiment.py', experiment, '5000', '--eval', 'dice', '--name', name])
        
# ===========================================================================
# Notify when the program is done (requires a Twilio account):
from twilio.rest import Client
accountSID = 'ACc8ddc4aed0debd304e9e25ce2611af1b'
authToken = '3904817de24ea234a444959f1e220e5b'
client = Client(accountSID, authToken)

Twilionumber = '+4759447913'
cellnumber = '+4798052085'

message = client.messages.create(to=cellnumber, from_=Twilionumber, 
                                body=f'CV-running has stopped/is finished.')

subprocess.run(['python', 'overview.py', '--cv', True])
