import argparse
import subprocess
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', choices=['bnn', 'cartpole', 'cc18', 'countingones', 'nas101', 'nas1shot1', 'nas201', 'svm', 'paramnet'], default='svm')
parser.add_argument('--ablation', choices=['F', 'Cr'], default='F')

args = parser.parse_args()

program = None
if args.benchmark == 'bnn':
    program = 'dehb/examples/bnn/scripts/run_bnn_dehb_ablation.sh'
elif args.benchmark == 'cartpole':
    program = 'dehb/examples/cartpole/scripts/run_cartpole_dehb_ablation.sh'
elif args.benchmark == 'cc18':
    program = 'dehb/examples/cc18/scripts/run_cc18_dehb_ablation.sh'
elif args.benchmark == 'countingones':
    program = 'dehb/examples/countingones/scripts/run_counting_dehb_ablation.sh'
elif args.benchmark == 'nas101':
    program = 'dehb/examples/nas101/scripts/run_nas101_dehb_ablation.sh'
elif args.benchmark == 'nas1shot1':
    program = 'dehb/examples/nas1shot1/scripts/run_nas1shot1_dehb_ablation.sh'
elif args.benchmark == 'nas201':
    program = 'dehb/examples/nas201/scripts/run_nas201_dehb_ablation.sh'
elif args.benchmark == 'paramnet':
    program = 'dehb/examples/paramnet/scripts/run_paramnet_dehb_ablation.sh'
else:
    program = 'dehb/examples/svm/scripts/run_svm_dehb_ablation.sh'



values = np.arange(start=0.1, stop=1, step=0.2)
values = [0.1, 0.3, 0.5, 0.7, 0.9]
params = []
for i in range(len(values)):
    F = 0.5 if args.ablation == 'Cr' else values[i]
    Cr = 0.5 if args.ablation == 'F' else values[i]

    print("\n\nAblation study on {} for F={} and Cr={}\n\n".format(args.benchmark, F, Cr))

    subprocess.call(['bash', program, str(F), str(Cr)])






