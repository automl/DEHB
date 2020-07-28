import os
import numpy as np
from shutil import copyfile, move


dehbs = os.listdir()

for dehb in dehbs:
    if 'dehb' not in dehb or 'scipy' not in dehb:
        continue
    for ssp in ['1', '2', '3']:
        os.makedirs(os.path.join(dehb, ssp), exist_ok='True')
        for i in range(500):
            try:
                move(os.path.join(dehb, 'DEHB_{}_ssp_{}_seed_{}.obj'.format(i, ssp, i)),
                     os.path.join(dehb, ssp, 'DEHB_{}_ssp_{}_seed_{}.obj'.format(i, ssp, i)))
                # os.remove(os.path.join(dehb, 'DEHB_{}_ssp_{}_seed_{}.obj'.format(i, ssp, i)))
            except:
                print("Failing {}-th run for {} for ss{}".format(i, dehb, ssp))


