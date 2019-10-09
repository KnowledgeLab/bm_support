import subprocess
from itertools import product

ns = [20]
# ns = [1]

thrs = [0, 9]
# thrs = [0]
# thrs = [9]

# classes = ['neutral', 'posneg']
classes = ['posneg']

modes = ['lr', 'rf']
# modes = ['rf']

for cc, mm, thr, nn in product(classes, modes, thrs, ns):
    if cc == 'neutral':
        oversample = True
    else:
        oversample = False
    str_run = ['python', 'predict_m.py',
               '-c', f'{cc}',
               '-m', f'{mm}',
               '-o', f'{oversample}',
               '-n', f'{nn}',
               '-t', f'{thr}',
               '-v', 'True',
               ]
    subprocess.run(str_run)
