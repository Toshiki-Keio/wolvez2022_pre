import time
import os
for i in range (24):
    os.system('git add .')
    os.system('git commit -m "ytpc2019a"')
    os.system('git push origin spm')
    time.sleep(1800)