import os
import time

while True:
    os.system('git add .')
    os.system('git commit -m "from ytpc2019a"')
    os.system('git push origin spm')
    time.sleep(60)