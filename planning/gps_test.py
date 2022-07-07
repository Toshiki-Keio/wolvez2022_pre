import time
import gps

class cansat(object):
    def __init__(self):
        self.gps = gps.GPS()

    def setup(self):
        self.gps.setupGps()
        
    def writeData(self):
        self.gps.gpsread()
        timer = 1000*(time.time() - start_time)
        timer = int(timer)
        datalog = str(timer) + ","\
                  + "Time:" + str(self.gps.Time) + ","\
                  + "緯度:" + str(self.gps.Lat) + ","\
                  + "経度:" + str(self.gps.Lon)
        print(datalog)
        dictionary = self.gps.vincenty_inverse(self.gps.Lat,self.gps.Lon,35.55518,139.65578)
#         dictionary = self.gps.vincenty_inverse(35.55550,139.65457,35.55518,139.65578)
#         x = gps.gpsdis*math.cos(math.radians(gps.gpsdegrees))
        print(dictionary)
#         print("x: "+str(x))
    
    '''
        with open("test.txt",mode = 'a') as test:
            test.write(datalog + '\n')
'''

start_time = time.time()
cansat = cansat() 
cansat.setup()
while True:
    cansat.writeData()
    time.sleep(1)
