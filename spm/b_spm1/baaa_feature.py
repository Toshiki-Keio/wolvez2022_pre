import cv2
from pyrsistent import s
from FEATURE import Feature_img
import tempfile

class ReadFeaturedImg():
    """画像読込関数
    
    Args:
        importPath (str): Original img path
        saveDir (str): Save directory path that allowed tmp
        Save(bool):Save or not, defalt:False
    """
    def __init__(self, importPath:str=None, saveDir:str=None, Save:bool=False):
        self.imp_p = importPath
        if Save:
            self.sav_d = saveDir
        #else:
        #    self.sav_d = tempfile.TemporaryDirectory()
        self.save = Save
    
    def feature_img(self, frame_num, feature_name=None):
        '''Change to treated img
        Args:
            frame_num(int):Frame number or time
            feature_name(str):
        '''
        self.treat = Feature_img(self.imp_p, frame_num, self.sav_d)
        
        self.treat.normalRGB()
        self.treat.vari()
        self.treat.enphasis()
        self.treat.edge()
        self.treat.r()
        self.treat.b()
        self.treat.g()
        self.treat.rb()
        self.treat.gb()
        self.treat.rg()
        
        fmg_list = self.treat.output()
        
        return fmg_list


    def read_img(self, path):
        #print("===== func read_img starts =====")
        self.img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        self.img = self.img[int(0.5*self.img.shape[0]):]
        # 読み込めないエラーが生じた際のロバスト性も検討したい
        return self.img