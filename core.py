import random
import pathlib

# This class manages the list of words that are being trained..
class WordManage():
    # load a file of words to train.
    def load(self,flnm):
        self._wrds = {}
        with open(flnm,"r") in fl:
            self._wrds = { wrd : 0 for wrd in fl.readlines() }
        return
    
    # choose n words from the list
    def choose_words(self,nm):
        lst = [wrd for wrd in self._wrds.keys()]
        return random.sample(lst,nm)
    
    # filter by a category
    def filter_list(self, category):
        wrd = WordManage()
        dc = { wrd : category for (wrd,itm) in self._wrds.items() if itm == category  }
        wrd.set_dic(dc)
        return wrd
    
    # set the hidden dictionary for the managed list
    def set_dic(self,dc):
        self._wrds = dc
        return
    
    # get a list of available words
    def all_words(self):
        return [k for k in self._wrds.keys()]
    
    # set category value for a list of words (alternative to load method)
    def set_words(self,wrds,category):
        self._wrds = { wrd : category for wrd in wrds  }
        return
    
    # sets a words category
    def set_category(self,wrd,vlu):
        self._wrds[wrd] = vlu
        return
    
    # number of words
    def word_count(self):
        return len(self._wrds.keys())
    
def load_keys():
    kvy = None
    pth = pathlib.Path(__file__).parent.absolute()
    pth = str(pth) + "/encoding_keys.json"
    
    with open(pth,"r") as fl:
        data = fl.read()
        kyv = json.loads(data)
        
    return kyv


def encode_word_keys(txt,k1,k2,k3):
    c,c2,c3 = "","",""
    WIDTH = 256
    lyr = np.zeros((3,WIDTH))
    txt = txt.lower()
    ln = len(txt)
    
    if ln > WIDTH - 1:
        raise Exception("text to long..")
        
    strt = int((WIDTH/2)-(ln/2))    
    
    for c in txt:
        if not c in ky:
            c2 = ""
            c3 = ""
            strt = strt + 1
            continue
        #print(c, " = ", math.log(k1[c]))
        lyr[1][strt] = math.log(k1[c])/(-7.0)
       
        #enc.append(math.log(k1[c])/(-7.0))
        c2 = c2 + c
        c3 = c3 + c
        if len(c2) > 2:
            c2 = c2[1:]
        if len(c3) > 3:
            c3 = c3[1:]
        
        if len(c2) == 2 and math.log(k2[c2]) < -5.5:
            vl = math.log(k2[c2]) / (-15.0)
            #print("somewhat rare",c2, " = ",math.log(k2[c2]), " vl = ",vl)
            lyr[0][strt-1] = lyr[0][strt-1] + vl
            lyr[0][strt] = lyr[0][strt] + vl
            
            #enc.append(math.log(k2[c2])/(-15.0))
        
        if len(c3) == 3 and math.log(k3[c3]) < -7.5:
            vl = math.log(k3[c3]) / (-16.0)
            #print("rare..",c3, " ", math.log(k3[c3]), " vl = ",vl)
            lyr[2][strt-2] = lyr[2][strt-2] + vl
            lyr[2][strt-1] = lyr[2][strt-1] + vl
            lyr[2][strt] = lyr[2][strt]
            #enc.append(math.log(k3[c3])/(-16.0))
            
        strt = strt + 1
    return lyr


ky_dct = None


def encode_word(wrd):
    global ky_dct
    
    if ky_dct is None:
        ky_dct = load_keys()
    
    return encode_word_keys(wrd,ky_dct['ky'],ky_dct['ky2'],ky_dct['ky3'])