import random

class WordManage():
    def load(self,flnm):
        self._wrds = {}
        with open(flnm,"r") in fl:
            self._wrds = { wrd : 0 for wrd in fl.readlines() }
        return
    def choose_words(self,nm):
        lst = [wrd for wrd in self._wrds.keys()]
        return random.sample(lst,nm)
    
    def filter_list(self, category):
        wrd = WordManage()
        dc = { wrd : category for (wrd,itm) in self._wrds.items() if itm == category  }
        wrd.set_dic(dc)
        return wrd
    
    def set_dic(self,dc):
        self._wrds = dc
        return
    
    
    def all_words(self):
        return [k for k in self._wrds.keys()]
    
    def set_words(self,wrds,category):
        self._wrds = { wrd : category for wrd in wrds  }
        return
    
    def set_category(self,wrd,vlu):
        self._wrds[wrd] = vlu
        return