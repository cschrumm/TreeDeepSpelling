import random

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