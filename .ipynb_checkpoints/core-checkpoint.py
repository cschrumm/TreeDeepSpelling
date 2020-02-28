import random
import pathlib
import json
import itertools
import numpy as np
import pickle
import math
import torch
import jellyfish

from imgCls import *

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
    lyr = np.zeros((3,WIDTH),np.float32)
    txt = txt.lower()
    ln = len(txt)
    
    if ln > WIDTH - 1:
        raise Exception("text to long..")
        
    strt = int((WIDTH/2)-(ln/2))    
    
    for c in txt:
        if not c in k1:
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


def best_device():
  return torch.device(type= "cuda" if torch.cuda.is_available() else "cpu")

def tensor_info(tn):
  return (tn.device,tn.shape,tn.dtype)

def best_move(tn): 
  return  tn.to(best_device())

def numpy_to_tensor(nn_ar):
    ar = torch.from_numpy(nn_ar)
    ar = best_move(ar)
    return ar

def remove_letter(wd):
    if len(wd) < 5:
        return wd
    cs = random.randint(0,len(wd)-1)
    return wd[0:cs] + wd[cs+1:]

def swap_letter(wd):
    if len(wd) < 3:
        return wd
    cs = random.randint(0,len(wd)-2)
    wd = [w for w in wd]
    t = wd[cs]
    wd[cs] = wd[cs+1]
    wd[cs+1] = t
    
    tmp = ""
    for w in wd:
        tmp += w
        
    return tmp

def change_letter(wd):    
    if len(wd) < 4:
        return wd
    
    alpha = ['a','b','c','d','e','f','g', \
        'h','i','j','k','l','m','n','o','p','q','r', \
        's','t','u','v','w','x','y','z']
    cs = random.randint(0,len(wd)-1)

    wd = [w for w in wd]
    
    wd[cs] = alpha[random.randint(0,25)]
    
    tmp = ""
    for w in wd:
        tmp += w
        
    return tmp

def drop_letter(wd):
    if len(wd) < 5:
        return wd
    cs = random.randint(0,len(wd)-2)
    
    wd = [w for w in wd]
    
    wd[cs] = ' '
    
    tmp = ""
    for w in wd:
        tmp += w
        
    return tmp   

def add_letter(wd):
    if len(wd) < 3:
        return wd
    
    #print("add letter")
    cs = random.randint(0,len(wd)-2)
    
    #wd = [w for w in wd]
    
    alpha = ['a','b','c','d','e','f','g', \
        'h','i','j','k','l','m','n','o','p','q','r', \
        's','t','u','v','w','x','y','z']
    
    #ln = len(wd)
    wd = wd[0:cs] + alpha[random.randint(0,25)] + wd[cs:] 
           
    return wd 


def word_mix(wd):
    fc_ar = [remove_letter,swap_letter,change_letter,drop_letter,add_letter]
    
    wd = fc_ar[random.randint(0,4)](wd)
    
    if random.random() < 0.4:
        wd = fc_ar[random.randint(0,4)](wd)
        
    if random.random() < 0.3:
        wd = fc_ar[random.randint(0,4)](wd)
        
    if random.random() < 0.2:
        wd = fc_ar[random.randint(0,4)](wd)
        
    return wd


def name_path(name):
    return "./net_data/" + name + ".bin"

def save_network(ntwrk,name):
    nt_path = name_path(name)
    torch.save(ntwrk.state_dict(),nt_path)
    return

def load_network(name):
    nt_pth = name_path(name)
    return torch.load(nt_pth)


def load_data_into(model,name):
    model.load_state_dict(load_network(name))
    model.eval()
    return

def get_category(ntwrk,wrds,btch=64):
    ntwrk.eval()
    tmp = []
    with torch.no_grad():
        for b in batch(wrds,btch):
            #print(b)
            bts = word_batch(b)
            tn_bts = numpy_to_tensor(bts)
            #print(tensor_info(tn_bts))
            rslts = ntwrk(tn_bts)
            rv = rslts.argmax(dim=1)
            #print(rv)
            tmp.append(rslts.argmax(dim=1).cpu().numpy())
        
    return np.concatenate(tmp,axis=0)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
        
def word_batch(wrds):
    tmp = []
    for wr in wrds:
        wrd = encode_word(wr)
        wrd = np.expand_dims(wrd,axis=0)
        tmp.append(wrd)
        
    return np.stack(tmp,axis=0)
        
def total_string_distance(t):
    tmp = []
    for a,b in itertools.permutations(t,2):
        tmp.append(jellyfish.damerau_levenshtein_distance(a,b) ** 2)        
    return sum(tmp)

def choose_spread_combo(nm,wrl):
    bts = []
    sms = []
    for i in range(75):
        t = wrl.choose_words(nm)
        bts.append(t)
        sms.append(total_string_distance(t))
    # print(sms,np.argmax(sms))
    return bts[np.argmax(sms)]
            
        
def build_net_opt_schedule():
    ntwrk = ImgNet(5)
    ntwrk = best_move(ntwrk)
    optimizer = optim.Adadelta(ntwrk.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    return (ntwrk,optimizer,scheduler)

def build_choose_and_train(wrl):
    out_num = 5
    targ_arr = []
    in_arr = []
    wrds = choose_spread_combo(out_num, wrl)
    
    for en,v in enumerate(wrds):
        targ_arr.append(en)
        wrd = encode_word(v)
        wrd = np.expand_dims(wrd,axis=0)
        in_arr.append(wrd)
        
    for i in range(2000):
        for en,v in enumerate(wrds):
            targ_arr.append(en)
            wd = word_mix(v)
            wrd = encode_word(wd)
            wrd = np.expand_dims(wrd,axis=0)
            in_arr.append(wrd)
    tn_in, tn_trg  = np.stack(in_arr),np.array(targ_arr,dtype=np.long)
    
    epoch = 7
    example_size = len(targ_arr)
    example_indexes = [x for x in range(example_size)]
    
    model, optimizer, scheduler = build_net_opt_schedule()
    
    for i in range(epoch):
        for b in batch(example_indexes,128):
            #print(b[0],b[-1])
            
            optimizer.zero_grad()
            
            data = tn_in[b[0]:b[-1]]
            target = tn_trg[b[0]:b[-1]]
            
            #print("training data ",data.shape)
            
            data = numpy_to_tensor(data)
            target = numpy_to_tensor(target)
            
            output = model(data)
            
            loss = F.nll_loss(output,target)
            loss.backward()
            
            optimizer.step()
            
        scheduler.step()
        #print("finished  epoch ", (i+1))
    
    return model
