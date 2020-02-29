import random
import pathlib
import json
import itertools
import numpy as np
import pickle
import math
import torch
import jellyfish
import pickle

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
        return random.sample(lst,min(nm,len(lst)))
    
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
        if not type(wrd)==str:
            raise Exception("Bad set type",str(type(wrd)))
            
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
    
    mnv,mxv = min(k1.values()),max(k1.values())
    
    mnv2,mxv2 = min(k2.values()),max(k2.values())
    
    mnv3,mxv3 = min(k3.values()),max(k3.values())
    
    for c in txt:
        if not c in k1:
            c2 = ""
            c3 = ""
            strt = strt + 1
            continue
        #print(c, " = ", math.log(k1[c]))
        
        nv = -2 + (4.0 / (mxv - mnv)) * (k1[c] - mnv)
        lyr[1][strt] = nv
        
        #lyr[1][strt] = math.log(k1[c])/(-7.0)
       
        #enc.append(math.log(k1[c])/(-7.0))
        c2 = c2 + c
        c3 = c3 + c
        if len(c2) > 2:
            c2 = c2[1:]
        if len(c3) > 3:
            c3 = c3[1:]
        
        if len(c2) == 2: # and math.log(k2[c2]) < -5.5:
            #vl = math.log(k2[c2]) / (-15.0)
            #print("somewhat rare",c2, " = ",math.log(k2[c2]), " vl = ",vl)
            
            vl = -1 + (2.0 / (mxv2 - mnv2)) * (k2[c2] - mnv2)
            
            lyr[0][strt-1] = lyr[0][strt-1] + vl
            lyr[0][strt] = lyr[0][strt] + vl
            
            #enc.append(math.log(k2[c2])/(-15.0))
        
        if len(c3) == 3: # and math.log(k3[c3]) < -7.5:
            #vl = math.log(k3[c3]) / (-16.0)
            vl = -1 + (2.0 / (mxv3 - mnv3)) * (k3[c3] - mnv3)
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
    #wd = wd
    
    if random.random() < 0.50:
        wd = fc_ar[random.randint(0,4)](wd)
    
    if random.random() < 0.30:
        wd = fc_ar[random.randint(0,4)](wd)
        
    if random.random() < 0.20:
        wd = fc_ar[random.randint(0,4)](wd)
        
    if random.random() < 0.1:
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

def get_category(ntwrk,wrds,btch=64,dbg=False):
    ntwrk.eval()
    tmp = []
    with torch.no_grad():
        for b in batch(wrds,btch):
            #print(b)
            if dbg:
                print("batch.. ",b)
                
            bts = word_batch(b)
            tn_bts = numpy_to_tensor(bts)
            #print(tensor_info(tn_bts))
            rslts = ntwrk(tn_bts)
            if dbg:
                print("result..")
                print(rslts)
            #rv = rslts.argmax(dim=1)
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
        
        #print(type(wr))
        
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
            
        
def build_net_opt_schedule(out_cat=5):
    ntwrk = ImgNet(out_cat)
    ntwrk = best_move(ntwrk)
    optimizer = optim.Adadelta(ntwrk.parameters(), lr=0.5)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.9)
    
    return (ntwrk,optimizer,scheduler)

def build_choose_and_train(wrl,dbg=False,out_cat=5):
    out_num = 5
    targ_arr = []
    in_arr = []
    wrds = choose_spread_combo(out_num, wrl)
    
    for en,v in enumerate(wrds):
        targ_arr.append(en)
        wrd = encode_word(v)
        wrd = np.expand_dims(wrd,axis=0)
        in_arr.append(wrd)
        if dbg:
            print("word ",v," category ",en)
      
    
    for i in range(2000):
        for en,v in enumerate(wrds):
            targ_arr.append(en)
            wd = word_mix(v)
            if dbg:
                print("word ",v," mix ",wd," category ",en)
            wrd = encode_word(wd)
            wrd = np.expand_dims(wrd,axis=0)
            in_arr.append(wrd)
                        
            
    if dbg:
        print("length of input.. ",len(in_arr))
        
    tn_in, tn_trg  = np.stack(in_arr),np.array(targ_arr,dtype=np.long)
    
    epoch = 50
    
    example_size = len(targ_arr)
    example_indexes = [x for x in range(example_size)]
    
    model, optimizer, scheduler = build_net_opt_schedule(out_cat)
    
    for i in range(epoch):
        for b in batch(example_indexes,256):
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
        if dbg:
            print("lr.. ",scheduler.get_lr())
        #print("finished  epoch ", (i+1))
    
    if dbg:
        model.eval()
        with torch.no_grad():
            
            for b in batch(example_indexes,256):
                data = tn_in[b[0]:b[-1]]
                target = tn_trg[b[0]:b[-1]]
                
                data = numpy_to_tensor(data)
                target = numpy_to_tensor(target)
                output = model(data)
                
                # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct = pred.eq(target.view_as(pred)).sum().item()
                
                print("correct on batch.. ", (100.0 * correct) / len(b))
    return model

# rtTreeName contains the name of the network to load 
# avWords (instance of WordManage) contains the list of available words.
# wdDic contains the chosen words root network choice
# maxDepth
def train_and_choose(rtTreeName, avWords, wdDic, maxDepth, btch_count):
    '''
    Note: Tree Name is the following format.. RootName_Subcategory_Subcategory_Subcategory_SubCategory
           where subcategory is a number 0 to 4
    1. Choose N words from avWords min(300,avWords.count)
    2. Create a new instance of get_network with rtTreeName
    3. For the chosen words do the following
        A. find out the category 1 through (category_count) of the words
        B. Now for train_batch count modify all the words using the initially selected
           word as the target.
    4. Now for all words in avWords assign them to on of the (categor_count) categories
    5. Save the network.
    6. If maxDepth <= 1 return..
    7. For 1 to (cagtegory_count) number of categories:
            maxDepth = maxDepth - 1
            A. Select all the elements of chosen category from the avWords.filter_cat(category)
               and append them to a new instance of avWords
            B. If there is more then one element in a category...
               B1. Recursively call train_and_choose with new avWords category and the sub network
                   name netname_ + category_number
            
    '''
    print("train tree..")
    mdl = build_choose_and_train(avWords)
   
    # write the code to save the model..
    save_network(mdl,rtTreeName)

    all_wrds = avWords.all_words()

    #print(type(all_wrds),"      ",type(all_wrds[0]))

    cat_w = get_category(mdl,all_wrds)

    for i,wrd in enumerate(all_wrds):
        #print(wrd,"  ",cat_w[i])
        avWords.set_category(wrd,cat_w[i])
   
    if maxDepth <= 0:
        print("max depth exceeded..")
        return

    for i in range(5):
        sb_tree = rtTreeName + "_" + str(i)
        nw_wrl = avWords.filter_list(i)
        
        for w in nw_wrl.all_words():
            wdDic[w] = sb_tree
    
        print("category ",i," total word count ",len(nw_wrl.all_words()))
    
        # recurse..
        wrds = nw_wrl.all_words()
        if len(wrds) > 1:
            # recurse..
            print("recursing on ",sb_tree," total words.. ", len(wrds))    
            if len(wrds) < 5:
                print("Words..",wrds)
            train_and_choose(sb_tree, nw_wrl, wdDic, maxDepth - 1, btch_count)
    return

def save_json(sjn,flnm):
    data = json.dumps(sjn)
    with open(flnm,"w") as fl:
        fl.write(data)
    return
def load_json(flnm):
    data = None
    with open(flnm,"r") as fl:
        data = fl.read()
    return json.parse(data)