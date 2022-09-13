from datetime import datetime
try:
    import cPickle as pickle
except:
    import pickle
import os
from io import StringIO
import pprint

class Preset(object):
    def __init__(self, name, string):
        self.name   = name
        self.string = string
        self.date   = datetime.today().strftime('%Y-%m-%d')
        self.usages = 0
        return
        
def save_Preset(obj):
    obj.date=datetime.today().strftime('%Y-%m-%d')
    try:
        with open("save_preset/"+obj.name, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)
        
def load_Preset(filename):
    try:
        with open("save_preset/"+filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during loading preset: (", ex,")  ", filename)
     
def exist_Preset(name): # check if a preset exists
    return os.path.isfile("./save_preset/"+name) 
    
def delete_Preset(name): # delete a preset
    return os.remove("./save_preset/"+name) 

def show_Preset():
    Presets=[]
    for set in os.listdir("./save_preset/"):
        f = os.path.join("./save_preset/", set)
        # checking if it is a file
        if os.path.isfile(f):
            Presets.append(load_Preset(set))
    return Presets
            

def prompt_preset_delete(prompt, load_str="!sd"):
    p = prompt.split()
    names = [p[i+1] for i, x in enumerate(p) if x == load_str]  
    delete_Preset(names[0])
    
    
def prompt_preset_load(prompt, load_str="!l"):
    # split the sentence into words
    p = prompt.split()
    names = [p[i+1] for i, x in enumerate(p) if x == load_str]
    out = 0 
    
    for n in names:
        if not exist_Preset(n):
            print("Warning: ",n, " preset does not exist")
            sust = ""
            out = 1
        else:
            pres_temp=load_Preset(n)
            pres_temp.usages += 1
            save_Preset(pres_temp)
            sust = pres_temp.string
            out = 0
        prompt = prompt.replace(load_str+" "+n, sust)
    return out, prompt
    
def prompt_preset_save(prompt, override=False, load_str="!s"):
    p = prompt.split()
    name   = [p[i+1] for i, x in enumerate(p) if (x == load_str)or(x == load_str+"!")][0]
    string = " ".join([p[i+2:] for i, x in enumerate(p) if (x == load_str)or(x == load_str+"!")][0])
    
    if not exist_Preset(name) or override:
        save_Preset(Preset(name,string))
        print("preset saved", name,": ",string)
        out = 0
    else:
        print("Warning: preset ", name, "already existing, add <<override = True>> if you want to overwrite any preset")
        out = 1
    return out, name
    
    
    
if not os.path.exists("./save_preset/"):
  os.makedirs("./save_preset/")
     