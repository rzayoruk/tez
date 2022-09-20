# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tldextract
import math
from collections import Counter
import os
import pickle
import numpy as np
import warnings
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
#from typing import Optional
#from pyndatic import BaseModel

#class dgaModel(BaseModel):
#    texts: str
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "https://127.0.0.1:8000/admin/dgahunter",
    "http://127.0.0.1:8000/admin/dgahunter",
    "http://127.0.0.1:8000",
    "https://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



templates = Jinja2Templates(directory="templates")


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})



#@app.post('/predict')
#async def make_predict(request:Request, uri:str):
#    sonuc=evaluate_url(model,uri)
#    return {"sonuc":sonuc}


@app.get("/predict")
#url-> templatedeki inputun name i 
def make_predict(request:Request, uri : str ):
    sonuc=evaluate_url(model, uri)
    print(sonuc)
    return {"sonuc":sonuc}
    #return sonuc
    #return templates.TemplateResponse("prediction.html", {"request": request,"uri":uri,"sonuc":sonuc}) 



######################################################################################


warnings.filterwarnings("ignore", category=DeprecationWarning)

def domain_extract(uri):
    ext = tldextract.extract(uri)
    if (not ext.suffix):
        return np.nan
    else:
        return ext.domain
    

 
def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum( count/lns * math.log(count/lns, 2) for count in p.values())

def contains_digit(domain):
    """
     Contains Digits 
    """
    #subdomain = ignoreVPS(domain)
    for item in domain:
        if item.isdigit():
            return 1
    return 0

def vowel_ratio(domain):
    """
    calculate Vowel Ratio 
    """
    VOWELS = set('aeiou')
    v_counter = 0
    a_counter = 0
   # subdomain = ignoreVPS(domain)
    for item in domain:
        if item.isalpha():
            a_counter+=1
            if item in VOWELS:
                v_counter+=1
    if a_counter>1:
        ratio = np.float(v_counter/a_counter)
        return ratio
    else :
        return 0


def load_model_from_disk(name, model_dir='models'):

    # Model directory is relative to this file
    model_path = os.path.join(model_dir, name+'.model')

    # Put a try/except around the model load in case it fails
    try:
        model = pickle.loads(open(model_path,'rb').read())
    except:
        print('Could not load model: %s from directory %s!' % (name, model_path))
        return None

    return model

def evaluate_url(model, url):

    domain = domain_extract(url)
    alexa_match = model['alexa_counts'] * model['alexa_vc'].transform([url]).T
    dict_match = model['dict_counts'] * model['dict_vc'].transform([url]).T

    # Assemble feature matrix (for just one domain)
    X = [[len(domain), entropy(domain), alexa_match, dict_match,contains_digit(domain),vowel_ratio(domain)]]
    y_pred = model['clf'].predict(X)[0]
    print('%s : %s' % (domain, y_pred))
    return y_pred



print('Loading Models...')
clf = load_model_from_disk('C:/Users/rizay/tez/models/dga_model_random_forest')
alexa_vc = load_model_from_disk('C:/Users/rizay/tez/models/dga_model_alexa_vectorizor')
alexa_counts = load_model_from_disk('C:/Users/rizay/tez/models/dga_model_alexa_counts')
dict_vc = load_model_from_disk('C:/Users/rizay/tez/models/dga_model_dict_vectorizor')
dict_counts = load_model_from_disk('C:/Users/rizay/tez/models/dga_model_dict_counts')
model = {'clf':clf, 'alexa_vc':alexa_vc, 'alexa_counts':alexa_counts,
                 'dict_vc':dict_vc, 'dict_counts':dict_counts}

# Examples (feel free to change these and see the results!)
#evaluate_url(model, 'www.google.com')
#evaluate_url(model, 'www.14ss431qwfghgjghfrsf.com')
evaluate_url(model, 'www.1cb8gg53543dfa5f36f.com')