# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 17:09:53 2019

@author: ashish soni
"""
import re
import pandas as pd
from pandas import Series
import string
import pandas as pd
import numpy as np
import os
from os import listdir
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import DataFrame
from matplotlib import pyplot
import collections
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer

import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score, make_scorer, recall_score, accuracy_score, cohen_kappa_score
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support, average_precision_score )
                             
from sklearn.feature_selection import RFE


from python.TranscribeVoice import *
import python.GridSearchforBestModelv1  
from python.GridSearchforBestModelv1 import *


import lime
import lime.lime_tabular

nltk.download("stopwords")

FEATURES = r.FEATURES
doc = pd.read_csv(r.DATA)
cases = doc.shape[0]
LIME_EXPLANATION = r.LIME_EXPLANATION
stemmer = PorterStemmer()

stemmer_snowball = SnowballStemmer("english", ignore_stopwords=True)

# save list to file
def save_list(lines, filename):
    # convert lines to a single blob of text
    data = '\n'.join(lines)
    # open file
    file = open(filename, 'w')
    # write text
    file.write(data)
    # close file
    file.close()

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text



def remove_stopwords(tokens):
    
    stop = stopwords.words('english')

    additional_stopwords = ['I','ing','i','the','you','is','and','of','that','title','numeric','isn','dublin','just','theres','were','ahead',
                            'to','transcriber','are','You','speaking','sweet','people','then', 'can','very','able','it','its',
                            'would','again','look','but','some','not','there','going','doing','put','putting','each','every',
                            'zero','summarize','summary','dial','dial','havery','call','tour','client','ctly','will','would',
                            'record','made','noise','esi','mhm','audio','email','words','accent','customer','dear','finish',
                           'great','sort','whilst','bye','spell','helo', 'commercial','let','inutes','have','had','off',
                            'hello','line','long','bit','...','stamp','simply','whatever','mobile','donote','illion','recording']

    
    keywords = load_doc('../training/keywords.txt').split('\n')

    Names1=[]
    from nltk.corpus import wordnet
    for Nouns in tokens:
        if not wordnet.synsets(Nouns):
            #Not an English word
            Names1.append(Nouns)
    Names1 = list(set(Names1))


    x=dict(nltk.pos_tag(tokens))
    Names2 = list({k for (k,v) in x.items() if v in ['NN']})
    Names2 = list(set(Names2))

    Names = list(set(Names1 + Names2))

    Names_new = [x for x in Names if x not in keywords]
    Names1=[Names_new]
    Names1.append([[x[:-1] for x in Names_new]])
    Names1.append([x+'s' for x in Names_new])

    def combine_nlist(nlist,init=0,combiner=lambda x,y: x+y):
        '''
        apply function: combiner to a nested list element by element(treated as flatten list)
        '''
        current_value=init
        for each_item in nlist:
            if isinstance(each_item,list):
                current_value =combine_nlist(each_item,current_value,combiner)
            else:
                current_value = combiner(current_value,each_item)
        return current_value
    def flatten_nlist(nlist):
        return combine_nlist(nlist,[],lambda x,y:x+[y])
    flatten_nlist(Names1)

    stop = list(additional_stopwords + flatten_nlist(Names1)+stop)

    #stop = list(flatten_nlist(Names1))

    output=[]
    for w in tokens:
        if w not in stop:
            output.append(w)
    return output
       

def stemming(tokens):
    for i in range(len(tokens)):
        tokens[i] = stemmer_snowball.stem(tokens[i])
    return tokens


def remove_punct_short_tokens(tokens):
    re_punc = re.compile('[%s]' % re.escape(string.punctuation.replace("_","")))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
 
    # remove remaining tokens that are not digit
    tokens = [word for word in tokens if not word.isdigit()]
     
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 2]
    
    return tokens



def remove_nonalpha_tokens(tokens):
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]  
    
    return tokens

def remove_stopwords_(tokens):
    stop = stopwords.words('english')
    additional_stopwords = ['I','i','the','you','is','and','of','that','title','numeric','isn',
                            'dublin','just','theres','were','ahead','erm','...','..','..','theres',
                            'to','transcriber','are','You','speaking','sweet','people','then', 'can',
                            'very','able','it','its','great','sort','whilst','bye','spell','helo','hello','line',
                            'again','look','but','some','not','there','going','doing','put','putting','each','every',
                            'zero','summarize','summary','dial','dial','havery','call','tour','customer','dear','finish',
                            'client','ctly','record','made','noise','esi','mhm','audio','email','words','accent',
                            'commercial','let','inutes','have','had','off','long','bit','stamp','simply',
                            'whatever','mobile','donote','illion','recording']
    
    keywords = load_doc('keywords.txt').split('\n')
    keyword1 = [x+'s' for x in keywords]
    keyword2 = [x+'ed' for x in keywords]
    keyword3 = [x+'ing' for x in keywords]
    keyword4 = [x+'ly' for x in keywords]
    keyword5 = [x+'th' for x in keywords]
    keywordx = list(keywords + keyword1 + keyword2 + keyword3 + keyword4 + keyword5)
    
    Names1=[]
    for Nouns in tokens:
        if wordnet.synsets(Nouns):
            #Not an English word
            Names1.append(Nouns)
    Names1 = list(set(Names1))
    
    x=dict(nltk.pos_tag(tokens))
    Names2 = list({k for (k,v) in x.items() if v in ['NN','JJ']})
    
    Names = list(set(Names1 + Names2))

    Names = [x for x in Names if x not in keywordx]

    stop = list(additional_stopwords+stop + Names)
    output=[]
    for w in tokens:
        if w not in stop:
            output.append(w)
    return output
    
def CleanTextFromText(txt):
    content = txt.splitlines(True)
    pattern = r"^TITLE.*|^HSBC.*|^FOR.*|^AT.*|^AUDIO\sLENGTH.*|^LENGTH.*|^TRANSCRIPT.*|^TYPE.*|^(\s\\n)|^\s+LENGTH.*|^TRANSCRIBER.*|^COMPLETION.*|^www\.essentialsecretary.*|^\[Start\sof.*|^\[End\sof.*|^AuDIO|^\[Telephone.*"
    new_txt = []
    for line in content:
        t=re.sub(pattern, "", line)
        if (t!='\n'):
            new_txt.append(t)

    new_txt=''.join(new_txt)
    new_txt = re.sub('spk\_1','bank',new_txt.lower())
    new_txt = re.sub('spk\_2','customer',new_txt)
    new_txt = re.sub(r'pv|present\svalue|expected\scash\sflow\w*|cash\sflow\w*|mark\-to\-market|mark\sto\smarket|fair\svalue|asset\w*|liabilit\w*',' financial_concepts ',new_txt)
    new_txt = re.sub(r'[A-Z][A-Z]\d\s\d[A-Z][A-Z]','POSTCODE',new_txt)
    new_txt = re.sub(r'95\-97\sprospect\sstreet|[h]ull','ADDRESS',new_txt)
    new_txt = re.sub(r'any\sdifficulties.*but\sfollowed\sit\swith','',new_txt)    
    new_txt = re.sub('www\.essentialsecretary\.co\.uk',' ',new_txt)    
    new_txt = re.sub('\[.*?\]|\-\s*\w*\s*\-',' ',new_txt)
    new_txt = re.sub("\\\'","'",new_txt)
    new_txt = re.sub(r'\s+',' ',new_txt)
    pattern1='LENGTH\:\s[\d]+\sMinutes.*?(?=[a-zA-Z]*\:.*)'
    new_txt = re.sub(pattern1,' ',new_txt)
   
    pattern2='^.*?(?=[a-zA-Z]*\:.*)'
    new_txt = re.sub(pattern2,'',new_txt)
    new_txt = re.sub(r'[Cc]lose\-out|[Bb]reak\w+|\bbreak\sout|\b[Cc]ancel\w+|\bcancel|\bcome\sout|\bclose\syour\strade|termina\w+','exit',new_txt)
    new_txt = re.sub(r'[Rr]remainder|time\svalue|left\sto\srun|left\sto\sgo|to\srun|how\slong\sthere\sis\sto\srun|worth\sleft','remaining',new_txt)
    new_txt = re.sub(r"\bper\scent|\bpr\scent|%"," percent", new_txt)
    new_txt = re.sub(r"\s*\d+\s*percent|\s(\d*[,.]*\d*)\s*percent|\s(\d*[,.]*\d*)\s*percent"," percent", new_txt)

    new_txt = re.sub(r"assume|let\'s\ssay|(e\s+x*a*mple)|(in\s+my\s+example)|(example\s+let\s+[s]*\s+say)|(in\spart(s)*)|work\sout|worked\sout|calculat\w+|determine"," example ",new_txt)     
    new_txt = re.sub(r'i\sam\swith\syou|clear\sto\sme|understood|\bagree|(i\s+get\s+it)|(\sgot\s+it)|([(iI]\sdo\s)', 'understand',new_txt)
    new_txt = re.sub(r'times|\sx\s', '  multiply ',new_txt)   
    new_txt = re.sub(r'why\shave\syou\sgone|collapse|\blow\w*|go\sdown|come\soff|below|down|fall\w+|drop\w*|\bvary\w*|surprise|fluctuat\w+|\brise|go\sup|balloon\sup|increas\w*|high\w+|above|\bup|market\smove\w+|move\smuch|move\sless|more\sthan|less\sthan|lot\smore|more|substantial\w*|large\w*|significa\w*|huge\w*', ' variation',new_txt)
    new_txt = re.sub(r'profit|compensat\w+|gain|won\s|win\s|lottery|credit|refund|pay\syou',' benefit', new_txt)
    new_txt = re.sub(r'loss\w+|paid\s|charg\w+|debit\w+|pay\sus',' cost',new_txt)
    new_txt = re.sub(r'\sfine|(\s*yes)|(\s*yep)|(\s*[y]+eah)|(\sokay)|(i\s+think\s+so)',' right', new_txt)
    new_txt = re.sub(r"\£\d*[,.]\d*\w+|\d*[,.]\d*\s*million|\d*[,.]\d*\s*pound\w+|\d*([\,\.]\d+)+"," money_value",new_txt)
    new_txt = re.sub(r'guess|forsee|believe|hope|speculate|envisage|reckon|\bsense|imagine|feel|predict','expect',new_txt)
    new_txt = re.sub(r"\d+[.,]\d*|\d+|d+\,d+\,d+"," numeric",new_txt)
      
    return(new_txt.lower())

# generate features from dataframe
def generate_initial_features(doc):
    #doc2=pd.DataFrame()    
    doc2 = doc    
    doc2["n_exit"] = doc["text"].map(lambda s: s.count(r'exit'))
    doc2["n_multiply"] = doc["text"].map(lambda s: s.count(r'multipl'))
    #doc2["n_papers"] = doc["text"].map(lambda s: s.count(r'papers'))
    #doc2["n_would"] = doc["text"].map(lambda s: s.count(r'would'))
    doc2["n_differ"] = doc["text"].map(lambda s: s.count(r'differ'))
    #doc2["n_explore"] = doc["clean_text"].map(lambda s: s.count(r'explore'))
    #doc2["n_level_range"] = doc["clean_text"].map(lambda s: s.count(r'level_range'))
    #doc2["n_prevailing"] = doc["text"].map(lambda s: s.count(r'prevailing'))
    #doc2["n_pay"] = doc["text"].map(lambda s: s.count(r'pay'))
    doc2["n_benefit"] = doc["text"].map(lambda s: s.count(r'benefit'))
    doc2["n_financial_concepts"] = doc["text"].map(lambda s: s.count(r'financial_concepts'))
   # doc2["n_break"] = doc["text"].map(lambda s: s.count(r'break'))
    doc2["n_cost"] = doc["text"].map(lambda s: s.count(r'cost'))
    #doc2["n_multiplied_by"] = doc["clean_text"].map(lambda s: s.count(r'multiplied_by'))
    doc2["n_example"] = doc["text"].map(lambda s: s.count(r'example'))
    #doc2["n_confirming"] = doc["clean_text"].map(lambda s: s.count(r'confirming'))
    doc2["n_percent"] = doc["text"].map(lambda s: s.count(r'percent'))
    doc2["n_explain"] = doc["text"].map(lambda s: s.count(r'explain'))
    doc2["n_year"] = doc["text"].map(lambda s: s.count(r'year'))
    #doc2["n_yes"] = doc["clean_text"].map(lambda s: s.count(r'yes'))
    #doc2["n_money_value"] = doc["clean_text"].map(lambda s: s.count(r'money_value'))
    #doc2["n_market_rate"] = doc["clean_text"].map(lambda s: s.count(r'market_rate'))
    doc2["n_swap"] = doc["text"].map(lambda s: s.count(r'swap'))
    doc2["n_notional"] = doc["text"].map(lambda s: s.count(r'notion'))
    doc2["n_remaining"] = doc["text"].map(lambda s: s.count(r'remaining'))
    doc2["n_significant"] = doc["text"].map(lambda s: s.count(r'significant'))    
    doc2["n_right"] = doc["text"].map(lambda s: s.count(r'right'))
    #doc2["n_pound"] = doc["text"].map(lambda s: s.count(r'pound'))
    doc2["n_money_value"] = doc["text"].map(lambda s: s.count(r'money_value'))
    doc2["n_understand"] = doc["text"].map(lambda s: s.count(r'understand'))
    doc2["n_rate"] = doc["text"].map(lambda s: s.count(r'rate'))                                                         
    doc2["n_variation"] = doc["text"].map(lambda s: s.count(r'variation'))   
    doc2["n_expect"] = doc["text"].map(lambda s: s.count(r'expect')) 
    doc2["n_numeric"] = doc["text"].map(lambda s: s.count(r'numeric')) 
    
    return doc2



def generate_features(txt):
    #print(txt)
    clean_text = CleanTextFromText(txt.lower())
    p = re.split('\n*\s+(?=\w*\:\s)',clean_text) 
    p = list(filter(str.strip, p))
    df1=pd.DataFrame()
    df1['speaker'], df1['text'] =pd.DataFrame(p)[0].str.split(':', 1).str
    df1 = df1[((df1['speaker']=='customer')|(df1['speaker']=='bank'))]
    df1 = generate_initial_features(df1)
    df1.loc[(df1.speaker == 'customer') & 
        (
            (df1.n_variation>0)|
            (df1.n_right>0)|
            (df1.n_rate>0)|
            (df1.n_percent>0)|
            (df1.n_cost>0)|
            (df1.n_money_value>0)|
            (df1.n_year>0)|
            (df1.n_financial_concepts>0)|
            (df1.n_numeric>0)|
            (df1.n_remaining>0)),
        'customer_understood'] = 1
    df1.loc[ (df1.n_exit>0) \
             & (df1.n_differ>0) \
             & (df1.n_multiply>0) \
             & (df1.n_numeric>0) \
             & (df1.n_swap>0) \
             & (df1.n_year>0) \
          #  & (df1.n_notional>0)
             & (df1.n_percent>0)\
             & (df1.n_money_value>0)
            # & (df1.n_remaining>0)
            ,'explain_example'] = 1
    df1.loc[(df1.speaker=='bank')& 
            ((df1.n_example>0)|
             (df1.n_benefit>0)|        
             (df1.n_numeric>0)|
             (df1.n_percent>0)| 
             (df1.n_differ>0)|
             (df1.n_money_value>0)|
             (df1.n_remaining>0)|        
             (df1.n_notional>0)
            ),'explain_mechanics'] = 1
    df2 = df1.loc[:,'n_exit':'explain_mechanics'].sum(axis = 0).to_frame().transpose()
    nconversations_bank = len(df1[df1.speaker == 'bank'])
    df2['n_bank_conversations'] = nconversations_bank
    return(df2)






lst1 = list()
for case_no in list(range(1,cases+1)):
    raw_text = list(doc[case_no-1:case_no]['text'])[0] 
    label = list(doc[case_no-1:case_no]['label'])[0]    
    df = generate_features(raw_text)  
    df['case']= case_no
    df['label'] = label
    lst1.append(df)
doc2 = pd.concat(lst1)

doc2.to_csv(r.FEATURES, sep=',') 


X = doc2.drop(['case','label'], axis=1)
Y = doc2['label']
index = range(len(Y))
Train_Test_Split = 0.15
seed = 100


X_tsne = TSNE(learning_rate=200,n_components=2).fit_transform(X)
X_pca = PCA(n_components=2, svd_solver="arpack").fit_transform(X) # using arpack svd_solver for sparse matrices

# Partition count data into Train and Test sets

from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=Train_Test_Split,random_state=seed)


seed = 9


explainer = lime.lime_tabular.LimeTabularExplainer(X.astype(int).values, 
                                                   mode='classification',
                                                   #training_labels=Y,
                                                   feature_names=list(X.columns)
                                                   )


# LIME model interpret Training/Test dataset
def model_interpret_traintestdata(X,model,idx,num_features):
    # asking for explanation for LIME model
    exp = explainer.explain_instance(X.loc[idx,:].astype(int).values, model.predict_proba, num_features=num_features)
    exp.class_names = ['compliant','non-compliant']
    return exp
    #exp.show_in_notebook(show_table=True)
    
# LIME model interpret new data
def model_interpret_newdata_features(X,model,num_features):
    # asking for explanation for LIME model
    exp = explainer.explain_instance(X.iloc[0].astype(int).values, model.predict_proba, num_features=num_features)
    exp.class_names = ['compliant','non-compliant']
    return exp

def model_interpret_new_text(text,label,model,num_features):
    exp = (preprocess_and_generate_features_from_text(text,label,case_no)
                   .drop(['case','label'],axis=1)
                   .pipe(model_interpret_newdata_features,model,num_features))
    exp.class_names = ['compliant','non-compliant']
    exp.save_to_file(LIME_EXPLANATION)
    return exp

# Generating prediction results dataframe
def predicted_df(df,model):
    X = df.drop(['label','case'], axis=1)
    Y = df['label']
    
    y_pred = model.predict(X)
    probs =model.predict_proba(X)
    y_pred_prob_NC = probs[:, 1]   
    results_df=pd.DataFrame({
                      #  'caseNo':X['case'],
                        'actualLabel': Y,
                        'PredictedLabel':y_pred,
                        'PredictedProbability_NC':y_pred_prob_NC
                         })
    return results_df


# prediction
def predict_model(x,model):
    y_pred = model.predict(x) #changed x_test to x - 06Aug
    return(y_pred)

# data frame with prediction probablities 
# this is required as LIME requires class probabilities in case of classification example
# LightGBM directly returns probability for class 1 by default 
def prob(df,model):
    X = df.drop(['label'], axis=1)
    probs =model.predict_proba(X)
    y_pred_prob_NC = probs[:, 1] 
    y_pred_prob_C = probs[:, 0] 
    return np.array(list(zip(y_pred_prob_NC,y_pred_prob_C)))

# Predict
def read_casefiles(caselist,path):
    lst = list()
    for case_no in caselist:
        case_voice_dirpath = path + r'/case'+str(case_no)

        corpus_file = case_voice_dirpath + r'/maincorpus.txt' 
        file = open(corpus_file, 'rt',encoding='utf8',errors='ignore')
        text = file.read()
        file.close()

        label_file = case_voice_dirpath + r'/label.txt'
        file = open(label_file, 'rt',encoding='utf8',errors='ignore')
        label = file.read()
        file.close()

        data = {'case': case_no, 'text': text, 'label': label}
        lst.append(pd.DataFrame(data, index=[case_no]))

        doc = pd.concat(lst)
        
    return doc

#preprocess and generate features from text
def preprocess_and_generate_features_from_text(raw_text,label,case_no):
    df = generate_features(raw_text)
    df['case']= case_no 
    df['label'] = label
    return df

#preprocess and generate features from datframe
def preprocess_cases(doc):
    lst = []
    for case_no in list(range(1,len(doc))):
        raw_text = list(doc[case_no-1:case_no]['text'])[0]
        label = list(doc[case_no-1:case_no]['label'])[0]
        df = preprocess_and_generate_features_from_text(raw_text,label,case_no)
        lst.append(df)
    doc2 = pd.concat(lst)
    return doc2

# Generating predictions for cases in caselist
def generate_predictions(caselist,cases_path,model):
    predictions = (read_casefiles(caselist,cases_path)
                   .pipe(preprocess_cases)   
                   .pipe(predicted_df,model)
                  )

    return predictions

def generate_prediction_FromText(text,model):
    label='unknown'
    case_no = 115
    predictions = (preprocess_and_generate_features_from_text(text,label,case_no)
                   .pipe(predicted_df,model))
    
    exp = (preprocess_and_generate_features_from_text(text,label,case_no)
                      .drop(['label','case'],axis=1)
                      .pipe(model_interpret_newdata_features,model,num_features=10))
    exp.class_names = ['compliant','non-compliant']
   # exp.show_in_notebook(show_table=False)
    exp.save_to_file(LIME_EXPLANATION,show_table=False,predict_proba=False)
                  
    return predictions

def generate_preprocessed_Text(text):
    #clean_text = text       
    clean_text = CleanTextFromText(text)
    p = re.split('\n*\s+(?=\w*\:\s)',clean_text) 
    p = list(filter(str.strip, p))
    df1=pd.DataFrame()
    df1['speaker'], df1['text'] =pd.DataFrame(p)[0].str.split(':', 1).str
    df1 = df1[((df1['speaker']=='customer')|(df1['speaker']=='bank'))]
    bank_txt=df1[df1['speaker']=='bank']['text'].str.cat(sep=' ')
    customer_txt=df1[df1['speaker']=='customer']['text'].str.cat(sep=' ')
    lst = list([bank_txt,customer_txt])    
    return(lst)
    


def feature_selection(model,x_train, y_train,num_features):
    rfe_selector = RFE(model, num_features) #estimator
    rfe_selector = rfe_selector.fit(x_train, y_train)
    return rfe_selector



def model_grid_tuning(x_train,y_train,options):  
   grid = EstimatorSelectionHelper(options)
   grid.fit(x_train, y_train)
   return grid



f1_scorer = make_scorer(f1_score, pos_label="non-compliant")
recall_scorer = make_scorer(recall_score, pos_label="non-compliant")
accuracy_scorer= make_scorer(recall_score, pos_label="non-compliant")

options = {'scoring': 'roc_auc',
           'cv': 5}

#grid = model_grid_tuning(x_train,y_train,options)


def classificationReport(y,y_pred):
    report = classification_report(y, y_pred)
    return report

