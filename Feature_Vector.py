# -*- coding: utf-8 -*-
#!/usr/bin/python
from __future__ import absolute_import
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
import csv
import nltk
import os
from textblob import TextBlob
import re
import sys
import textstat
import math
import liwc
from collections import Counter

sentences=[]
Final_Vector=[]

#This function is used to calcualte the polarity score of a text.
def Polarity_Score(text,k):
    try:
        #print (text,k)
        pol_score = TextBlob(text)
        val="{:.3f}".format(pol_score.sentiment.polarity)
        #print ('Polarity score:', pol_score.sentiment.polarity)
        Final_Vector[k].append(val)
    except:
        print('Exception occurs in Polarity_Score function')

#This function is used to calcualte the subjectivity score of a text.
def Subjectivity_Score(text,k):
    try:
        subjectivity_score = TextBlob(text)
        val="{:.3f}".format(subjectivity_score.sentiment.subjectivity)
        #print ('Subjectivity score:', subjectivity_score.sentiment.subjectivity)
        Final_Vector[k].append(val)
    except:
        print('Exception occurs in Subjectivity_Score function')

#This function is used to calcualte the number of positive words in a text.
def Positive_Words_Count(text,k):
    try:
        pos_word_list=[]
        token = word_tokenize(text)
        for word in token:               
            testimonial = TextBlob(word)
            if testimonial.sentiment.polarity >= 0.5:
                pos_word_list.append(word)
                
        Final_Vector[k].append(len(pos_word_list))            
    except:
        print("Exception occur in Positive_Words_Count function")

#This function is used to calcualte the number of negative words in a text.
def Negative_Words_Count(text,k):
    try:
        neg_word_list=[]
        token = word_tokenize(text)    
        #print (token)
        for word in token:               
            testimonial = TextBlob(word)
            if testimonial.sentiment.polarity <= -0.5:
                neg_word_list.append(word)
            
        Final_Vector[k].append(len(neg_word_list))            
    except:
        print("Exception occur in Negative_Words_Count function")

#This function is used to calcualte the number of Exclamation marks in a text.
def Exclamation_count(row,k):
        try:
                count = row.count('!')
                Final_Vector[k].append(count)
                                                                        
        except:
                print("Exception occur in Exclaimation_Count function")

#This function is used to calcualte the number of dot (.) marks in a text.
def Dot_count(row,k):
        try:
                count = row.count('.')
                Final_Vector[k].append(count)                                                    
        except:
                print("Exception occur in Dot_Count function")

#This function is used to calcualte the number of Question marks (?) in a text.
def Question_count(row,k):
        try:
                count = row.count('?')
                Final_Vector[k].append(count)                                                         
        except:
                print("Exception occur in Question_Count function")


#This function is used to calcualte the number of interjections in a text.
def Interjection_count(row,k):
        try:
                count=0
                text = word_tokenize(row)                
                pos=nltk.pos_tag(text)
                selective_pos = ['UH']
                #for word,tag in pos:
                        #print (tag)
                selective_pos_words = []
                for word,tag in pos:
                        if tag in selective_pos:
                                selective_pos_words.append((word,tag))
                                count+=1
                
                Final_Vector[k].append(count)
                                                                        
        except:
                print("Exception occur in Interjection_Count function")
                

#This function is used to calcualte the number of adverbs in a text.
def Adverb_count(row,k):
        try:
                count=0
                text = word_tokenize(row)                
                pos=nltk.pos_tag(text)
                selective_pos = ['RB']
                #for word,tag in pos:
                        #print (tag)
                selective_pos_words = []
                for word,tag in pos:
                        if tag in selective_pos:
                                selective_pos_words.append((word,tag))
                                count+=1
                Final_Vector[k].append(count)                            
        except:
                print("Exception occur in Adverb_Count function")   
                
#This function is used to calcualte the number of adjectives in a text.
def Adjective_count(row,k):
        try:
                count=0
                text = word_tokenize(row)                
                pos=nltk.pos_tag(text)
                selective_pos = ['JJ']
                #for word,tag in pos:
                #       print (tag)
                selective_pos_words = []
                for word,tag in pos:
                        if tag in selective_pos:
                                selective_pos_words.append((word,tag))
                                count+=1
                                
                Final_Vector[k].append(count)
        except:
                print("Exception occur in Adjective_count function")   

#This function is used to calcualte the sum of affective valence scores of all tokens in a text.  
def Affective_Valence_Score(row,k):
        try:
                df = pd.read_csv(r'Affective\all.csv',delimiter=',',encoding='latin-1')        
                res=0
                token = word_tokenize(row)                
                for word in token:                
                        df1=(df['Valence Mean'].loc[df['Description'] == word])                       
                        for line in list(df1):
                                res+=line                                
                #res=res/len(token)
                Final_Vector[k].append(res)                            
        except:
                print('Exception occurs in Affective_Valence_Score function')

#This function is used to calcualte the sum of affective arousal scores of all tokens in a text.  
def Affective_Arousal_Score(row,k):
        try:
                df = pd.read_csv(r'Affective\all.csv',delimiter=',',encoding='latin-1')        
                res=0
                token = word_tokenize(row)                
                for word in token:                
                        df1=(df['Arousal Mean'].loc[df['Description'] == word])                       
                        for line in list(df1):
                                res+=line                                
                #res=res/len(token)
                Final_Vector[k].append(res)                            
        except:
                print('Exception occurs in Affective_Arousal_Score function')
                
#This function is used to calcualte the sum of affective dominance scores of all tokens in a text.  
def Affective_Dominance_Score(row,k):
        try:
                df = pd.read_csv(r'Affective\all.csv',delimiter=',',encoding='latin-1')        
                res=0
                token = word_tokenize(row)                
                for word in token:                
                        df1=(df['Dominance Mean'].loc[df['Description'] == word])                       
                        for line in list(df1):
                                res+=line                                
                #res=res/len(token)
                Final_Vector[k].append(res)                            
        except:
                print('Exception occurs in Affective_Dominance_Score function')

def tokenize(text):
    # you may want to use a smarter tokenizer
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)


with open(r'..\Preprocess_CSV\Satire_280.csv', 'r') as my_file:
#with open(r'..\Preprocess_CSV\News_Headlines.csv', 'r') as my_file:
#with open(r'..\Preprocess_CSV\News_Tweets.csv', 'r') as my_file:
    
    f = csv.reader(my_file, quoting=csv.QUOTE_ALL)
    reader = next(f) 
    Tag_list = list(f)
    print (Tag_list)
for sublst in Tag_list:
	Final_Vector.append([])        
k=0
for sublst in Tag_list:
  print (sublst)
  label = sublst[1]
  temp=' '.join(map(str, sublst))
  #print (temp)
  
  Polarity_Score(temp,k)
  Subjectivity_Score(temp,k)
  Positive_Words_Count(temp,k)
  Negative_Words_Count(temp,k)
  Exclamation_count(row,k):
  Dot_count(row,k):
  Question_count(row,k):    
  Interjection_count(temp,k):
  Adverb_count(temp,k)
  Adjective_count(temp,k)
  Affective_Valence_Score(temp,k)
  Affective_Arousal_Score(temp,k)
  Affective_Dominance_Score(temp,k)
  
  Final_Vector[k].append(label)
  print (k)
  k=k+1
print (Final_Vector)

with open(r'Auxiliary_Features\FV_Results\Satire_280.csv', 'w', newline='') as f:
#with open(r'Auxiliary_Features\FV_Results\News_Headlines.csv', 'w', newline='') as f:
#with open(r'Auxiliary_Features\FV_Results\News_Tweets.csv', 'w', newline='') as f:
    
    writer = csv.writer(f)
    writer.writerows(Final_Vector)

print (Final_Vector)
print ('Success')


