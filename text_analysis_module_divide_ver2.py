#!/usr/bin/env python
# coding: utf-8

# # 데이터 읽기 및 전처리

# In[11]:


pip install -U sentence-transformers


# In[1]:


pip install -e .


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))
pd.set_option('display.max_rows', 999) # pd.options.display.max_rows = 999
pd.set_option('display.max_columns', 999) # pd.options.display.max_columns = 999
pd.set_option('display.width', 1000)

# 한글폰트 적용
plt.rcParams['font.family'] = 'Malgun Gothic'


# In[2]:


absolute_file_address = input("")

# read text data set
text_set = pd.read_csv(absolute_file_address)
text_set = text_set.fillna("")
text_set


# In[3]:


text_set.shape


# In[4]:


text_set = text_set.astype({'text':'string', 'url':'string'})
text_set.dtypes


# # 데이터 분석 실행

# In[5]:


import copy
import openpyxl
import csv

import re
from sentence_transformers import SentenceTransformer, util

f = open('submission.csv', 'w', newline='')
wr = csv.writer(f)
# 과도한 정보 표기 , 고유 단어 사용, 고료 표기, 유사한 문장 사용, 유사한 문장 사용
wr.writerow(["url", "use_tmi_words_value", "similar_sentence_value", "commissional_words_value"])

# Sentence-Transformers 패키지의 RoBERTa 알고리즘 사용
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

individual_text_set = list()
for text in text_set['text']:
    lines = text.split('\n')
    lines = [v for v in lines if v]
            
    individual_text_set = individual_text_set + [v for v in lines if len(v) > 5]
    
embeddings1 = model.encode(individual_text_set, convert_to_tensor=True)


# In[6]:


analysis_count = 0
while analysis_count < len(text_set):
    
    ### 전처리 ###
    print("Analysis position:", analysis_count, "->", analysis_count+5)
    
    if analysis_count+5 < len(text_set)-1:
        text_sub_set = text_set[analysis_count : analysis_count+5]
    else:
        text_sub_set = text_set[analysis_count : len(text_set)-1]
        
    
    # matrix(DataFrame)에서 text데이터를 가져온다.
    text_list = text_sub_set['text']
    url_list = text_sub_set['url']
    url_list = url_list.tolist()
    
    
    
    # text 분할
    individual_text_sub_set = list()
    individual_text_sub_set_size = list()
    for text in text_list:
        lines = text.split('\n')
        lines = [v for v in lines if v]
        lines = [v for v in lines if len(v) > 5]
        
        if(len(lines) == 0):
            lines.append("NONE")
            
        individual_text_sub_set = individual_text_sub_set + lines
        individual_text_sub_set_size.append(len(lines))
        
        
    
    
    ### 무의미한 문장 찾기 ###
    
    use_tmi_words_value = np.zeros(shape=(len(text_list),), dtype=np.int64)

    # 주소
    address_keywords = ["인천", "서울", "경기", "강원", "충청", "충남", "세종", "충북", "대전", "경상", "경북", "경남", "대구", "전라", "전남", "전북", "울산", "부산"]
    # 전화번호
    phoneNumRegex = re.compile(r'((\d{2}|\(\d{2}\)|\d{3}|\(\d{3}\))?(|-|\.)?(\d{3}|\d{4})(\s|-|\.)(\d{4}))')
    # 불규칙 키워드 (피드백 조정)
    irregular_keywords = ['수상', '대회', '출연', '전화번호', '전화', '번호', '운영시간', '운영 시간', '영업시간', '영업 시간']



    dest_keywords = address_keywords + irregular_keywords
    for i in range(0, len(text_list)):
        for line in text_list.tolist()[i].split('\n'):
            if(line == "\n"):
                continue

            for keyword in dest_keywords:
                if(line.find(keyword) != -1):
                    use_tmi_words_value[i] = use_tmi_words_value[i] + 1

            use_tmi_words_value[i] = use_tmi_words_value[i] + len(phoneNumRegex.findall(line))
            
    print("use_tmi_words_value:", use_tmi_words_value)
    
    
    
    
    ### 문장 유사도 분석 ###
    
    similar_sentence_value = np.zeros(shape=(len(text_list),), dtype=np.int64)        

    embeddings2 = model.encode(individual_text_sub_set, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings2, embeddings1) # 코사인 유사도 공식
    
    
    for index in range(0, len(individual_text_sub_set)):
        if(individual_text_sub_set[index] == "NONE"):
            continue
        
        temp = cosine_scores[index]
        
        for i in temp.argsort(descending=True):
            if(cosine_scores[index][i] < 0.95):
                break
                
                
            
            temp_pos = individual_text_sub_set_size[0]
            for l in range(0, len(individual_text_sub_set_size)):
                if(index < temp_pos):
                    similar_sentence_value[l] = similar_sentence_value[l] + 1
                    break
                   
                else:
                   if(l+1 < len(individual_text_sub_set_size)):
                       temp_pos = temp_pos + individual_text_sub_set_size[l+1]
                    
    print("similar_sentence_value", similar_sentence_value)
    
    
    
    
    ### 공정위 문구 분석 ###
    
    commissional_words = ["협찬", "고료", "광고", "후원", "원고", "지원", "제공", "업체", "서비스"]

    commissional_words_value = np.zeros(shape=(len(text_list),), dtype=np.int64)

    for index in range(0, len(text_list)):
        for word in commissional_words:
            if(len(text_list) == 0):
                continue
            if(text_list.tolist()[index].find(word) != -1):
                commissional_words_value[index] = 1

    print("commissional_words_value:", commissional_words_value)
    
    for row in range(0, 5):
        if(len(use_tmi_words_value) <= row):
            break
        wr.writerow([url_list[row], use_tmi_words_value[row], similar_sentence_value[row], commissional_words_value[row]])
        
        
    analysis_count = analysis_count+5
    print("\nData cast")
    
f.close()

print("Analysis finish")


# # 모듈 종료

# In[18]:


print("text_module_finish")


# In[ ]:




