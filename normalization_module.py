#!/usr/bin/env python
# coding: utf-8

# # 분석 결과 확인

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


analysis_result = pd.read_csv("submission.csv")
analysis_result = analysis_result.fillna(0)
analysis_result


# In[3]:


analysis_result.shape


# # 분석 결과 정규화

# In[4]:


def normalization(data_set):
    x_max = max(data_set)
    x_min = min(data_set)
    
    # 최댓값 최솟값이 둘다 0이라면 어차피, 0으로 이루어진 리스트이므로 그대로 반환
    if(x_max - x_min) == 0:
        return data_set
    
    result = np.array((data_set - x_min) / (x_max - x_min))
    
    return result.tolist()


# In[5]:


analysis_result["use_tmi_words_value"] = normalization(analysis_result["use_tmi_words_value"])
analysis_result["similar_sentence_value"] = normalization(analysis_result["similar_sentence_value"])
analysis_result["commissional_words_value"] = normalization(analysis_result["commissional_words_value"])
analysis_result["commission_image_value"] = normalization(analysis_result["commission_image_value"])
analysis_result["image_similarity_value"] = normalization(analysis_result["image_similarity_value"])

analysis_result


# # Classification

# In[6]:


from sklearn.cluster import KMeans
import matplotlib.pyplot  as plt

# 특징 요소 분리
feature = analysis_result[["use_tmi_words_value", "similar_sentence_value", "commissional_words_value", "commission_image_value", "image_similarity_value"]]

# 세개의 군집으로 분리
# 각 군집은, 바이럴 데이터 정도 낮음, 중간, 높음을 의미함
model = KMeans(n_clusters=3,algorithm='auto')
model.fit(feature)
predict = pd.DataFrame(model.predict(feature))
predict.columns=['predict']
print(predict['predict'])


# In[8]:


# submission.csv에 기록
f = pd.read_csv("submission.csv")
f = f.fillna(0)
f['class'] = predict['predict']

f.to_csv('submission.csv', mode='w')


# # 모듈 종료 

# In[9]:


print("text_module_finish")


# In[ ]:




