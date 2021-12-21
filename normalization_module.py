#!/usr/bin/env python
# coding: utf-8

# # 분석 결과 확인

# In[69]:


import numpy as np
import pandas as pd


# In[70]:


analysis_result = pd.read_csv("submission.csv")
analysis_result = analysis_result.fillna(0)
analysis_result


# In[71]:


analysis_result.shape


# # 분석 결과 정규화

# In[72]:


def normalization(data_set):
    x_max = max(data_set)
    x_min = min(data_set)
    
    # 최댓값 최솟값이 둘다 0이라면 어차피, 0으로 이루어진 리스트이므로 그대로 반환
    if(x_max - x_min) == 0:
        return data_set
    
    result = np.array((data_set - x_min) / (x_max - x_min))
    
    return result.tolist()


# In[73]:


analysis_result["use_tmi_words_value"] = normalization(analysis_result["use_tmi_words_value"])
analysis_result["similar_sentence_value"] = normalization(analysis_result["similar_sentence_value"])
analysis_result["commissional_words_value"] = normalization(analysis_result["commissional_words_value"])
analysis_result["commission_image_value"] = normalization(analysis_result["commission_image_value"])
analysis_result["image_similarity_value"] = normalization(analysis_result["image_similarity_value"])

analysis_result


# # Classification

# In[74]:


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

r = pd.concat([feature, predict], axis=1)

predict['predict']


# In[75]:


doubt_num = 0
viral_num = 1
none_num = 2

for i in range(0, len(analysis_result)):
    if analysis_result['commissional_words_value'][i] == 1 or analysis_result['commission_image_value'][i] == 1:
        viral_num = predict['predict'][i]
        break
        
        
tmi_avr = np.mean(analysis_result['use_tmi_words_value'].tolist())
sentence_avr = np.mean(analysis_result['similar_sentence_value'].tolist())
image_avr = np.mean(analysis_result['image_similarity_value'].tolist())

for i in range(0, len(analysis_result)):
    if analysis_result['commissional_words_value'][i] == 0 and analysis_result['commission_image_value'][i] == 0: 
        if analysis_result['use_tmi_words_value'][i] <= tmi_avr and analysis_result['similar_sentence_value'][i] <= sentence_avr and analysis_result['image_similarity_value'][i] <= image_avr:
            none_num = predict['predict'][i]
            break
        
for i in range(0, len(analysis_result)):
    if predict['predict'][i] != viral_num and predict['predict'][i] != none_num:
        doubt_num = predict['predict'][i]
        break
        
        
        
        
for i in range(0, len(predict['predict'])):
    if predict['predict'][i] == viral_num:
        predict['predict'][i] = 1
    elif predict['predict'][i] == none_num:
        predict['predict'][i] = 2
    elif predict['predict'][i] == doubt_num:
        predict['predict'][i] = 0


# In[55]:


# submission.csv에 기록
f = pd.read_csv("submission.csv")
f = f.fillna(0)
f['class'] = predict['predict']

f.to_csv('submission.csv', mode='w')


# # XGBoost Regression

# In[76]:


import xgboost
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error


# In[77]:


temp_tmi_avr = np.mean(analysis_result['use_tmi_words_value'].tolist())
temp_sentence_avr = np.mean(analysis_result['similar_sentence_value'].tolist())
temp_image_avr = np.mean(analysis_result['image_similarity_value'].tolist())

commission_data_mask = (analysis_result.commissional_words_value == 1) | (analysis_result.commission_image_value == 1)
none_commission_data_mask = (analysis_result.commissional_words_value != 1) & (analysis_result.commission_image_value != 1) & (analysis_result.use_tmi_words_value <= temp_tmi_avr) & (analysis_result.similar_sentence_value <= temp_sentence_avr) & (analysis_result.image_similarity_value <= temp_image_avr)
commission_data = analysis_result.loc[commission_data_mask,:]
none_commission_data = analysis_result.loc[none_commission_data_mask,:]

temp_data_set = commission_data.append(none_commission_data)



X = temp_data_set[["use_tmi_words_value", "similar_sentence_value", "image_similarity_value"]]
Y_words = temp_data_set[["commissional_words_value", "commission_image_value"]]["commissional_words_value"].tolist()
Y_image = temp_data_set[["commissional_words_value", "commission_image_value"]]["commission_image_value"].tolist()

Y = list()

for i in range(0, len(Y_words)):
    if(Y_words[i] == 1 or Y_image[i] == 1):
        Y.append(1)
    else:
        Y.append(0)

        
data_dmatrix = xgboost.DMatrix(data=X,label=Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y ,test_size=0.1)
xgb_model = xgboost.XGBRegressor(objective = 'reg:linear', n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)

print(len(X_train), len(X_test))
xgb_model.fit(X_train,y_train)


# In[78]:


xgboost.plot_importance(xgb_model)


# In[79]:


predictions_probs = xgb_model.predict(X_test)
predictions_probs


# In[80]:


predictions = [ 1 if x > 0.5 else 0 for x in predictions_probs]
predictions


# In[81]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


# In[82]:


def get_clf_eval(y_test, pred, pred_probs):
   confusion = confusion_matrix(y_test, pred)
   accuracy = accuracy_score(y_test, pred)
   precision = precision_score(y_test, pred)
   recall = recall_score(y_test, pred)
   f1 = f1_score(y_test, pred)
   # ROC-AUC
   roc_auc = roc_auc_score(y_test, pred_probs)
   print('오차 행렬')
   print(confusion)
   # ROc-AUC
   print('정확도 : {:.4f}, 정밀도 : {:.4f}, 재현율 : {:.4f},   F1 : {:.4f}, AUC : {:.4f}'.format(accuracy,precision,recall,f1,roc_auc))


# In[83]:


get_clf_eval(y_test, predictions, predictions_probs)


# In[64]:


r_sq = xgb_model.score(X_train, y_train)
rmse = np.sqrt(mean_squared_error(y_test, predictions_probs))
params = {"objective":"reg:linear",'n_estimators': 100,'learning_rate': 0.08,'gamma': 0, 'subsample': 0.75, 'colsample_bytree': 1, 'max_depth': 7}
cv_results = xgboost.cv(dtrain=data_dmatrix, params=params, nfold=3,num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
print("score:", r_sq)
print("f1-score:", r_sq)
print("explained_variance_score:", explained_variance_score(predictions_probs,y_test))
print("RMSE:", rmse)


# In[65]:


cv_results.head()


# 실제 예측단계

# In[66]:


predictions_probs = xgb_model.predict(analysis_result[["use_tmi_words_value", "similar_sentence_value", "image_similarity_value"]])
predictions = [ 1 if x > 0.5 else 0 for x in predictions_probs]
regression_result = list()

for p in predictions:
    if p == 0:
        regression_result.append(2)
        
    else:
        regression_result.append(1)


# In[67]:


# submission.csv에 기록
f = pd.read_csv("submission.csv")
f = f.fillna(0)
f['regression'] = regression_result

f.to_csv('submission.csv', mode='w')


# # 모듈 종료 

# In[68]:


print("text_module_finish")


# In[ ]:




