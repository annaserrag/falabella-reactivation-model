"""Train the reactivation model and score today's inactive sellers.

Model: logistic regression on standardized numeric features and one-hot categories,
the same model family BigQuery ML trains with CREATE MODEL ... model_type='logistic_reg'.
Training: inactive-seller snapshots from months 12-30. Out-of-time test: months 32-34
(month 31 is skipped so label windows don't overlap).

Output: model/reactivation_model.joblib, model/metrics.json,
        outputs/inactive_sellers_scored.csv
"""
import numpy as np, pandas as pd, json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import os, joblib
os.makedirs('model',exist_ok=True); os.makedirs('outputs',exist_ok=True)
s=pd.read_csv('data/sellers.csv')
p=pd.read_csv('data/monthly_snapshots.csv.gz').merge(s,on='seller_id'); T=36
p['log_listings']=np.log1p(p.listings); p['ever_sold']=p.ever_sold.astype(int)
p['contact_complete']=p.contact_complete.astype(int)
num=['tenure_months','logins_last_3m','log_listings','ever_sold','months_inactive','onboarding_steps']
cat=['category','country','business_unit','onboarding_channel','size_band']
inact=p[(~p.active.astype(bool))]
hist=inact[inact.label.notna()&(inact.month>=12)]
train=hist[hist.month<=30]; test=hist[hist.month>=32]   # gap month 31 to avoid overlap
pre=ColumnTransformer([('n',StandardScaler(),num),('c',OneHotEncoder(handle_unknown='ignore'),cat)])
m=Pipeline([('pre',pre),('lr',LogisticRegression(max_iter=2000,C=1.0))]).fit(train[num+cat],train.label)
pt=m.predict_proba(test[num+cat])[:,1]; auc=roc_auc_score(test.label,pt)
t=test.assign(score=pt).sort_values('score',ascending=False); n=len(t)
base=t.label.mean()
dec=[t.iloc[int(i*n/10):int((i+1)*n/10)].label.mean() for i in range(10)]
cap30=t.iloc[:int(.3*n)].label.sum()/t.label.sum()
print('train',len(train),'test',len(test),'AUC',round(auc,3),'base',round(base,3))
print('decile rates',[round(d,3) for d in dec]); print('top10 lift',round(dec[0]/base,2),'top30 capture',round(cap30,3))
# coefficients
names=num+list(m.named_steps['pre'].named_transformers_['c'].get_feature_names_out(cat))
coef=pd.Series(m.named_steps['lr'].coef_[0],index=names).sort_values()
print(coef.round(2).to_string())
# score today's inactive sellers
today=p[(p.month==T)&(~p.active.astype(bool))].copy()
today['score']=m.predict_proba(today[num+cat])[:,1]
today=today.sort_values('score',ascending=False)
print('inactive today',len(today), 'expected activations if contact all',today.score.sum().round(0))
for k in [1000,2000,3000,6000]:
    print(k,'top-k expected',today.score.iloc[:k].sum().round(0),'random',round(today.score.mean()*k))
joblib.dump(m,'model/reactivation_model.joblib')
today[['seller_id','category','country','business_unit','onboarding_channel','size_band','listings','onboarding_steps','months_inactive','logins_last_3m','ever_sold','score']].round(4).to_csv('outputs/inactive_sellers_scored.csv',index=False)
json.dump({'auc':auc,'base':base,'deciles':dec,'cap30':cap30,'n_train':len(train),'n_test':len(test),
 'coef':coef.to_dict()},open('model/metrics.json','w'),indent=2)
