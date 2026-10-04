"""Generate synthetic seller data for the Falabella Sellers Management case.

20,000 sellers observed monthly for 36 months (month 36 = today), calibrated to the
assignment annex: ~40% of sellers active today. A seller is "active" in a month if they
sell. Each monthly snapshot records the seller's state at the start of the month and a
label: 1 if an inactive seller becomes active within the next 3 months.

The transition probabilities depend on seller features plus an unobserved engagement
factor, so the model cannot be perfect. All data is synthetic.

Output: data/sellers.csv, data/monthly_snapshots.csv.gz
"""
import numpy as np, pandas as pd
rng=np.random.default_rng(42)
N=20000; T=36  # month 36 = today
cats=['Electronics','Home','Fashion','Beauty','Sports','Toys','Hardware','Food']
cat_eff=dict(zip(cats,[0.35,0.2,0.25,0.15,0.0,-0.1,0.1,-0.25]))
countries=['Chile','Peru','Colombia']; c_eff={'Chile':0.15,'Peru':0.0,'Colombia':-0.1}
bus=['Falabella Retail','Sodimac','Tottus','Marketplace direct']; bu_eff=dict(zip(bus,[0.1,0.15,-0.05,0.0]))
chans=['Self sign-up','Sales rep','Referral','Event']; ch_eff=dict(zip(chans,[-0.3,0.45,0.3,0.0]))
sizes=['Micro','Small','Medium']; sz_eff=dict(zip(sizes,[-0.25,0.1,0.4]))
s=pd.DataFrame({
 'seller_id':[f'S{i:05d}' for i in range(N)],
 'category':rng.choice(cats,N,p=[.16,.18,.17,.1,.1,.08,.12,.09]),
 'country':rng.choice(countries,N,p=[.5,.25,.25]),
 'business_unit':rng.choice(bus,N,p=[.35,.25,.15,.25]),
 'onboarding_channel':rng.choice(chans,N,p=[.5,.2,.15,.15]),
 'size_band':rng.choice(sizes,N,p=[.55,.33,.12]),
 'reg_month':rng.integers(0,T,N),
 'onboarding_steps':rng.integers(0,6,N),
 'contact_complete':rng.random(N)<0.5,
})
eng=rng.normal(0,1,N)  # latent engagement (unobserved)
base=(s.category.map(cat_eff)+s.country.map(c_eff)+s.business_unit.map(bu_eff)+s.onboarding_channel.map(ch_eff)
      +s.size_band.map(sz_eff)+0.22*s.onboarding_steps+0.5*eng).values
listings=np.zeros(N); active=np.zeros(N,bool); ever=np.zeros(N,bool); since_active=np.zeros(N)  # months inactive
rows=[]; status=np.full((N,T+1),-1)
for t in range(T+1):
    reg=s.reg_month.values<=t
    tenure=t-s.reg_month.values
    logins=rng.poisson(np.clip(np.exp(0.4+0.6*eng+0.35*active+rng.normal(0,.5,N)),0,30))*reg
    listings=listings+reg*rng.poisson(np.clip(np.exp(-1.2+0.5*eng+0.6*active),0,20))
    # snapshot features at start of month t (before transition)
    snap=pd.DataFrame({'seller_id':s.seller_id,'month':t,'tenure_months':tenure,'logins_last_3m':logins,
        'listings':listings.copy(),'ever_sold':ever.copy(),'months_inactive':since_active.copy(),'active':active.copy(),'registered':reg})
    rows.append(snap)
    # transitions for next month
    lp=-3.4+base+0.06*np.log1p(listings)*3+0.09*np.minimum(logins,15)+0.8*ever-0.10*np.minimum(since_active,24)-0.03*np.maximum(tenure-6,0)
    p_act=1/(1+np.exp(-lp))
    p_churn=1/(1+np.exp(-(-2.8-0.5*base)))
    u=rng.random(N)
    new_active=np.where(active,u>p_churn,u<p_act)&reg
    newly=new_active&~active
    ever|=new_active
    since_active=np.where(new_active,0,since_active+reg)
    active=new_active
    status[:,t]=active
panel=pd.concat(rows,ignore_index=True)
panel=panel[panel.registered].drop(columns='registered')
panel=panel.merge(s,on='seller_id')
# label: inactive at month t, becomes active in t..t+2 (status recorded after month transition)
panel['label']=np.nan
idx=panel.seller_id.str[1:].astype(int).values; m=panel.month.values
fut=np.zeros(len(panel)); ok=m+2<=T
for k in range(3):
    mm=np.clip(m+k,0,T)
    fut=np.maximum(fut,status[idx,mm])
panel['label']=np.where(ok,fut,np.nan)
import os
os.makedirs('data',exist_ok=True)
s.to_csv('data/sellers.csv',index=False)
cols=['seller_id','month','tenure_months','logins_last_3m','listings','ever_sold','months_inactive','active','label']
panel[cols].to_csv('data/monthly_snapshots.csv.gz',index=False,compression='gzip')
today=panel[panel.month==T]
print('registered today',len(today),'active today',today.active.sum(), today.active.mean().round(3))
inact=panel[(~panel.active)&panel.label.notna()&(panel.month>=12)]
print('inactive snapshots',len(inact),'3m activation rate',inact.label.mean().round(3))
