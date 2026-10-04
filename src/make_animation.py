"""Render the 34-second animation of the model learning and scoring (needs ffmpeg).
Optional: put Public Sans .ttf files in fonts/ as PS400.ttf, PS700.ttf, PS800.ttf.
Output: outputs/model_simulation.mp4
"""
import numpy as np, pandas as pd, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle
import os
os.makedirs('outputs',exist_ok=True)
F={w:(FontProperties(fname=f'fonts/PS{w}.ttf') if os.path.exists(f'fonts/PS{w}.ttf') else FontProperties(family='sans-serif',weight='bold' if w>400 else 'normal')) for w in (400,700,800)}
DG='#1b401a'; MG='#6a9368'; LIME='#AAD500'; SL='#343E49'; GR='#9AA39A'; LG='#E4E8E1'; INK='#1F2A1F'
cmap=LinearSegmentedColormap.from_list('f',['#E4E8E1','#C9E36A',LIME,'#4E8A2A',DG])
W,H=1600,900; FPS=30
s=pd.read_csv('data/sellers.csv'); p=pd.read_csv('data/monthly_snapshots.csv.gz')
met=json.load(open('model/metrics.json')); today=pd.read_csv('outputs/inactive_sellers_scored.csv').sort_values('score',ascending=False)
# monthly new activations
piv=p.pivot_table(index='seller_id',columns='month',values='active',aggfunc='first').astype(bool)
acts=[int(((piv[t+1]==True)&(piv[t]==False)).sum()) for t in range(0,35)]
n_snap=met['n_train']; total_act=sum(acts[:31])
sc=np.sort(today.score.values)[::-1]; NI=len(sc)
q=np.quantile(sc,np.linspace(0,1,1200))[::-1]  # 1200 dots, descending
rng=np.random.default_rng(3); order0=rng.permutation(1200)
top_exp=int(round(sc[:3000].sum())); rnd_exp=int(round(sc.mean()*3000)); ratio=top_exp/rnd_exp
top5=today.head(5)
coef=met['coef']
drivers=[('Products listed',coef['log_listings']),('Onboarding steps completed',coef['onboarding_steps']),
 ('Portal logins, last 3 months',coef['logins_last_3m']),('Has sold before',coef['ever_sold']),
 ('Time since registration',coef['tenure_months']),('Micro-sized seller',coef['size_band_Micro']),
 ('Self sign-up onboarding',coef['onboarding_channel_Self sign-up']),('Months inactive',coef['months_inactive'])]
S=[(0,210),(210,405),(405,660),(660,960)]; TOT=1020
ease=lambda x:0.5-0.5*np.cos(np.pi*np.clip(x,0,1))
fig=plt.figure(figsize=(W/100,H/100),dpi=100); ax=fig.add_axes([0,0,1,1])
def T(x,y,s,w=400,size=20,c=INK,**k): ax.text(x,y,s,fontproperties=F[w],fontsize=size,color=c,**k)
steps=['Learn from history','Find what drives activation','Score today\u2019s inactive sellers','Decide who to call first']
def header(k):
    for i,s in enumerate(steps):
        x=[40,350,720,1120][i]
        on=i==k; done=i<k
        c=DG if on else (MG if done else GR)
        ax.add_patch(plt.Circle((x+16,850),16,color=c))
        T(x+16,850,str(i+1),700,15,'white',ha='center',va='center')
        T(x+42,850,s,700 if on else 400,16,DG if on else GR,va='center')
    ax.plot([40,1560],[810,810],color=LG,lw=2)
def scene1(f):
    a=f/210; header(0)
    T(40,760,'Learning from 3 years of seller history',800,34,DG,va='top')
    T(40,700,'Every month, some inactive sellers start selling again. The star schema\u2019s status history records each one.',400,18,SL,va='top')
    k=int(ease(a*1.25)*35)
    x0,y0,bw,hmax=420,150,30,430; mx=max(acts)
    for i in range(35):
        h=acts[i]/mx*hmax if i<k else 0
        ax.add_patch(Rectangle((x0+i*bw,y0),bw-6,h,color=MG if i<31 else LIME))
    for i in range(0,35,6): T(x0+i*bw+12,y0-28,f'M{i+1}',400,13,GR,ha='center')
    T(x0+31*bw-3,y0+hmax+20,'held out to test',400,13,SL,ha='left')
    ax.plot([x0+31*bw-3]*2,[y0,y0+hmax+10],color=SL,lw=1.2,ls=(0,(4,4)))
    T(x0,y0-60,'New activations per month (synthetic history)',400,14,GR)
    cnt=int(sum(acts[:min(k,31)]))
    T(40,560,f'{cnt:,}',800,58,DG); T(40,525,'activations the model learns from',400,17,SL)
    T(40,420,f'{int(n_snap*min(1,ease(a*1.25))):,}',800,40,DG); T(40,390,'monthly snapshots of inactive sellers',400,17,SL)
def scene2(f):
    a=f/195; header(1)
    T(40,760,'What makes an inactive seller come back',800,34,DG,va='top')
    T(40,700,'The model weighs each feature. Green pushes the score up; grey pushes it down.',400,18,SL,va='top')
    cx=900; sc_=520
    ax.plot([cx,cx],[150,620],color=GR,lw=1.5)
    for i,(nm,v) in enumerate(drivers):
        y=590-i*58; g=ease(a*1.6-i*0.08)
        w=v*sc_*g
        ax.add_patch(Rectangle((cx if v>0 else cx+w,y-17),abs(w),34,color=MG if v>0 else '#B7BEB6'))
        T(cx-20 if v>0 else cx+20,y,nm,700 if i<4 else 400,17,INK,ha='right' if v>0 else 'left',va='center')
    T(40,95,f'Model: logistic regression, the same model type BigQuery ML trains with SQL.   Out-of-time test AUC {met["auc"]:.2f}.',400,15,GR)
def grid_pos(r):
    cols=50; return 80+(r%cols)*22, 640-(r//cols)*22
def scene3(f):
    a=f/255; header(2)
    T(40,760,f'Scoring today\u2019s {NI:,} inactive sellers',800,34,DG,va='top')
    T(40,700,'Each dot is about 10 sellers. Colour shows the probability of becoming active in the next 3 months.',400,18,SL,va='top')
    appear=ease(a/0.12); colour=ease((a-0.15)/0.2); move=ease((a-0.42)/0.3); hl=ease((a-0.78)/0.15)
    xs=[];ys=[];cs=[]
    for r in range(1200):
        x0,y0=grid_pos(order0[r]); x1,y1=grid_pos(r)
        xs.append(x0+(x1-x0)*move); ys.append(y0+(y1-y0)*move)
        cs.append(cmap(colour*(q[r]/q[0])**0.55))
    ax.scatter(xs,ys,s=110*appear,c=cs,edgecolors='none')
    if hl>0:
        ax.add_patch(FancyBboxPatch((66,640-6*22+11),50*22-8,6*22,boxstyle='round,pad=2,rounding_size=8',fill=False,ec=DG,lw=3,alpha=hl))
        T(1200,640-3*22+11,'Top 3,000 sellers:\ncontact these first',700,19,DG,ha='left',va='center',alpha=hl)
    # legend
    for i,v in enumerate(np.linspace(0,1,40)): ax.add_patch(Rectangle((80+i*8,62),8,16,color=cmap(v)))
    T(80,42,'Low',400,13,GR); T(400,42,'High',400,13,GR,ha='right'); T(420,70,'probability of activating',400,13,GR,va='center')
def scene4(f):
    a=f/300; header(3)
    T(40,760,'Who to call first, and what it is worth',800,34,DG,va='top')
    T(40,700,'Account managers get this ranked list in Looker every month.',400,18,SL,va='top')
    cols=[(40,'Seller'),(170,'Category'),(330,'Products'),(450,'Onboarding'),(590,'Inactive'),(700,'Score')]
    for x,h in cols: T(x,640,h,700,15,GR)
    ax.plot([40,900],[625,625],color=LG,lw=2)
    for i,(_,r) in enumerate(top5.iterrows()):
        g=ease(a*3-i*0.25); y=595-i*62
        if g<=0: continue
        vals=[r.seller_id,r.category,f'{int(r.listings)}',f'{int(r.onboarding_steps)}/5 steps',f'{int(r.months_inactive)} mo']
        for (x,_),v in zip(cols,vals): T(x,y,v,400,17,INK,va='center',alpha=g)
        ax.add_patch(Rectangle((700,y-12),150*r.score*g,24,color=MG))
        T(700+150*r.score*g+10,y,f'{r.score*100:.0f}%',700,17,DG,va='center',alpha=g)
    # comparison
    b=ease((a-0.35)/0.35); bx=1010; mxv=top_exp; L=400
    T(bx,640,'Calling 3,000 inactive sellers',700,18,INK)
    T(bx,585,'Picked at random',400,16,SL); ax.add_patch(Rectangle((bx,530),L*rnd_exp/mxv*b,36,color='#B7BEB6'))
    T(bx+L*rnd_exp/mxv*b+12,548,f'~{int(rnd_exp*b):,}',700,20,SL,va='center')
    T(bx,470,'Ranked by the model',400,16,SL); ax.add_patch(Rectangle((bx,415),L*top_exp/mxv*b,36,color=DG))
    T(bx+L*top_exp/mxv*b+12,433,f'~{int(top_exp*b):,}',700,20,DG,va='center')
    T(bx,375,'expected activations within 3 months',400,14,GR)
    r=ease((a-0.72)/0.15)
    if r>0:
        T(bx,250,f'{ratio:.1f}\u00d7',800,64,DG,alpha=r)
        T(bx+215,275,'more activations from\nthe same number of calls',400,18,SL,va='center',alpha=r)
    T(40,95,'Simulation on synthetic data built from the case figures (20,000 sellers, ~40% active). Real results will differ.',400,14,GR)
w=FFMpegWriter(fps=FPS,bitrate=6000,codec='libx264',extra_args=['-pix_fmt','yuv420p'])
with w.saving(fig,'outputs/model_simulation.mp4',dpi=100):
    for fr in range(TOT):
        ax.clear(); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis('off'); fig.patch.set_facecolor('white')
        if fr<210: scene1(fr)
        elif fr<405: scene2(fr-210)
        elif fr<660: scene3(fr-405)
        else: scene4(min(fr-660,299))
        w.grab_frame()
print('done',top_exp,rnd_exp,round(ratio,2))
