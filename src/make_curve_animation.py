"""Text-free animation of the logistic regression: sellers move onto the fitted curve.
1. Past snapshots appear at 0 (did not activate) or 1 (activated), placed by model score.
2. The S-curve fits: it bends from a flat line into the trained logistic curve.
3. A score axis appears. Fourteen of today's inactive sellers, spread across the score
   range, are mapped one by one: a line rises from the seller to the curve, then runs
   across to the axis, where the seller's probability (score) is marked.
Output: outputs/logistic_curve.mp4 (needs ffmpeg)
"""
import os, numpy as np, pandas as pd, joblib, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap

DG, MG, LIME, GR = '#1b401a', '#6a9368', '#AAD500', '#C4CAC2'
cmap = LinearSegmentedColormap.from_list('f', ['#D9DED6', '#C9E36A', LIME, '#4E8A2A', DG])
rng = np.random.default_rng(7)
m = joblib.load('model/reactivation_model.joblib')
s = pd.read_csv('data/sellers.csv')
p = pd.read_csv('data/monthly_snapshots.csv.gz').merge(s, on='seller_id')
p['log_listings'] = np.log1p(p.listings); p['ever_sold'] = p.ever_sold.astype(int)
num = ['tenure_months','logins_last_3m','log_listings','ever_sold','months_inactive','onboarding_steps']
cat = ['category','country','business_unit','onboarding_channel','size_band']
inact = p[~p.active.astype(bool)]
hist = inact[inact.label.notna() & inact.month.between(32, 34)]
tr = pd.concat([hist[hist.label == 1].sample(160, random_state=1), hist[hist.label == 0].sample(320, random_state=1)])
today_all = inact[inact.month == 36]
z_all = m.decision_function(today_all[num+cat])
z_tr = m.decision_function(tr[num+cat]); y_tr = tr.label.values
XL = (-8, 4); X0 = XL[0] + 0.25          # X0 = position of the score axis
clip = lambda z: np.clip(z, X0+0.5, XL[1]-0.3)
z_tr = clip(z_tr)
# 14 of today's inactive sellers, spread across the score range, shown left to right
qs = [0.03, 0.15, 0.3, 0.45, 0.58, 0.68, 0.76, 0.83, 0.88, 0.92, 0.95, 0.975, 0.99, 0.999]
z_td = clip(np.sort(np.quantile(z_all, qs)))
sig = lambda z: 1/(1+np.exp(-z))
ytr = y_tr*0.96 + 0.02 + rng.normal(0, 0.012, len(y_tr))
p_td = sig(z_td); col_td = cmap(np.clip((p_td/p_td.max())**0.6, 0.25, 1))
zz = np.linspace(X0, XL[1], 400)
ease = lambda x: 0.5-0.5*np.cos(np.pi*np.clip(x, 0, 1))
T0, STEP, END = 11.5, 0.85, 26

FPS = 30; W, H = 16, 9
fig = plt.figure(figsize=(W, H), dpi=100); ax = fig.add_axes([0.04, 0.08, 0.92, 0.86])
def frame(t):  # t in seconds
    ax.clear(); ax.set_xlim(XL[0], XL[1]); ax.set_ylim(-0.06, 1.06); ax.axis('off')
    ax.plot([X0, XL[1]], [0, 0], color='#E6E9E4', lw=1.5); ax.plot([X0, XL[1]], [1, 1], color='#E6E9E4', lw=1.5)
    # part 1: past sellers at 0 / 1, the line bends into the fitted curve
    a_in = ease(t/1.5); a_out = 1-ease((t-9)/1.5)
    if a_in*a_out > 0:
        ax.scatter(z_tr, ytr, s=55, c=np.where(y_tr == 1, MG, GR), alpha=0.85*a_in*a_out, edgecolors='none')
    k = ease((t-3)/4)
    if k > 0:
        ax.plot(zz, sig(k*zz), color=DG, lw=4, alpha=min(1, k*3), zorder=2)
    # part 2: score axis appears, today's sellers are mapped onto the curve one by one
    ax_a = ease((t-10)/1)
    if ax_a > 0:
        ax.plot([X0, X0], [0, 1], color='#9AA39A', lw=2, alpha=ax_a)
        for yy in (0, 0.25, 0.5, 0.75, 1):
            ax.plot([X0-0.08, X0], [yy, yy], color='#9AA39A', lw=2, alpha=ax_a)
    for i, (z, pv, c) in enumerate(zip(z_td, p_td, col_td)):
        ti = T0 + i*STEP
        app = ease((t-(ti-0.6))/0.4)
        if app <= 0: continue
        up = ease((t-ti)/0.35); left = ease((t-ti-0.35)/0.35)
        current = ti <= t < ti + STEP
        la = 0.9 if current else 0.3
        lw = 2.4 if current else 1.4
        ax.scatter([z], [0], s=110, c=[c], edgecolors=DG, linewidths=1.2, alpha=app, zorder=4)
        if up > 0:
            ax.plot([z, z], [0, pv*up], color=DG, lw=lw, ls=(0, (4, 3)), alpha=la, zorder=3)
        if up >= 1:
            ax.scatter([z], [pv], s=150 if current else 90, c=[c], edgecolors=DG, linewidths=1.2, zorder=5)
        if left > 0:
            ax.plot([z, z-(z-X0)*left], [pv, pv], color=DG, lw=lw, ls=(0, (4, 3)), alpha=la, zorder=3)
        if left >= 1:
            ax.scatter([X0], [pv], s=150 if current else 90, c=[c], edgecolors=DG, linewidths=1.2, zorder=6)
os.makedirs('outputs', exist_ok=True)
w = FFMpegWriter(fps=FPS, bitrate=5000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
with w.saving(fig, 'outputs/logistic_curve.mp4', dpi=100):
    for f in range(int(END*FPS)):
        frame(f/FPS); w.grab_frame()
print('done')
