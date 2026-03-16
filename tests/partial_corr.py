import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('seraphim_results.csv')
n   = df['median_n'].values
chi = df['median_chi_eff'].values
q   = df['median_q'].values

# Direct q→n
r_q, p_q = stats.pearsonr(q, n)
print(f"Direct r(q, n)       = {r_q:.4f}, p = {p_q:.4f}")

# Direct chi→n  
r_c, p_c = stats.pearsonr(chi, n)
print(f"Direct r(chi_eff, n) = {r_c:.4f}, p = {p_c:.4f}")

# Partial: chi_eff → n | q (residuals method)
s1,i1,_,_,_ = stats.linregress(q, n)
n_resid = n - (s1*q + i1)
s2,i2,_,_,_ = stats.linregress(q, chi)
chi_resid = chi - (s2*q + i2)
r_p, _ = stats.pearsonr(chi_resid, n_resid)
N = len(n)
t = r_p * np.sqrt((N-3)/(1-r_p**2))
p2 = 2*stats.t.sf(abs(t), df=N-3)
print(f"Partial r(chi|q)     = {r_p:.4f}, t = {t:.3f}, p = {p2:.4f}")
print(f"Spin signal survives q control: {'YES' if p2 < 0.05 else 'NO'}")

