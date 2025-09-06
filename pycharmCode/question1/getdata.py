# @Version : 1.0
# @Author  : 亥子曜
# @File    : getdata.py
# @Time    : 2025/9/5 0:36
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

df = pd.read_excel("../../data/附件1.xlsx")
df = df[df['波数 (cm-1)'] > 1500].copy()

x = df["波数 (cm-1)"].values
y = df["反射率 (%)"].values.astype(float)

bg_win = max(101, (len(y)//12)//2*2+1)
baseline = savgol_filter(y, window_length=bg_win, polyorder=2, mode='interp')
residual = y - baseline
dnu = np.median(np.diff(x))
z = residual - residual.mean()
acf = np.correlate(z, z, mode='full')
acf = acf[acf.size//2:]
lag_skip = max(3, int(round(20/dnu)))
peaks_acf, _ = find_peaks(acf[lag_skip:], distance=3)
if len(peaks_acf) > 0:
    lag_pts_est = peaks_acf[np.argmax(acf[lag_skip:][peaks_acf])] + lag_skip
else:
    lag_pts_est = np.argmax(acf[lag_skip:]) + lag_skip
period_est = lag_pts_est * dnu
min_distance = max(5, int(round(0.6 * lag_pts_est)))
min_width = max(1, int(round(0.15 * lag_pts_est)))
mad = np.median(np.abs(residual - np.median(residual))) * 1.4826
prom_thr = max(0.5 * mad, 0.08 * np.ptp(residual))
peaks, props = find_peaks(residual, distance=min_distance,
                          width=min_width, prominence=prom_thr)
peak_nu = x[peaks]
spacing = np.diff(peak_nu)
print("自相关估算周期 ≈ %.2f cm-1" % period_est)
if spacing.size > 0:
    print("平均间隔 = %.2f cm-1" % spacing.mean())
    print("中位数间隔 = %.2f cm-1" % np.median(spacing))
    print("标准差 = %.2f cm-1" % spacing.std(ddof=1))


plt.figure(figsize=(10,4))
plt.plot(x, residual, label="Residual (signal - baseline)")
plt.scatter(peak_nu, residual[peaks], c='r', s=20, label="Detected peaks")
plt.gca().invert_xaxis()
plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Residual (a.u.)")
plt.title("Residual spectrum (>1500 cm$^{-1}$)")
plt.legend()
plt.show()



peak_table = pd.DataFrame({
    "peak_wavenumber_cm-1": peak_nu
})
spacing_table = pd.DataFrame({
    "from_peak_cm-1": peak_nu[:-1],
    "to_peak_cm-1": peak_nu[1:],
    "spacing_cm-1": spacing
})
peak_table.to_csv("detected_peaks_gt1500.csv", index=False, encoding="utf-8-sig")
spacing_table.to_csv("peak_spacings_gt1500.csv", index=False, encoding="utf-8-sig")
