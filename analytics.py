"""
╔══════════════════════════════════════════════════════════════╗
║          BLIND ASSIST — ANALYTICS DASHBOARD                  ║
║  Pehle blind_assist_v7.py chalao — detection_log.csv banega ║
║  Phir yeh script chalao — saare graphs ban jaayenge         ║
╚══════════════════════════════════════════════════════════════╝



import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

LOG_FILE = "detection_log.csv"

# CSV check
if not os.path.exists(LOG_FILE):
    print(f"ERROR: '{LOG_FILE}' nahi mili!")
    print("first run  blind_assist_v7.py for sometime")
    print("then press q to stop and run this script now ")
    sys.exit(1)

df = pd.read_csv(LOG_FILE)
if df.empty:
    print("CSV mein koi data nahi hai! Pehle detection karo.")
    sys.exit(1)

df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"Total detections loaded: {len(df)}")
print(f"Unique objects: {df['object'].nunique()}")
print(f"Time range: {df['timestamp'].min()} → {df['timestamp'].max()}")
print("-" * 50)


#  DASHBOARD 

fig = plt.figure(figsize=(18, 12))
fig.suptitle("Blind Assist — Object Detection Analytics Dashboard",
             fontsize=20, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── COLOR PALETTE
HIGH_COLOR   = "#E74C3C"
NORMAL_COLOR = "#2ECC71"
BLUE_SHADES  = ["#2980B9", "#3498DB", "#5DADE2", "#85C1E9",
                "#AED6F1", "#D6EAF8", "#EBF5FB"]


# ── GRAPH 1: Top 10 most detected objects 
ax1 = fig.add_subplot(gs[0, 0])
top_objects = df["object"].value_counts().head(10)
colors_bar  = [HIGH_COLOR if obj in {"car","bus","truck","motorcycle","train","bicycle","pole"}
               else NORMAL_COLOR for obj in top_objects.index]
bars = ax1.barh(top_objects.index[::-1], top_objects.values[::-1], color=colors_bar[::-1])
ax1.set_title("Top 10 Detected Objects", fontweight='bold', fontsize=12)
ax1.set_xlabel("Detection Count")
for bar, val in zip(bars, top_objects.values[::-1]):
    ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             str(val), va='center', fontsize=9)
ax1.set_xlim(0, top_objects.max() * 1.15)
# Legend
from matplotlib.patches import Patch
ax1.legend(handles=[Patch(color=HIGH_COLOR, label='HIGH danger'),
                    Patch(color=NORMAL_COLOR, label='NORMAL')],
           loc='lower right', fontsize=8)


# ── GRAPH 2: HIGH vs NORMAL pie 
ax2 = fig.add_subplot(gs[0, 1])
tier_counts = df["danger_tier"].value_counts()
ax2.pie(tier_counts.values,
        labels=tier_counts.index,
        colors=[HIGH_COLOR if t == "HIGH" else NORMAL_COLOR for t in tier_counts.index],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2})
ax2.set_title("Danger Tier Distribution\nHIGH vs NORMAL", fontweight='bold', fontsize=12)


# ── GRAPH 3: Average distance per object 
ax3 = fig.add_subplot(gs[0, 2])
avg_dist = (df[df["distance_m"] < 15]   # outliers hatane
            .groupby("object")["distance_m"]
            .mean()
            .sort_values()
            .head(10))
bars3 = ax3.bar(range(len(avg_dist)), avg_dist.values, color=BLUE_SHADES[1])
ax3.set_xticks(range(len(avg_dist)))
ax3.set_xticklabels(avg_dist.index, rotation=40, ha='right', fontsize=8)
ax3.set_title("Average Distance per Object\n(Top 10 closest)", fontweight='bold', fontsize=12)
ax3.set_ylabel("Distance (metres)")
for i, (bar, val) in enumerate(zip(bars3, avg_dist.values)):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{val:.1f}m", ha='center', fontsize=8)


# ── GRAPH 4: Direction distribution pie 
ax4 = fig.add_subplot(gs[1, 0])
dir_counts = df["direction"].value_counts()
dir_colors = {"left": "#F39C12", "center": "#3498DB", "right": "#9B59B6"}
ax4.pie(dir_counts.values,
        labels=dir_counts.index,
        colors=[dir_colors.get(d, "#95A5A6") for d in dir_counts.index],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2})
ax4.set_title("Object Direction Distribution\nLeft / Center / Right", fontweight='bold', fontsize=12)


# ── GRAPH 5: Detections over time (per minute) ──────────────
ax5 = fig.add_subplot(gs[1, 1])
df_time = df.copy()
df_time = df_time.set_index("timestamp")
per_minute = df_time.resample("1min").size()
ax5.plot(range(len(per_minute)), per_minute.values,
         color=BLUE_SHADES[1], linewidth=2, marker='o', markersize=4)
ax5.fill_between(range(len(per_minute)), per_minute.values,
                 alpha=0.2, color=BLUE_SHADES[1])
ax5.set_title("Detections Over Time\n(per minute)", fontweight='bold', fontsize=12)
ax5.set_xlabel("Time (minutes)")
ax5.set_ylabel("Detection Count")
ax5.set_xticks(range(len(per_minute)))
ax5.set_xticklabels([f"Min {i+1}" for i in range(len(per_minute))],
                    rotation=30, ha='right', fontsize=8)


# ── GRAPH 6: Confidence score histogram ─────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(df["confidence"], bins=20, color=BLUE_SHADES[2],
         edgecolor='white', linewidth=0.8)
ax6.axvline(df["confidence"].mean(), color=HIGH_COLOR,
            linestyle='--', linewidth=2,
            label=f'Mean: {df["confidence"].mean():.2f}')
ax6.set_title("Confidence Score Distribution\n(Model accuracy)", fontweight='bold', fontsize=12)
ax6.set_xlabel("Confidence Score (0–1)")
ax6.set_ylabel("Frequency")
ax6.legend(fontsize=9)


# ── SAVE + SHOW ──────────────────────────────────────────────
plt.savefig("analytics_dashboard.png", dpi=150, bbox_inches='tight')
print("\nDashboard saved: analytics_dashboard.png")
plt.show()



#  SUMMARY STATS — Mam ko bolne ke liye numbers

print("\n" + "="*55)
print("  SUMMARY — Mam ke saamine bolne ke liye")
print("="*55)
print(f"  Total detections recorded   : {len(df)}")
print(f"  Unique object types         : {df['object'].nunique()}")
print(f"  HIGH danger detections      : {(df['danger_tier']=='HIGH').sum()} "
      f"({(df['danger_tier']=='HIGH').mean()*100:.1f}%)")
print(f"  NORMAL detections           : {(df['danger_tier']=='NORMAL').sum()} "
      f"({(df['danger_tier']=='NORMAL').mean()*100:.1f}%)")
print(f"  Most detected object        : {df['object'].value_counts().index[0]} "
      f"({df['object'].value_counts().values[0]} times)")
print(f"  Average confidence score    : {df['confidence'].mean():.2f}")
print(f"  Average distance detected   : {df[df['distance_m']<15]['distance_m'].mean():.2f} metres")
print(f"  Closest detection ever      : {df['distance_m'].min():.2f} metres")
print("="*55)
print("\nYeh numbers mam ko bolo — bahut impressive lagenge! 😎")
