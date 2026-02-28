"""
================================================================
Day 08 — Longitudinal Microbiome Stability Analysis (REAL DATA)
Author  : Subhadip Jana
Dataset : peerj32 — LGG Probiotic vs Placebo
          44 samples | 22 subjects × 2 time points (T1, T2)
          130 real gut taxa

Research Questions:
  1. How stable is each subject's gut microbiome T1→T2?
  2. Does LGG probiotic destabilize or stabilize the microbiome?
  3. Which taxa are the most stable vs most variable?
  4. Are there "resilient" vs "susceptible" individuals?
  5. Does baseline diversity predict stability?

Methods:
  • Intra-individual Bray-Curtis distance (T1→T2 per subject)
  • Aitchison distance (log-ratio based)
  • Per-taxon coefficient of variation (CV)
  • Stability score per subject
  • Baseline diversity → stability regression
  • Responder classification (stable vs shifter)
  • Taxa stability ranking
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.spatial.distance import braycurtis
from scipy.stats import spearmanr, mannwhitneyu, wilcoxon
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# SECTION 1: LOAD DATA
# ─────────────────────────────────────────────────────────────

print("🔬 Loading peerj32 dataset...")
otu_raw = pd.read_csv("data/otu_table.csv", index_col=0)
meta    = pd.read_csv("data/metadata.csv",  index_col=0)

otu_df  = otu_raw.T.astype(float)
taxa    = otu_df.columns.tolist()
rel_df  = otu_df.div(otu_df.sum(axis=1), axis=0) * 100

group   = meta["group"]
time    = meta["time"].astype(str)
subject = meta["subject"]

print(f"✅ {len(otu_df)} samples × {len(taxa)} taxa")
print(f"   Subjects : {subject.nunique()} (each measured at T1 + T2)")
print(f"   LGG      : {(group=='LGG').sum()} samples | "
      f"Placebo: {(group=='Placebo').sum()} samples")

# ─────────────────────────────────────────────────────────────
# SECTION 2: ALPHA DIVERSITY (needed for downstream analysis)
# ─────────────────────────────────────────────────────────────

def shannon(row):
    c = row.values.astype(float); c = c[c>0]; p = c/c.sum()
    return -np.sum(p * np.log(p))

def observed_otus(row):
    return int(np.sum(row.values > 0))

meta["Shannon"] = rel_df.apply(shannon, axis=1).values
meta["OTUs"]    = rel_df.apply(observed_otus, axis=1).values

# ─────────────────────────────────────────────────────────────
# SECTION 3: INTRA-INDIVIDUAL STABILITY (T1→T2 distance)
# ─────────────────────────────────────────────────────────────

print("\n📐 Computing intra-individual stability...")

stability_records = []

for subj in subject.unique():
    subj_meta = meta[meta["subject"] == subj]
    if len(subj_meta) < 2:
        continue

    t1_idx = subj_meta[subj_meta["time"] == 1].index
    t2_idx = subj_meta[subj_meta["time"] == 2].index

    if len(t1_idx) == 0 or len(t2_idx) == 0:
        continue

    t1_sample = t1_idx[0]
    t2_sample = t2_idx[0]

    t1_vec = rel_df.loc[t1_sample].values / 100  # proportions
    t2_vec = rel_df.loc[t2_sample].values / 100

    # Bray-Curtis dissimilarity (0=identical, 1=completely different)
    bc_dist = braycurtis(t1_vec, t2_vec)

    # Stability score (inverse of distance, 0–1 where 1=perfectly stable)
    stability = 1 - bc_dist

    # Aitchison distance (CLR-based)
    eps = 1e-6
    t1_clr = np.log(t1_vec + eps) - np.log(t1_vec + eps).mean()
    t2_clr = np.log(t2_vec + eps) - np.log(t2_vec + eps).mean()
    aitchison = np.sqrt(np.sum((t1_clr - t2_clr)**2))

    # Per-subject diversity at T1 (baseline)
    t1_shannon = meta.loc[t1_sample, "Shannon"]
    t2_shannon = meta.loc[t2_sample, "Shannon"]

    grp = meta.loc[t1_sample, "group"]

    stability_records.append({
        "Subject"     : subj,
        "Group"       : grp,
        "T1_sample"   : t1_sample,
        "T2_sample"   : t2_sample,
        "BC_distance" : round(bc_dist,    4),
        "Stability"   : round(stability,  4),
        "Aitchison"   : round(aitchison,  4),
        "T1_Shannon"  : round(t1_shannon, 4),
        "T2_Shannon"  : round(t2_shannon, 4),
        "Shannon_change": round(t2_shannon - t1_shannon, 4),
    })

stab_df = pd.DataFrame(stability_records)

# Classify subjects: stable vs shifter (median split)
median_bc = stab_df["BC_distance"].median()
stab_df["Stability_class"] = stab_df["BC_distance"].apply(
    lambda x: "Stable" if x <= median_bc else "Shifter"
)

stab_df.to_csv("outputs/stability_results.csv", index=False)
print(f"✅ Stability computed for {len(stab_df)} subjects")
print(f"\n   Median BC distance : {median_bc:.4f}")
print(f"   Stable subjects    : {(stab_df['Stability_class']=='Stable').sum()}")
print(f"   Shifter subjects   : {(stab_df['Stability_class']=='Shifter').sum()}")

# ─────────────────────────────────────────────────────────────
# SECTION 4: STATISTICAL TESTS
# ─────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("STATISTICAL RESULTS")
print("="*55)

lgg_stab = stab_df[stab_df["Group"]=="LGG"]["BC_distance"].values
plc_stab = stab_df[stab_df["Group"]=="Placebo"]["BC_distance"].values

_, p_group = mannwhitneyu(lgg_stab, plc_stab, alternative="two-sided")
print(f"\n[1] LGG vs Placebo BC distance (stability):")
print(f"    LGG     mean BC: {lgg_stab.mean():.4f} ± {lgg_stab.std():.4f}")
print(f"    Placebo mean BC: {plc_stab.mean():.4f} ± {plc_stab.std():.4f}")
print(f"    p={p_group:.4f} {'*' if p_group<0.05 else 'ns'}")

# Baseline Shannon → stability correlation
r_base, p_base = spearmanr(stab_df["T1_Shannon"], stab_df["BC_distance"])
print(f"\n[2] Baseline Shannon vs BC distance:")
print(f"    Spearman r={r_base:.4f}, p={p_base:.4f} {'*' if p_base<0.05 else 'ns'}")

# Shannon change T1→T2
_, p_shannon_change = wilcoxon(stab_df["T1_Shannon"], stab_df["T2_Shannon"])
print(f"\n[3] Shannon T1 vs T2 (paired Wilcoxon):")
print(f"    T1 mean: {stab_df['T1_Shannon'].mean():.4f}")
print(f"    T2 mean: {stab_df['T2_Shannon'].mean():.4f}")
print(f"    p={p_shannon_change:.4f} {'*' if p_shannon_change<0.05 else 'ns'}")

# Stable vs Shifter Shannon
stable_bc  = stab_df[stab_df["Stability_class"]=="Stable"]["T1_Shannon"].values
shifter_bc = stab_df[stab_df["Stability_class"]=="Shifter"]["T1_Shannon"].values
_, p_class = mannwhitneyu(stable_bc, shifter_bc, alternative="two-sided")
print(f"\n[4] Stable vs Shifter baseline Shannon:")
print(f"    Stable  mean: {stable_bc.mean():.4f}")
print(f"    Shifter mean: {shifter_bc.mean():.4f}")
print(f"    p={p_class:.4f} {'*' if p_class<0.05 else 'ns'}")

# ─────────────────────────────────────────────────────────────
# SECTION 5: PER-TAXON STABILITY
# ─────────────────────────────────────────────────────────────

print("\n🔬 Computing per-taxon stability...")

taxon_stability = []
for taxon in taxa:
    t1_vals, t2_vals = [], []
    for _, row in stab_df.iterrows():
        t1_vals.append(rel_df.loc[row["T1_sample"], taxon])
        t2_vals.append(rel_df.loc[row["T2_sample"], taxon])

    t1_arr = np.array(t1_vals)
    t2_arr = np.array(t2_vals)
    all_vals = np.concatenate([t1_arr, t2_arr])

    # Coefficient of variation
    cv = (all_vals.std() / (all_vals.mean() + 1e-9)) * 100
    # Mean change T1→T2
    mean_change = np.mean(np.abs(t2_arr - t1_arr))
    # Correlation T1 vs T2 (stability across subjects)
    if t1_arr.std() > 0 and t2_arr.std() > 0:
        r, _ = spearmanr(t1_arr, t2_arr)
    else:
        r = 1.0

    taxon_stability.append({
        "Taxon"       : taxon,
        "Mean_abund"  : round(all_vals.mean(), 4),
        "CV_pct"      : round(cv, 2),
        "Mean_change" : round(mean_change, 4),
        "T1_T2_corr"  : round(r, 4),
        "Stability"   : round(r, 4),  # high r = stable taxon
    })

taxon_stab_df = pd.DataFrame(taxon_stability)
taxon_stab_df = taxon_stab_df.sort_values("T1_T2_corr", ascending=False)
taxon_stab_df.to_csv("outputs/taxon_stability.csv", index=False)

top_stable   = taxon_stab_df[taxon_stab_df["Mean_abund"] > 0.5].head(15)
top_variable = taxon_stab_df[taxon_stab_df["Mean_abund"] > 0.5].tail(15)

print(f"\nTop 5 most STABLE taxa (high T1-T2 correlation):")
print(top_stable[["Taxon","T1_T2_corr","CV_pct"]].head(5).to_string(index=False))
print(f"\nTop 5 most VARIABLE taxa (low T1-T2 correlation):")
print(top_variable[["Taxon","T1_T2_corr","CV_pct"]].head(5).to_string(index=False))

# ─────────────────────────────────────────────────────────────
# SECTION 6: DASHBOARD
# ─────────────────────────────────────────────────────────────

print("\n🎨 Generating dashboard...")

PAL_GROUP = {"LGG": "#E74C3C", "Placebo": "#3498DB"}
PAL_CLASS = {"Stable": "#2ECC71", "Shifter": "#E74C3C"}

fig = plt.figure(figsize=(22, 18))
fig.suptitle(
    "Longitudinal Microbiome Stability Analysis — REAL DATA\n"
    "LGG Probiotic vs Placebo | peerj32 dataset\n"
    "22 subjects × 2 time points (Before / After intervention)",
    fontsize=15, fontweight="bold", y=0.99
)

# ── Plot 1: BC distance per subject (ranked) ──
ax1 = fig.add_subplot(3, 3, 1)
stab_sorted = stab_df.sort_values("BC_distance")
colors_bar  = [PAL_GROUP[g] for g in stab_sorted["Group"]]
bars = ax1.bar(range(len(stab_sorted)),
               stab_sorted["BC_distance"].values,
               color=colors_bar, edgecolor="black", linewidth=0.5, alpha=0.85)
ax1.axhline(median_bc, color="black", lw=2, linestyle="--",
            label=f"Median={median_bc:.3f}")
ax1.axhline(stab_df[stab_df["Group"]=="LGG"]["BC_distance"].mean(),
            color="#E74C3C", lw=1.5, linestyle=":", label="LGG mean")
ax1.axhline(stab_df[stab_df["Group"]=="Placebo"]["BC_distance"].mean(),
            color="#3498DB", lw=1.5, linestyle=":", label="Placebo mean")
ax1.set_xticks(range(len(stab_sorted)))
ax1.set_xticklabels(stab_sorted["Subject"].values, fontsize=7, rotation=45)
ax1.set_ylabel("Bray-Curtis Distance (T1→T2)")
ax1.set_title("Intra-individual Stability\n(lower = more stable)",
              fontweight="bold", fontsize=10)
ax1.legend(fontsize=7)
lgg_patch = mpatches.Patch(color="#E74C3C", label="LGG")
plc_patch = mpatches.Patch(color="#3498DB", label="Placebo")
ax1.legend(handles=[lgg_patch, plc_patch], fontsize=8, loc="upper left")

# ── Plot 2: BC distance LGG vs Placebo boxplot ──
ax2 = fig.add_subplot(3, 3, 2)
sns.boxplot(data=stab_df, x="Group", y="BC_distance",
            palette=PAL_GROUP, order=["LGG","Placebo"],
            width=0.4, ax=ax2, linewidth=1.5,
            flierprops={"marker":"o","markersize":5})
sns.stripplot(data=stab_df, x="Group", y="BC_distance",
              palette=PAL_GROUP, order=["LGG","Placebo"],
              size=8, jitter=True, alpha=0.8, ax=ax2)
for i, g in enumerate(["LGG","Placebo"]):
    mv = stab_df[stab_df["Group"]==g]["BC_distance"].mean()
    ax2.hlines(mv, i-0.2, i+0.2, colors="black", lw=2.5)
    ax2.text(i, mv+0.005, f"μ={mv:.3f}", ha="center",
             fontsize=10, fontweight="bold")
ax2.set_title(f"Stability: LGG vs Placebo\np={p_group:.4f} "
              f"{'*' if p_group<0.05 else 'ns'}",
              fontweight="bold", fontsize=10)
ax2.set_ylabel("BC Distance (T1→T2)"); ax2.set_xlabel("")

# ── Plot 3: Shannon T1 vs T2 paired lines ──
ax3 = fig.add_subplot(3, 3, 3)
for _, row in stab_df.iterrows():
    color = PAL_GROUP[row["Group"]]
    ax3.plot(["T1","T2"], [row["T1_Shannon"], row["T2_Shannon"]],
             color=color, alpha=0.4, linewidth=1.5, marker="o", markersize=5)
for g, color in PAL_GROUP.items():
    sub = stab_df[stab_df["Group"]==g]
    ax3.plot(["T1","T2"],
             [sub["T1_Shannon"].mean(), sub["T2_Shannon"].mean()],
             color=color, linewidth=4, marker="o",
             markersize=12, label=f"{g} mean", zorder=5)
ax3.set_ylabel("Shannon Entropy")
ax3.set_title(f"Shannon Diversity T1→T2\nPaired p={p_shannon_change:.4f} "
              f"{'*' if p_shannon_change<0.05 else 'ns'}",
              fontweight="bold", fontsize=10)
ax3.legend(fontsize=9)

# ── Plot 4: Baseline Shannon vs Stability ──
ax4 = fig.add_subplot(3, 3, 4)
for g, color in PAL_GROUP.items():
    sub = stab_df[stab_df["Group"]==g]
    ax4.scatter(sub["T1_Shannon"], sub["BC_distance"],
                c=color, label=g, s=90, alpha=0.85,
                edgecolors="white", linewidth=0.8)
    # Subject labels
    for _, row in sub.iterrows():
        ax4.annotate(row["Subject"],
                     (row["T1_Shannon"], row["BC_distance"]),
                     fontsize=6, xytext=(3,3), textcoords="offset points")
z   = np.polyfit(stab_df["T1_Shannon"], stab_df["BC_distance"], 1)
p_  = np.poly1d(z)
x_l = np.linspace(stab_df["T1_Shannon"].min(),
                  stab_df["T1_Shannon"].max(), 100)
ax4.plot(x_l, p_(x_l), "k--", lw=1.5, alpha=0.6)
ax4.set_xlabel("Baseline Shannon (T1)")
ax4.set_ylabel("BC Distance (T1→T2)")
ax4.set_title(f"Does Diversity Predict Stability?\n"
              f"Spearman r={r_base:.3f}, p={p_base:.4f}",
              fontweight="bold", fontsize=10)
ax4.legend(fontsize=9)

# ── Plot 5: Aitchison distance comparison ──
ax5 = fig.add_subplot(3, 3, 5)
sns.violinplot(data=stab_df, x="Group", y="Aitchison",
               palette=PAL_GROUP, inner=None, alpha=0.4,
               order=["LGG","Placebo"], ax=ax5)
sns.stripplot(data=stab_df, x="Group", y="Aitchison",
              palette=PAL_GROUP, size=8, jitter=True,
              order=["LGG","Placebo"], alpha=0.8, ax=ax5)
_, p_ait = mannwhitneyu(
    stab_df[stab_df["Group"]=="LGG"]["Aitchison"].values,
    stab_df[stab_df["Group"]=="Placebo"]["Aitchison"].values)
ax5.set_title(f"Aitchison Distance T1→T2\np={p_ait:.4f} "
              f"{'*' if p_ait<0.05 else 'ns'}",
              fontweight="bold", fontsize=10)
ax5.set_ylabel("Aitchison Distance"); ax5.set_xlabel("")

# ── Plot 6: Stable vs Shifter — Shannon boxplot ──
ax6 = fig.add_subplot(3, 3, 6)
sns.boxplot(data=stab_df, x="Stability_class", y="T1_Shannon",
            palette=PAL_CLASS, order=["Stable","Shifter"],
            width=0.4, ax=ax6, linewidth=1.5,
            flierprops={"marker":"o","markersize":5})
sns.stripplot(data=stab_df, x="Stability_class", y="T1_Shannon",
              hue=stab_df["Group"].values, palette=PAL_GROUP,
              order=["Stable","Shifter"],
              size=8, jitter=True, alpha=0.8, ax=ax6,
              legend=False)
ax6.set_title(f"Stable vs Shifter Baseline Shannon\n"
              f"p={p_class:.4f} {'*' if p_class<0.05 else 'ns'}",
              fontweight="bold", fontsize=10)
ax6.set_ylabel("Baseline Shannon (T1)"); ax6.set_xlabel("")

# ── Plot 7: Top stable taxa ──
ax7 = fig.add_subplot(3, 3, 7)
plot_stable = top_stable.head(12)
colors_s = ["#2ECC71" if r > 0.6 else "#F39C12" if r > 0.3
            else "#E74C3C" for r in plot_stable["T1_T2_corr"]]
ax7.barh(range(len(plot_stable)),
         plot_stable["T1_T2_corr"].values[::-1],
         color=colors_s[::-1], edgecolor="black", linewidth=0.4)
ax7.set_yticks(range(len(plot_stable)))
ax7.set_yticklabels([t.replace("et rel.","").strip()[:26]
                     for t in plot_stable["Taxon"][::-1]], fontsize=7)
ax7.axvline(0.6, color="green", lw=1, linestyle="--", alpha=0.7, label="r=0.6")
ax7.axvline(0.3, color="orange",lw=1, linestyle="--", alpha=0.7, label="r=0.3")
ax7.set_xlabel("T1–T2 Spearman Correlation (stability)")
ax7.set_title("Most Stable Taxa\n(abundant, high T1-T2 correlation)",
              fontweight="bold", fontsize=10)
ax7.legend(fontsize=8)

# ── Plot 8: Most variable taxa (CV) ──
ax8 = fig.add_subplot(3, 3, 8)
top_var = taxon_stab_df[taxon_stab_df["Mean_abund"] > 0.1].nlargest(12, "CV_pct")
colors_v = ["#E74C3C" if cv > 150 else "#F39C12" if cv > 100
            else "#2ECC71" for cv in top_var["CV_pct"]]
ax8.barh(range(len(top_var)),
         top_var["CV_pct"].values[::-1],
         color=colors_v[::-1], edgecolor="black", linewidth=0.4)
ax8.set_yticks(range(len(top_var)))
ax8.set_yticklabels([t.replace("et rel.","").strip()[:26]
                     for t in top_var["Taxon"][::-1]], fontsize=7)
ax8.set_xlabel("Coefficient of Variation (%)")
ax8.set_title("Most Variable Taxa\n(high CV across samples)",
              fontweight="bold", fontsize=10)
ax8.axvline(100, color="orange", lw=1, linestyle="--", alpha=0.7)
ax8.axvline(150, color="red",    lw=1, linestyle="--", alpha=0.7)

# ── Plot 9: Summary table ──
ax9 = fig.add_subplot(3, 3, 9)
ax9.axis("off")
rows = [
    ["LGG — mean BC dist",     f"{lgg_stab.mean():.4f}",
     f"±{lgg_stab.std():.4f}", ""],
    ["Placebo — mean BC dist", f"{plc_stab.mean():.4f}",
     f"±{plc_stab.std():.4f}", ""],
    ["LGG vs Placebo",         f"p={p_group:.4f}",
     "*" if p_group<0.05 else "ns", ""],
    ["Shannon T1 vs T2",       f"p={p_shannon_change:.4f}",
     "*" if p_shannon_change<0.05 else "ns", ""],
    ["Diversity→Stability r",  f"{r_base:.4f}",
     f"p={p_base:.4f}",""],
    ["Stable subjects",        str((stab_df["Stability_class"]=="Stable").sum()),
     "", ""],
    ["Shifter subjects",       str((stab_df["Stability_class"]=="Shifter").sum()),
     "", ""],
    ["Median BC threshold",    f"{median_bc:.4f}", "", ""],
    ["Most stable taxon",
     top_stable["Taxon"].iloc[0].replace("et rel.","").strip()[:22],"",""],
    ["Most variable taxon",
     top_var["Taxon"].iloc[0].replace("et rel.","").strip()[:22],"",""],
]
tbl = ax9.table(
    cellText=rows,
    colLabels=["Metric","Value","Stat",""],
    cellLoc="center", loc="center"
)
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1.4, 1.9)
for j in range(4): tbl[(0,j)].set_facecolor("#BDC3C7")
tbl[(1,0)].set_facecolor("#E74C3C"); tbl[(1,0)].set_text_props(color="white",fontweight="bold")
tbl[(2,0)].set_facecolor("#3498DB"); tbl[(2,0)].set_text_props(color="white",fontweight="bold")
ax9.set_title("Stability Summary", fontweight="bold", fontsize=11, pad=20)

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig("outputs/longitudinal_stability_dashboard.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("✅ Dashboard saved → outputs/longitudinal_stability_dashboard.png")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("FINAL SUMMARY")
print("="*55)
print(f"\nSubjects analysed : {len(stab_df)}")
print(f"Stable subjects   : {(stab_df['Stability_class']=='Stable').sum()}")
print(f"Shifter subjects  : {(stab_df['Stability_class']=='Shifter').sum()}")
print(f"\nLGG mean BC dist    : {lgg_stab.mean():.4f} ± {lgg_stab.std():.4f}")
print(f"Placebo mean BC dist: {plc_stab.mean():.4f} ± {plc_stab.std():.4f}")
print(f"LGG vs Placebo p    : {p_group:.4f}")
print(f"\nBaseline diversity → stability r={r_base:.4f} p={p_base:.4f}")
print(f"\nMost stable taxon  : {top_stable['Taxon'].iloc[0]}")
print(f"Most variable taxon: {top_var['Taxon'].iloc[0]}")
print("\n✅ Block 1 (Days 01-08) COMPLETE! 🎉")
