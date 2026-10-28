import matplotlib.pyplot as plt
import numpy as np
 
# Real measured latency values from Scenario A (audit_three_articles_results.json)
stages = [
    'Decision Fusion',
    'Source Assessment',
    'Text ML Model',
    'Claim-Evidence Matching',
    'Article Extraction',
    'Claim Decomposition',
    'External Evidence Retrieval'
]
latencies_ms = [0.03, 0.34, 0.56, 0.69, 129.69, 5086.04, 8474.88]
 
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(10, 5.5))
 
y_pos = np.arange(len(stages))
colors = ['#10b981', '#10b981', '#10b981', '#10b981', '#3b82f6', '#f59e0b', '#ef4444']
 
bars = ax.barh(y_pos, latencies_ms, color=colors, edgecolor='none', height=0.6)
 
# Formatting log scale due to 6 orders of magnitude variance (0.03ms to 8474ms)
ax.set_xscale('log')
ax.set_yticks(y_pos)
ax.set_yticklabels(stages, fontsize=11, fontweight='medium')
ax.set_xlabel('Execution Duration in Milliseconds (Logarithmic Scale)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('Figure 7.1: VeritasAI Latency Profile Across Verification Stages', fontsize=14, fontweight='bold', pad=15)
 
# Annotate exact millisecond values
for bar, val in zip(bars, latencies_ms):
    ax.text(val * 1.2, bar.get_y() + bar.get_height()/2, f'{val:,.2f} ms', 
            va='center', ha='left', fontsize=10, fontweight='bold', color='#1f2937')
 
ax.set_xlim(0.01, 20000)
plt.tight_layout()
plt.savefig('figure_7_1_latency_breakdown.png', dpi=300)
plt.show()
