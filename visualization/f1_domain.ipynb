import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Read .csv file
df = pd.read_csv('f1_domain.csv', sep='\t')  # 改成你实际的路径

# Step 2: Group models
def model_group(model):
    model = model.strip()
    if model in ["zero-shot Llama-3.1-70B-Instruct","zero-shot Mistral-Small-24B-Instruct-2501","zero-shot Gemma-3-27B-it","zero-shot Qwen-2.5-72B-Instruct"]:
        return "Zero-shot LLMs"
    elif "SciBERT" in model:
        return "Fine-tuned SciBERT"
    elif "fine-tuned Qwen-2.5-14B-Instruct" in model:
        return "Fine-tuned Qwen small"
    else:
        print(model)
        return "Other"

df["Model Group"] = df["Model"].apply(model_group)

# Step 3: Avg. of models
grouped_df = df.groupby(["Schema", "Model Group"])[["ACL-ARC Marco F1", "Cross-domain (ACT2) Marco F1"]].mean().reset_index()
grouped_df[["ACL-ARC Marco F1", "Cross-domain (ACT2) Marco F1"]] = grouped_df[["ACL-ARC Marco F1", "Cross-domain (ACT2) Marco F1"]].round(2)

# Step 4: Dumbbell plt
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3.2), sharey=True)
schema_order = ["Citation Intent", "Cited Content Type", "SciCite-3 types", "ACL-ARC-6 types"]

for ax, (data, title) in zip(axes, [
    (grouped_df[grouped_df["Model Group"] == "Fine-tuned SciBERT"], "Fine-tuned SciBERT"),
    (grouped_df[grouped_df["Model Group"] == "Zero-shot LLMs"], "Zero-shot LLMs (Avg.)"),
    (grouped_df[grouped_df["Model Group"] == "Fine-tuned Qwen small"], "Fine-tuned Qwen small")
]):
    data = data.copy()
    data["Schema"] = pd.Categorical(data["Schema"], categories=schema_order, ordered=True)
    data = data.sort_values("Schema")
    y_labels = data["Schema"]
    y_pos = range(len(data))

    ax.hlines(y=y_pos, xmin=data["Cross-domain (ACT2) Marco F1"], xmax=data["ACL-ARC Marco F1"], color='gray', alpha=0.6)

    ax.axvline(x=0.2, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=0.4, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=0.6, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=0.8, color='gray', linestyle='--', linewidth=0.5)

    ax.scatter(data["ACL-ARC Marco F1"], y_pos, color='blue', label='ACL-ARC', zorder=3, marker='o')
    ax.scatter(data["Cross-domain (ACT2) Marco F1"], y_pos, color='red', label='ACT2', zorder=3, marker='x', s=50)
    for j, (_, row) in enumerate(data.iterrows()):
        ax.text(row["ACL-ARC Marco F1"] + 0.025, y_pos[j], f"{row['ACL-ARC Marco F1']:.2f}", 
            va='center', ha='left', fontsize=11, color='blue')
        ax.text(row["Cross-domain (ACT2) Marco F1"] - 0.02, y_pos[j], f"{row['Cross-domain (ACT2) Marco F1']:.2f}", 
            va='center', ha='right', fontsize=11, color='red')
    ax.set_yticks(list(y_pos))
    if ax == axes[0]:
        ax.set_yticklabels(y_labels, fontsize=12)
    else:
        ax.set_yticklabels([])
    ax.set_xlim(0, 0.85)
    ax.set_title(title, fontsize=12, weight='medium', loc='center', pad=15)
    #

# Y轴统一标注 + 图例
ax.set_yticklabels(y_labels)
#ax.set_xlabel("Macro F1 Score", fontsize=15, weight='bold', labelpad=10, loc='center')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', fontsize=11.5, bbox_to_anchor=(0.495, 0.46), bbox_transform=fig.transFigure)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("cross_domain_dumbbell.pdf", format="pdf", dpi=300, bbox_inches='tight')

plt.show()
