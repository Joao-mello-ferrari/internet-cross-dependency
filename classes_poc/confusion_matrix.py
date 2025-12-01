import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =======================
# CLASS LABELS
# =======================
class_labels = {
    1: 'Critical services',
    2: 'News',
    3: 'General Digital services',
    4: 'Entertainment'
}


# =======================
# LOAD INPUT DATA
# =======================
df = pd.read_csv('poc_grouped.csv')

# Remove records where manual classification is 5
df_filtered = df[df['class from me'] != 5].copy()

print(f"Total original records: {len(df)}")
print(f"Records after removing class 5: {len(df_filtered)}")
print(f"\nClass distribution (my classification):")
print(df_filtered['class from me'].value_counts().sort_index())


# =======================
# BALANCED SAMPLING
# =======================
df_balanced = pd.DataFrame()

for class_num in [1, 2, 3, 4]:
    class_data = df_filtered[df_filtered['class from me'] == class_num]
    
    if len(class_data) >= 200:
        sampled = class_data.sample(n=200, random_state=42)
    else:
        print(f"\nWarning: Class {class_num} has only {len(class_data)} records (< 200)")
        sampled = class_data
    
    df_balanced = pd.concat([df_balanced, sampled], ignore_index=True)

print(f"\nTotal balanced records: {len(df_balanced)}")
print(f"\nFinal distribution:")
print(df_balanced['class from me'].value_counts().sort_index())

# Extract classifications
y_true = df_balanced['class from me'].values
y_pred = df_balanced['class from llm'].values


# =======================
# CONFUSION MATRIX
# =======================
def create_confusion_matrix(y_true, y_pred, labels):
    n_classes = len(labels)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    for true, pred in zip(y_true, y_pred):
        true_idx = label_to_idx[true]
        pred_idx = label_to_idx[pred]
        cm[true_idx, pred_idx] += 1
    
    return cm

# Calculate confusion matrix
cm = create_confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4])

# Calculate accuracy
accuracy = np.trace(cm) / np.sum(cm)

print(f"\n{'='*60}")
print(f"OVERALL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*60}")


# =======================
# CLASSIFICATION REPORT
# =======================
print("\nClassification Report:")
print("-" * 80)
print(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
print("-" * 80)

for i, class_num in enumerate([1, 2, 3, 4]):
    # True Positives, False Positives, False Negatives
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    support = cm[i, :].sum()
    
    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{class_labels[class_num]:<30} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<8}")

print("-" * 80)

# Macro average
precisions = []
recalls = []
f1s = []
total_support = 0

for i in range(4):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    support = cm[i, :].sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    total_support += support

print(f"{'Macro avg':<30} {np.mean(precisions):<12.4f} {np.mean(recalls):<12.4f} {np.mean(f1s):<12.4f} {total_support:<8}")
print(f"{'Weighted avg':<30} {accuracy:<12.4f} {accuracy:<12.4f} {accuracy:<12.4f} {total_support:<8}")
print("-" * 80)


# =======================
# PLOT CONFUSION MATRIX
# =======================
plt.figure(figsize=(6, 5))

# Normalize by row (each row sums to 100%)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot normalized matrix
sns.heatmap(cm_normalized, 
            annot=True, 
            fmt='.2%', 
            cmap='viridis',
            xticklabels=[class_labels[i] for i in [1, 2, 3, 4]],
            yticklabels=[class_labels[i] for i in [1, 2, 3, 4]],
            cbar_kws={'label': 'Proportion'},
            linewidths=1,
            linecolor='white',
            square=True,
            annot_kws={'size': 9},
            vmin=0,
            vmax=1)

plt.ylabel('Real class', fontsize=12)

# Move x-axis to top
ax = plt.gca()
ax.xaxis.tick_top()
plt.xlabel('Predicted class (by LLM)', fontsize=12)

# Rotate x-axis labels and keep y-axis horizontal, color x labels blue
plt.xticks(rotation=45, ha='left', fontsize=7, color='blue')
plt.yticks(rotation=0, fontsize=7)

plt.tight_layout()
plt.savefig('confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
print("\nNormalized confusion matrix saved as 'confusion_matrix_normalized.png'")

# Show matrices
plt.show()


# =======================
# DETAILED ANALYSIS
# =======================
print("\n" + "="*60)
print("Detailed analysis by class:")
print("="*60)

for i in [1, 2, 3, 4]:
    total = cm[i-1].sum()
    correct = cm[i-1, i-1]
    accuracy_class = correct / total if total > 0 else 0
    print(f"\n{class_labels[i]}:")
    print(f"  Total: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy_class:.4f} ({accuracy_class*100:.2f}%)")
    
    # Show main errors
    errors = [(j+1, cm[i-1, j]) for j in range(4) if j != i-1 and cm[i-1, j] > 0]
    if errors:
        errors.sort(key=lambda x: x[1], reverse=True)
        print(f"  Main errors:")
        for class_num, count in errors[:3]:
            print(f"    - Classified as {class_labels[class_num]}: {count} ({count/total*100:.1f}%)")
