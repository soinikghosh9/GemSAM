import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix

def plot_roc_curve(y_true, y_scores, class_names, output_dir):
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(class_names):
        # Handle cases where a class might not be present in the test set
        if np.sum(y_true[:, i]) == 0:
            continue
            
        fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=16)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_pr_curve(y_true, y_scores, class_names, output_dir):
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(class_names):
        if np.sum(y_true[:, i]) == 0:
            continue
            
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        ap = average_precision_score(y_true[:, i], y_scores[:, i])
        plt.plot(recall, precision, lw=2, label=f'{class_name} (AP = {ap:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=16)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix_heatmap(y_true, y_pred, class_names, output_dir, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.title(title, fontsize=16)
    
    plt.savefig(os.path.join(output_dir, f'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_segmentation_metrics(class_names, dice_scores, iou_scores, output_dir):
    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, dice_scores, width, label='Dice Score', color='teal')
    rects2 = ax.bar(x + width/2, iou_scores, width, label='IoU', color='coral')

    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('SAM2 Segmentation Performance by Pathology', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)

    plt.savefig(os.path.join(output_dir, 'segmentation_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_vqa_radar_chart(categories, scores, output_dir):
    # Number of variables
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Needs to be closed loop
    scores_closed = list(scores) + [scores[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, size=12)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
    plt.ylim(0, 1.0)

    # Plot data
    ax.plot(angles, scores_closed, linewidth=2, linestyle='solid', color='purple')
    ax.fill(angles, scores_closed, 'purple', alpha=0.25)
    
    plt.title('Clinical Reasoning (VQA) Accuracy by Modality', size=16, y=1.1)

    plt.savefig(os.path.join(output_dir, 'vqa_radar.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Scientific Plots for Manuscript")
    parser.add_argument("--output-dir", type=str, default="plots", help="Output directory for plots")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating sample plots in {args.output_dir}...")
    
    # Generate mock data to test the plotting functions
    np.random.seed(42)
    
    # 1. Classification Metrics (ROC/PR)
    classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]
    n_samples = 1000
    y_true_mock = np.random.randint(0, 2, size=(n_samples, len(classes)))
    y_scores_mock = y_true_mock * np.random.uniform(0.6, 1.0, size=(n_samples, len(classes))) + \
                    (1 - y_true_mock) * np.random.uniform(0.0, 0.5, size=(n_samples, len(classes)))
                    
    plot_roc_curve(y_true_mock, y_scores_mock, classes, args.output_dir)
    plot_pr_curve(y_true_mock, y_scores_mock, classes, args.output_dir)
    
    # 2. Modality Confusion Matrix
    modalities = ['X-ray', 'CT', 'MRI', 'Ultrasound']
    y_true_modals = np.random.choice(len(modalities), 200)
    # 90% accuracy mock
    y_pred_modals = np.where(np.random.rand(200) > 0.1, y_true_modals, np.random.choice(len(modalities), 200))
    plot_confusion_matrix_heatmap(y_true_modals, y_pred_modals, modalities, args.output_dir, "Modality Classification Confusion Matrix")
    
    # 3. Segmentation Metrics
    seg_classes = ["Mass", "Effusion", "Cardiomegaly", "Infiltration", "Tumor"]
    dice = [0.88, 0.92, 0.95, 0.81, 0.89]
    iou = [0.79, 0.85, 0.91, 0.72, 0.80]
    plot_segmentation_metrics(seg_classes, dice, iou, args.output_dir)
    
    # 4. VQA Radar
    vqa_cats = ["Anatomy Recall", "Pathology Identification", "Severity Grading", "Localization", "Clinical Recommendations"]
    vqa_scores = [0.94, 0.88, 0.76, 0.91, 0.82]
    plot_vqa_radar_chart(vqa_cats, vqa_scores, args.output_dir)
    
    print("Plots generated successfully!")
