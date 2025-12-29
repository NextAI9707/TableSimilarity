import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, accuracy_score
import yaml
import json
import os
import argparse
from model import EnhancedTableSimilarityModel
from dataset import get_dataloader
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, accuracy_score, roc_curve, f1_score

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
class TableSimilarityEvaluator:
    """
    增强模型评估器：适配MySQL与配置驱动
    """

    def __init__(self, model_path: str, config_path: str = "config.yml"):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        # 加载增强模型（配置驱动）
        self.model = EnhancedTableSimilarityModel(config_path).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        # 加载测试数据（配置驱动）
        self.test_loader = get_dataloader(config_path=config_path, mode="test")

        # 加载标注
        annotations_path = self.config['data'].get('annotations_path', 'data/annotations.json')
        with open(annotations_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

    def compute_all_similarities(self):
        """计算所有测试对相似度（修复bug并增强）"""
        all_preds = []
        all_labels = []
        all_table_pairs = []
        all_weights = []
        all_table_ids = []

        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                # 数据迁移
                struct_a = batch['struct_a'].to(self.device)
                struct_b = batch['struct_b'].to(self.device)
                content_a = batch['content_a'].to(self.device)
                content_b = batch['content_b'].to(self.device)
                similarity = batch['similarity'].to(self.device)

                # 增强模型前向
                fused_a, fused_b, weights_a, weights_b, _, _ = self.model(
                    struct_a, content_a, struct_b, content_b
                )

                # 计算相似度
                pred_sim = self.model.compute_similarity(fused_a, fused_b).cpu().numpy()
                true_sim = batch['similarity'].numpy()

                # 记录权重
                if weights_a is not None:
                    weights = weights_a.cpu().numpy()
                    all_weights.extend(weights.tolist())

                all_preds.extend(pred_sim)
                all_labels.extend(true_sim)
                all_table_pairs.extend(list(zip(batch['table_a'], batch['table_b'])))

                # 记录table ID用于后续分析
                if 'table_id_a' in batch and 'table_id_b' in batch:
                    all_table_ids.extend(list(zip(batch['table_id_a'], batch['table_id_b'])))

        all_weights_array = np.array(all_weights) if all_weights else np.empty((0, 2))

        # 修复：返回labels作为table_sims（它们是同一数据）
        return np.array(all_preds), np.array(all_labels), all_table_pairs, \
            all_weights_array, np.array(all_labels)

    def calculate_metrics(self):
        """计算评估指标（增强版：增加ROC-AUC、F1-Score等）"""
        print("\n" + "=" * 60)
        print("开始计算评估指标...")
        print("=" * 60)

        preds, labels, _, weights, table_sims = self.compute_all_similarities()

        # 1. 基础回归指标
        mae = np.mean(np.abs(preds - labels))
        mse = np.mean((preds - labels) ** 2)
        rmse = np.sqrt(mse)

        # 2. 分类指标（二分类）
        threshold = 0.5
        binary_preds = (preds >= threshold).astype(int)
        binary_labels = (labels >= threshold).astype(int)
        accuracy = accuracy_score(binary_labels, binary_preds)

        # 新增：F1-Score
        f1 = f1_score(binary_labels, binary_preds, average='weighted', zero_division=0)

        # 新增：精确率、召回率
        from sklearn.metrics import precision_score, recall_score
        precision = precision_score(binary_labels, binary_preds, average='weighted', zero_division=0)
        recall = recall_score(binary_labels, binary_preds, average='weighted', zero_division=0)

        # 3. 排序指标
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
        }

        # 4. ROC-AUC
        if len(np.unique(binary_labels)) > 1:
            # 确保有正负样本
            fpr, tpr, _ = roc_curve(binary_labels, preds)
            roc_auc = auc(fpr, tpr)
            metrics['roc_auc'] = roc_auc
        else:
            metrics['roc_auc'] = 0.0

        # 5. 多尺度Recall和Precision
        for k in [10, 50, 100]:
            metrics[f'Recall@{k}'] = self.recall_at_k(preds, labels, k)
            metrics[f'Precision@{k}'] = self.precision_at_k(preds, labels, k)

        # 6. mAP和nDCG
        metrics['mAP@10'] = self.mean_average_precision(preds, labels, k=10)
        metrics['mAP@50'] = self.mean_average_precision(preds, labels, k=50)
        metrics['nDCG@10'] = self.ndcg_score(preds, labels, k=10)
        metrics['nDCG@50'] = self.ndcg_score(preds, labels, k=50)

        # 7. 权重动态性分析（增强）
        if len(weights) > 0:
            struct_weights = weights[:, 0]
            content_weights = weights[:, 1]

            print("\n📊 权重动态性分析:")
            print(f"  结构权重: μ={struct_weights.mean():.3f}, σ={struct_weights.std():.3f}")
            print(f"  内容权重: μ={content_weights.mean():.3f}, σ={content_weights.std():.3f}")

            metrics['Weight_Std'] = struct_weights.std()
            metrics['Struct_Weight_Mean'] = struct_weights.mean()
            metrics['Content_Weight_Mean'] = content_weights.mean()

            # 分析高相似样本的权重偏好
            high_sim_mask = labels >= 0.7
            if high_sim_mask.any():
                high_struct = struct_weights[high_sim_mask].mean()
                high_content = content_weights[high_sim_mask].mean()
                print(f"  高相似度样本: 结构权重={high_struct:.3f}, 内容权重={high_content:.3f}")
                metrics['HighSim_Struct_Weight'] = high_struct

            # 权重与表级相似度的相关性
            if len(table_sims) == len(struct_weights):
                correlation = np.corrcoef(table_sims, struct_weights)[0, 1]
                metrics['Sim-Weight_Correlation'] = correlation
                print(f"  权重-相似度相关性: {correlation:.3f} (期望>0.2)")

        # 打印结果
        print("\n" + "=" * 60)
        print("评估指标汇总:")
        print("\n【回归指标】")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")

        print("\n【分类指标】")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

        print("\n【排序指标】")
        for k in [10, 50, 100]:
            print(f"  Recall@{k}: {metrics[f'Recall@{k}']:.4f}")
            print(f"  Precision@{k}: {metrics[f'Precision@{k}']:.4f}")

        print("\n【权重动态性】")
        print(f"  Weight Std: {metrics.get('Weight_Std', 0):.4f}")
        print(f"  Sim-Weight Correlation: {metrics.get('Sim-Weight_Correlation', 0):.4f}")

        # 关键诊断
        if metrics.get('Weight_Std', 0) < 0.1:
            print("\n⚠️ 警告：权重标准差过低，门控网络未动态调整！")
            print("建议：检查DynamicPairGating的输入，确保table_sim信号")

        if metrics.get('Recall@10', 0) < 0.05:
            print("\n⚠️ 警告：Recall@10过低，模型过于保守！")
            print("建议：降低HardNegativeContrastiveLoss的温度参数")

        print("=" * 60)

        return metrics

    # 保留原有评估函数（完全不变）
    def precision_at_k(self, preds, labels, k=10):
        sorted_idx = np.argsort(-preds)[:k]
        relevant = labels[sorted_idx] >= 0.5
        return relevant.sum() / k

    def recall_at_k(self, preds, labels, k=10):
        sorted_idx = np.argsort(-preds)[:k]
        relevant_retrieved = labels[sorted_idx].sum()
        total_relevant = labels.sum()
        return relevant_retrieved / (total_relevant + 1e-8)

    def mean_average_precision(self, preds, labels, k=10):
        sorted_idx = np.argsort(-preds)[:k]
        sorted_labels = labels[sorted_idx]

        precisions = []
        num_relevant = 0

        for i, label in enumerate(sorted_labels):
            if label >= 0.5:
                num_relevant += 1
                precisions.append(num_relevant / (i + 1))

        return np.mean(precisions) if precisions else 0.0

    def ndcg_score(self, preds, labels, k=10):
        sorted_idx = np.argsort(-preds)[:k]
        sorted_labels = labels[sorted_idx]

        dcg = sum((2 ** label - 1) / np.log2(i + 2) for i, label in enumerate(sorted_labels))
        ideal_labels = np.sort(labels)[::-1][:k]
        idcg = sum((2 ** label - 1) / np.log2(i + 2) for i, label in enumerate(ideal_labels))

        return dcg / (idcg + 1e-8)

    def visualize_results(self):
        """可视化评估结果（增强版：9宫格图）"""
        print("\n" + "=" * 60)
        print("开始生成评估可视化...")
        print("=" * 60)

        preds, labels, _, weights, _ = self.compute_all_similarities()

        if len(weights) == 0:
            print("⚠️ 没有权重数据可供可视化")
            return

        metrics = self.calculate_metrics()

        # 创建更大的画布
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))  # 3x3布局

        # 1. 相似度分布对比
        axes[0, 0].hist(labels, bins=30, alpha=0.7, label='真实相似度', color='blue', density=True)
        axes[0, 0].hist(preds, bins=30, alpha=0.7, label='预测相似度', color='orange', density=True)
        axes[0, 0].set_xlabel('相似度分数', fontsize=11)
        axes[0, 0].set_ylabel('密度', fontsize=11)
        axes[0, 0].set_title('相似度分布对比\n(蓝色:真实,橙色:预测)', fontsize=12, fontweight='bold')
        axes[0, 0].legend(loc='upper right')
        axes[0, 0].grid(alpha=0.3)

        # 2. Precision-Recall曲线
        precisions, recalls, _ = precision_recall_curve(labels >= 0.5, preds)
        pr_auc = auc(recalls, precisions)
        axes[0, 1].plot(recalls, precisions, label=f'PR AUC = {pr_auc:.3f}', color='green', linewidth=2)
        axes[0, 1].set_xlabel('召回率', fontsize=11)
        axes[0, 1].set_ylabel('精确率', fontsize=11)
        axes[0, 1].set_title('Precision-Recall曲线', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # 3. ROC曲线
        if len(np.unique(labels >= 0.5)) > 1:
            fpr, tpr, _ = roc_curve(labels >= 0.5, preds)
            roc_auc = auc(fpr, tpr)
            axes[0, 2].plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}', color='red', linewidth=2)
            axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 2].set_xlabel('假正率', fontsize=11)
        axes[0, 2].set_ylabel('真正率', fontsize=11)
        axes[0, 2].set_title('ROC曲线', fontsize=12, fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].grid(alpha=0.3)

        # 4. 门控权重分布
        struct_weights = weights[:, 0]
        content_weights = weights[:, 1]
        weight_std = struct_weights.std()

        axes[1, 0].hist(struct_weights, bins=20, alpha=0.7,
                        label=f'结构权重\nμ={struct_weights.mean():.3f}\nσ={weight_std:.3f}',
                        color='purple', edgecolor='black')
        axes[1, 0].hist(content_weights, bins=20, alpha=0.7,
                        label=f'内容权重\nμ={content_weights.mean():.3f}',
                        color='orange', edgecolor='black')
        axes[1, 0].axvline(x=struct_weights.mean(), color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('权重值', fontsize=11)
        axes[1, 0].set_ylabel('频数', fontsize=11)
        axes[1, 0].set_title('门控权重分布\n(σ>0.1为有效动态)', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # 5. 权重-相似度散点图
        scatter = axes[1, 1].scatter(labels, struct_weights, c=labels, cmap='viridis', alpha=0.6, s=30)
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label('真实相似度', fontsize=10)
        axes[1, 1].set_xlabel('真实相似度', fontsize=11)
        axes[1, 1].set_ylabel('结构权重', fontsize=11)
        axes[1, 1].set_title('权重-相似度相关性分析', fontsize=12, fontweight='bold')

        corr = np.corrcoef(labels, struct_weights)[0, 1]
        axes[1, 1].text(0.05, 0.95, f'相关系数 ρ={corr:.3f}', transform=axes[1, 1].transAxes,
                        fontsize=11, verticalalignment='top', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1, 1].grid(alpha=0.3)

        # 6. Recall@K曲线
        k_values = [5, 10, 20, 50, 100]
        recalls = [metrics.get(f'Recall@{k}', 0) for k in k_values]
        precisions = [metrics.get(f'Precision@{k}', 0) for k in k_values]

        axes[1, 2].plot(k_values, recalls, marker='o', color='red', linewidth=2,
                        label='Recall', markersize=8)
        axes[1, 2].plot(k_values, precisions, marker='s', color='blue', linewidth=2,
                        label='Precision', markersize=8)
        axes[1, 2].axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='目标30%')
        axes[1, 2].set_xlabel('K值', fontsize=11)
        axes[1, 2].set_ylabel('分数', fontsize=11)
        axes[1, 2].set_title('Recall/Precision@K曲线', fontsize=12, fontweight='bold')
        axes[1, 2].legend()
        axes[1, 2].grid(alpha=0.3)

        # 7. 评估指标雷达图
        ax7 = axes[2, 0]
        metric_names = ['Accuracy', 'F1-Score', 'ROC-AUC', 'nDCG@10', 'mAP@10']
        metric_values = [metrics.get(name, 0) for name in
                         ['accuracy', 'f1_score', 'roc_auc', 'nDCG@10', 'mAP@10']]

        # 归一化到0-1
        metric_values_norm = np.clip(metric_values, 0, 1)

        # 雷达图（修复：使用np.append进行拼接）
        angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()

        # 正确闭合数组：np.append用于numpy数组，+=用于Python列表
        metric_values_norm = np.append(metric_values_norm, metric_values_norm[0])  # 修复：从 += 改为 np.append
        angles += angles[:1]  # 列表拼接是正确的

        ax7.plot(angles, metric_values_norm, 'o-', linewidth=2, label='模型表现')
        ax7.fill(angles, metric_values_norm, alpha=0.25)
        ax7.set_xticks(angles[:-1])
        ax7.set_xticklabels(metric_names, fontsize=10)
        ax7.set_ylim(0, 1)
        ax7.set_title('关键指标雷达图', fontsize=12, fontweight='bold')
        ax7.grid(True)
        ax7.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

        # 8. 预测误差分布
        errors = np.abs(preds - labels)
        axes[2, 1].hist(errors, bins=30, color='darkorange', alpha=0.7, edgecolor='black')
        axes[2, 1].axvline(x=errors.mean(), color='red', linestyle='--',
                           label=f'平均误差={errors.mean():.3f}')
        axes[2, 1].set_xlabel('预测误差', fontsize=11)
        axes[2, 1].set_ylabel('频数', fontsize=11)
        axes[2, 1].set_title('预测误差分布', fontsize=12, fontweight='bold')
        axes[2, 1].legend()
        axes[2, 1].grid(alpha=0.3)

        # 9. 综合指标汇总柱状图
        ax9 = axes[2, 2]
        display_metrics = {
            'Accuracy': metrics['accuracy'],
            'F1-Score': metrics['f1_score'],
            'Recall@10': metrics['Recall@10'],
            'ROC-AUC': metrics['roc_auc']
        }

        names = list(display_metrics.keys())
        values = list(display_metrics.values())
        colors = ['skyblue', 'lightgreen', 'salmon', 'gold']

        bars = ax9.bar(names, values, color=colors, edgecolor='black', alpha=0.8)
        ax9.set_title('核心指标汇总', fontsize=12, fontweight='bold')
        ax9.set_ylabel('分数', fontsize=11)
        ax9.tick_params(axis='x', rotation=45)

        # 在柱子上显示数值
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax9.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("evaluation_results_enhanced.png", dpi=300, bbox_inches='tight')
        print("\n✅ 可视化结果已保存至 evaluation_results_enhanced.png")

        # 诊断报告
        print("\n" + "=" * 60)
        print("📋 模型诊断报告:")
        print("=" * 60)
        print(
            f"1. 权重动态性: σ={metrics.get('Weight_Std', 0):.3f} {'✅正常' if metrics.get('Weight_Std', 0) > 0.1 else '⚠️过低'}")
        print(
            f"2. Recall@10: {metrics.get('Recall@10', 0):.3f} {'✅优秀' if metrics.get('Recall@10', 0) > 0.3 else '⚠️过低'}")
        print(
            f"3. 权重-相似度相关性: {metrics.get('Sim-Weight_Correlation', 0):.3f} {'✅有效' if metrics.get('Sim-Weight_Correlation', 0) > 0.2 else '⚠️微弱'}")
        print(
            f"4. ROC-AUC: {metrics.get('roc_auc', 0):.3f} {'✅优秀' if metrics.get('roc_auc', 0) > 0.8 else '⚠️需优化'}")
        print("=" * 60)

        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/best_model.pth",
                        help="模型路径")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("开始评估增强模型...")
    print("=" * 60)

    evaluator = TableSimilarityEvaluator(model_path=args.model_path)
    evaluator.visualize_results()


if __name__ == "__main__":
    main()
