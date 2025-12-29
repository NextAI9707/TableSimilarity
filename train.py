import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
from model import EnhancedTableSimilarityModel, Config
from dataset import get_dataloader
import argparse
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class EarlyStoppingWithoutStop:
    """早停优化：完全配置化"""

    def __init__(self, config: Config):
        self.patience = config.get('training.early_stopping.patience', 10)
        self.delta = config.get('training.early_stopping.delta', 0.001)
        self.save_path = config.get('training.early_stopping.save_path', 'models/best_model.pth')
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.no_improve_epochs = 0
        self.save_counter = 0
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def step(self, val_loss: float, model, epoch: int) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.no_improve_epochs = 0
            self.save_counter += 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'config': model.config._config
            }, self.save_path)
            print(f"\n🎯 最佳模型保存: Epoch {epoch} | 验证损失: {val_loss:.4f} | 第 {self.save_counter} 次保存")
        else:
            self.no_improve_epochs += 1
        return True

    def get_best_model_info(self) -> dict:
        return {
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "save_counter": self.save_counter
        }


class TemperatureScheduler:
    """动态温度调度：完全配置化"""

    def __init__(self, config: Config):
        self.initial_temp = config.get('training.temperature_scheduler.initial_temp', 0.07)
        self.final_temp = config.get('training.temperature_scheduler.final_temp', 0.04)
        self.decay_epochs = config.get('training.temperature_scheduler.decay_epochs', 7)

    def get_temperature(self, epoch: int) -> float:
        ratio = min(epoch / self.decay_epochs, 1.0) if self.decay_epochs > 0 else 1.0
        return self.initial_temp * (1 - ratio) + self.final_temp * ratio


class CurriculumScheduler:
    """课程学习调度器：完全配置化"""

    def __init__(self, config: Config):
        self.enabled = config.get('training.curriculum_learning.enabled', True)
        self.start_threshold = config.get('training.curriculum_learning.start_threshold', 0.7)
        self.end_threshold = config.get('training.curriculum_learning.end_threshold', 0.0)
        self.transition_epochs = config.get('training.curriculum_learning.transition_epochs', 5)

    def get_threshold(self, epoch: int) -> float:
        if not self.enabled:
            return 0.0
        if epoch < self.transition_epochs:
            return self.start_threshold
        return self.end_threshold


def train_epoch(model, dataloader, optimizer, device, epoch, temp_scheduler=None, config=None):
    """
    训练一个epoch：增强版 - 增加指标计算与返回
    """
    model.train()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    total_samples = 0
    grad_norms = []
    weight_stats = []
    all_preds = []
    all_labels = []

    # 获取日志间隔
    log_interval = config.get('training.logging.log_interval', 20) if config else 20

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} 训练", file=sys.stdout, ncols=150)

    for batch_idx, batch in enumerate(pbar):
        # 数据迁移
        struct_a = batch['struct_a'].to(device)
        struct_b = batch['struct_b'].to(device)
        content_a = batch['content_a'].to(device)
        content_b = batch['content_b'].to(device)
        similarity = batch['similarity'].to(device)

        # 首次batch诊断
        if epoch == 0 and batch_idx == 0:
            print(f"\n{'=' * 60}")
            print(f"【首次Batch诊断】")
            print(f"struct_a 范围: [{struct_a.min():.3f}, {struct_a.max():.3f}]")
            print(f"content_a 范围: [{content_a.min():.3f}, {content_a.max():.3f}]")
            print(f"相似度标签: {similarity[:5].tolist()}")
            print(f"{'=' * 60}\n")

        # 标签二值化
        binary_labels = (similarity > 0.5).float()

        # 前向传播
        optimizer.zero_grad()

        loss, similarities, weights_a, weights_b = model.compute_loss(
            struct_a, content_a, struct_b, content_b, binary_labels
        )

        # Loss异常检测
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ 警告：批次 {batch_idx} 损失异常: {loss.item()}, 跳过")
            num_batches += 1
            continue

        # 反向传播与梯度监控
        loss.backward()

        # 梯度裁剪
        if config and config.get('training.gradient_clip.enabled', True):
            max_norm = config.get('training.gradient_clip.max_norm', 10.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            grad_norms.append(grad_norm.item())

        optimizer.step()

        # 统计与日志
        total_loss += loss.item()
        num_batches += 1

        # 准确率计算
        preds = (similarities > 0.5).float()
        total_correct += (preds == binary_labels).sum().item()
        total_samples += binary_labels.size(0)

        # 收集预测和标签用于指标计算
        all_preds.extend(similarities.detach().cpu().numpy())
        all_labels.extend(similarity.cpu().numpy())

        # 权重统计
        if weights_a is not None:
            weight_stats.append(weights_a.detach().cpu())

        # 进度条更新
        if batch_idx % log_interval == 0:
            if weight_stats:
                all_weights = torch.cat(weight_stats, dim=0)
                struct_weight_mean = all_weights[:, 0].mean().item()
                content_weight_mean = all_weights[:, 1].mean().item()
            else:
                struct_weight_mean = content_weight_mean = 0.0

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{(preds == binary_labels).float().mean():.2%}',
                'StructW': f'{struct_weight_mean:.2f}',
                'ContentW': f'{content_weight_mean:.2f}',
                'Grad': f'{grad_norms[-1]:.3f}' if grad_norms else 'N/A'
            })

    pbar.close()

    # ========== Epoch结束汇总（增强版）==========
    avg_loss = total_loss / max(num_batches, 1)
    avg_acc = total_correct / max(total_samples, 1)

    # 计算F1和AUC
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_f1, avg_auc = 0.0, 0.5

    if len(all_labels) > 0:
        # F1-Score
        avg_f1 = f1_score((all_labels >= 0.5).astype(int),
                          (all_preds >= 0.5).astype(int),
                          average='weighted', zero_division=0)

        # ROC-AUC
        if len(np.unique(all_labels)) > 1:
            try:
                avg_auc = roc_auc_score((all_labels >= 0.5).astype(int), all_preds)
            except:
                avg_auc = 0.5

    # 权重统计
    if weight_stats:
        all_weights = torch.cat(weight_stats, dim=0)
        struct_mean = all_weights[:, 0].mean().item()
        struct_std = all_weights[:, 0].std().item()
        content_mean = all_weights[:, 1].mean().item()
        content_std = all_weights[:, 1].std().item()
    else:
        struct_mean = struct_std = content_mean = content_std = 0

    print(f"\n{'=' * 60}")
    print(f"【Epoch {epoch + 1} 训练总结】")
    print(f"📊 Loss: {avg_loss:.4f} | Acc: {avg_acc:.2%} | F1: {avg_f1:.4f} | AUC: {avg_auc:.4f}")
    print(f"🎯 结构权重: {struct_mean:.3f}±{struct_std:.3f}")
    print(f"🎯 内容权重: {content_mean:.3f}±{content_std:.3f}")
    print(f"🧠 梯度范数: {np.mean(grad_norms) if grad_norms else 0:.3f}")
    print(f"{'=' * 60}\n")

    return avg_loss, avg_acc, avg_f1, avg_auc


def validate_epoch(model, dataloader, device, config=None):
    """验证模型：增强版 - 返回多指标"""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_preds = []
    all_labels = []

    if len(dataloader) == 0:
        print("⚠️ 警告：验证数据加载器为空！")
        return 0.0, 0.0, 0.0, 0.0

    pbar = tqdm(dataloader, desc="验证", file=sys.stdout, ncols=100)

    with torch.no_grad():
        for batch in pbar:
            struct_a = batch['struct_a'].to(device)
            struct_b = batch['struct_b'].to(device)
            content_a = batch['content_a'].to(device)
            content_b = batch['content_b'].to(device)
            similarity = batch['similarity'].to(device)

            # 修改前向调用，收集预测
            loss, similarities, _, _ = model.compute_loss(
                struct_a, content_a, struct_b, content_b, similarity
            )

            total_loss += loss.item()
            num_batches += 1

            # 收集预测和标签
            all_preds.extend(similarities.cpu().numpy())
            all_labels.extend(similarity.cpu().numpy())

            avg_loss = total_loss / num_batches
            pbar.set_postfix({'Avg Loss': f'{avg_loss:.4f}'})

    pbar.close()

    # 计算验证指标
    avg_loss = total_loss / max(num_batches, 1)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    avg_acc = accuracy_score((all_labels >= 0.5).astype(int),
                             (all_preds >= 0.5).astype(int))
    avg_f1 = f1_score((all_labels >= 0.5).astype(int),
                      (all_preds >= 0.5).astype(int),
                      average='weighted', zero_division=0)
    avg_auc = 0.5
    if len(np.unique(all_labels)) > 1:
        try:
            avg_auc = roc_auc_score((all_labels >= 0.5).astype(int), all_preds)
        except:
            pass

    return avg_loss, avg_acc, avg_f1, avg_auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml", help="配置文件路径")
    parser.add_argument("--epochs", type=int, help="覆盖配置中的训练轮数")
    parser.add_argument("--batch_size", type=int, help="覆盖配置中的批次大小")
    parser.add_argument("--lr", type=float, help="覆盖配置中的学习率")
    args = parser.parse_args()

    # 加载配置
    config = Config(args.config)

    # 命令行参数覆盖
    if args.epochs:
        config._config['training']['epochs'] = args.epochs
    if args.batch_size:
        config._config['training']['batch_size'] = args.batch_size
    if args.lr:
        config._config['training']['learning_rate'] = args.lr

    # 设备
    device_cfg = config.get_dict('device')
    device = torch.device(
        f"cuda:{device_cfg.get('cuda_device', 0)}" if device_cfg.get(
            'auto_select') and torch.cuda.is_available() else "cpu"
    )
    print(f"使用设备: {device}")

    # 创建模型
    model = EnhancedTableSimilarityModel(args.config).to(device)

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('training.learning_rate'),
        weight_decay=config.get('training.weight_decay')
    )

    # 调度器
    scheduler_type = config.get('training.lr_scheduler.type', 'CosineAnnealingLR')
    if scheduler_type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.get('training.lr_scheduler.T_max')
        )
    elif scheduler_type == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=config.get('training.lr_scheduler.gamma', 0.5)
        )

    # 温度调度
    temp_scheduler = TemperatureScheduler(config) if config.get('training.temperature_scheduler.enabled') else None

    # 课程调度
    curriculum_scheduler = CurriculumScheduler(config)

    # 数据加载器
    batch_size = config.get('training.batch_size')
    train_loader = get_dataloader(config_path=args.config, mode="train", batch_size=batch_size)
    val_loader = get_dataloader(config_path=args.config, mode="val", batch_size=batch_size)

    # 早停
    early_stopping = EarlyStoppingWithoutStop(config)

    # TensorBoard
    if config.get('training.logging.tensorboard_dir'):
        writer = SummaryWriter(config.get('training.logging.tensorboard_dir'))
    else:
        writer = None

    print("\n" + "=" * 60)
    print("开始训练增强模型...")
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"训练样本: {len(train_loader.dataset)} | 验证样本: {len(val_loader.dataset)}")
    print(f"训练批次: {len(train_loader)} | 批次大小: {batch_size}")
    print(f"训练轮数: {config.get('training.epochs')}")
    print("=" * 60 + "\n")

    # 训练历史记录
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_auc': [], 'val_auc': [],
        'train_f1': [], 'val_f1': []
    }

    for epoch in range(config.get('training.epochs')):
        # 课程学习
        threshold = curriculum_scheduler.get_threshold(epoch)
        if hasattr(train_loader.dataset, 'filter_similarity_threshold'):
            train_loader.dataset.filter_similarity_threshold = threshold

        # 温度调度
        if temp_scheduler:
            current_temp = temp_scheduler.get_temperature(epoch)
            model.contrastive_enhancer.temperature = current_temp
        else:
            current_temp = config.get('model.contrastive_loss.temperature')

        print(f"\nEpoch {epoch + 1}/{config.get('training.epochs')} | 阈值: {threshold:.2f} | 温度: {current_temp:.4f}")
        print("-" * 60)

        # 训练与验证（接收多返回值）
        train_loss, train_acc, train_f1, train_auc = train_epoch(
            model, train_loader, optimizer, device, epoch, temp_scheduler, config
        )
        val_loss, val_acc, val_f1, val_auc = validate_epoch(
            model, val_loader, device, config
        )

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        # 调度器
        scheduler.step()

        # TensorBoard记录（增强）
        if writer:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Val', val_acc, epoch)
            writer.add_scalar('F1/Train', train_f1, epoch)
            writer.add_scalar('F1/Val', val_f1, epoch)
            writer.add_scalar('AUC/Train', train_auc, epoch)
            writer.add_scalar('AUC/Val', val_auc, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Temperature', current_temp, epoch)

        # 早停
        if config.get('training.early_stopping.enabled'):
            early_stopping.step(val_loss, model, epoch)

        print(f"\nEpoch {epoch + 1} 总结:")
        print(f"  训练: Loss={train_loss:.4f}, Acc={train_acc:.2%}, F1={train_f1:.4f}, AUC={train_auc:.4f}")
        print(f"  验证: Loss={val_loss:.4f}, Acc={val_acc:.2%}, F1={val_f1:.4f}, AUC={val_auc:.4f}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  最佳轮次: {early_stopping.best_epoch} (损失: {early_stopping.best_loss:.4f})")
        print("-" * 60)

    # 训练完成后的可视化
    if writer:
        writer.close()

    # 绘制训练曲线
    def plot_training_curves(history, save_path='training_curves.png'):
        """绘制训练曲线图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        epochs = range(1, len(history['train_loss']) + 1)

        # Loss
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='训练Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='验证Loss', linewidth=2)
        axes[0, 0].set_title('Loss变化曲线', fontsize=13, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch');
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend();
        axes[0, 0].grid(alpha=0.3)

        # Accuracy
        axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='训练准确率', linewidth=2)
        axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='验证准确率', linewidth=2)
        axes[0, 1].set_title('准确率变化曲线', fontsize=13, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch');
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_ylim(0, 1);
        axes[0, 1].legend();
        axes[0, 1].grid(alpha=0.3)

        # ROC-AUC
        axes[1, 0].plot(epochs, history['train_auc'], 'b-', label='训练AUC', linewidth=2)
        axes[1, 0].plot(epochs, history['val_auc'], 'r-', label='验证AUC', linewidth=2)
        axes[1, 0].set_title('ROC-AUC变化曲线', fontsize=13, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch');
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].set_ylim(0.5, 1);
        axes[1, 0].legend();
        axes[1, 0].grid(alpha=0.3)

        # F1-Score
        axes[1, 1].plot(epochs, history['train_f1'], 'b-', label='训练F1', linewidth=2)
        axes[1, 1].plot(epochs, history['val_f1'], 'r-', label='验证F1', linewidth=2)
        axes[1, 1].set_title('F1-Score变化曲线', fontsize=13, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch');
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].set_ylim(0, 1);
        axes[1, 1].legend();
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ 训练曲线已保存至 {save_path}")

    print("\n" + "=" * 60)
    print("🎉 训练完成！生成可视化报告...")
    print("=" * 60)

    # 执行绘图
    plot_training_curves(history)

    best_info = early_stopping.get_best_model_info()
    print(f"\n最佳模型在第 {best_info['best_epoch']} 轮，验证损失: {best_info['best_loss']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    main()
