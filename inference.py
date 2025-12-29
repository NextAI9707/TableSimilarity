from typing import List, Tuple, Dict
import torch
import numpy as np
import mysql.connector  # 新增MySQL支持
from model import EnhancedTableSimilarityModel
from dataset import TableSimilarityDataset
import yaml
import os
import argparse

class TableSimilarityInference:
    """
    表相似性推理引擎（MySQL适配版）
    """

    def __init__(self, model_path: str = None, config_path: str = "config.yml"):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self._verify_database()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型路径从配置读取
        if model_path is None:
            model_path = self.config['training'].get('save_path', 'models/best_model.pth')

        # 初始化增强模型（配置驱动）
        self.model = EnhancedTableSimilarityModel(config_path).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        # Dataset初始化改为配置驱动
        self.dataset = TableSimilarityDataset(config_path=config_path, mode="train")

    def _verify_database(self):
        """验证MySQL数据库可连接且有表"""
        mysql_cfg = self.config.get('mysql', {})
        try:
            conn = mysql.connector.connect(
                host=mysql_cfg.get('host', 'localhost'),
                user=mysql_cfg.get('user', 'root'),
                password=mysql_cfg.get('password', ''),
                port=mysql_cfg.get('port', 3306),
                database=mysql_cfg.get('database', 'table_similarity'),
                charset='utf8mb4'
            )
            cursor = conn.cursor()
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            conn.close()

            if not tables:
                raise ValueError(f"MySQL数据库中没有表！")
        except mysql.connector.Error as e:
            raise ConnectionError(f"无法连接MySQL数据库: {e}")

    def get_all_table_names(self) -> list:
        """从MySQL获取所有表名"""
        mysql_cfg = self.config.get('mysql', {})
        conn = mysql.connector.connect(
            host=mysql_cfg.get('host', 'localhost'),
            user=mysql_cfg.get('user', 'root'),
            password=mysql_cfg.get('password', ''),
            port=mysql_cfg.get('port', 3306),
            database=mysql_cfg.get('database', 'table_similarity'),
            charset='utf8mb4'
        )
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables

    def compute_table_similarity(self, table_a_name: str, table_b_name: str) -> float:
        """
        增强模型相似度计算：适配新版模型接口
        """
        # 验证表存在
        all_tables = self.get_all_table_names()
        if table_a_name not in all_tables or table_b_name not in all_tables:
            raise ValueError(f"表不存在。可用表: {all_tables[:10]}...")

        # 加载两张表
        table_a = self.dataset._load_table(table_a_name)
        table_b = self.dataset._load_table(table_b_name)

        # 编码特征
        struct_a = self.dataset._encode_structure(table_a, reference_table=table_b)
        content_a = self.dataset._encode_content(table_a)
        struct_b = self.dataset._encode_structure(table_b, reference_table=table_a)
        content_b = self.dataset._encode_content(table_b)

        # 构建batch
        struct_a_batch = struct_a.unsqueeze(0).to(self.device)
        content_a_batch = content_a.unsqueeze(0).to(self.device)
        struct_b_batch = struct_b.unsqueeze(0).to(self.device)
        content_b_batch = content_b.unsqueeze(0).to(self.device)

        # 新版模型前向（6个返回值）
        with torch.no_grad():
            fused_a, fused_b, weights_a, weights_b, _, _ = self.model(
                struct_a_batch, content_a_batch, struct_b_batch, content_b_batch
            )

            # 计算相似度（使用torch.cosine_similarity）
            similarity = torch.cosine_similarity(fused_a, fused_b, dim=1).item()

            # 提取可解释性信息
            struct_weight = weights_a[0, 0].item()
            content_weight = weights_b[0, 1].item()

            print(f"  结构权重: {struct_weight:.3f} | 内容权重: {content_weight:.3f}")

            return similarity

    def recommend_similar_tables(self, table_name: str, top_k: int = 5) -> List[Dict]:
        """推荐相似表（增强版）"""
        index_path = self.config['vector_store']['path']

        if not os.path.exists(index_path):
            print("⚠️ 向量库缺失，正在自动构建...")
            try:
                from vector_store import VectorStore
                store = VectorStore(config_path="config.yml")  # 传递config_path
                store.build_vector_store()
            except Exception as e:
                print(f"✗ 自动构建失败: {e}")
                return []

        try:
            from vector_store import VectorStore
            store = VectorStore(config_path="config.yml")
            if not store.load_vector_store():
                return []

            results = store.search_similar_tables(table_name, top_k)
            return results
        except Exception as e:
            print(f"✗ 推荐功能出错: {e}")
            return []

    def batch_compare(self, table_name: str, candidate_tables: List[str]) -> List[Tuple[str, float]]:
        """
        批量比较：返回表名+相似度排序
        """
        results = []
        for candidate in candidate_tables:
            if candidate == table_name:
                continue
            try:
                sim = self.compute_table_similarity(table_name, candidate)
                results.append((candidate, sim))
            except Exception as e:
                print(f"比较 {table_name} vs {candidate} 失败: {e}")

        results.sort(key=lambda x: x[1], reverse=True)
        return results


# 全局存储测试用例用于调试
_hard_case_buffer = []


def demo_inference():
    """演示推理功能（增强版：自动检测问题）"""
    try:
        infer = TableSimilarityInference()
        all_tables = infer.get_all_table_names()
        print(f"\n📊 数据库中包含的表: {len(all_tables)}个")

        if len(all_tables) < 2:
            print("错误：数据库中至少需要2个表才能演示")
            return

        # 演示1：计算指定表对
        test_tables = all_tables[:3]
        print("\n" + "=" * 60)
        print("演示1：计算表对相似度")
        print("=" * 60)

        # 存储硬案例
        global _hard_case_buffer
        _hard_case_buffer.clear()

        for i in range(len(test_tables)):
            for j in range(i + 1, len(test_tables)):
                try:
                    print(f"\n计算 '{test_tables[i]}' 和 '{test_tables[j]}' 的相似度...")
                    sim = infer.compute_table_similarity(test_tables[i], test_tables[j])
                    print(f"✅ 相似度: {sim:.4f}")

                    # 存储用于后续分析
                    _hard_case_buffer.append({
                        'table_a': test_tables[i],
                        'table_b': test_tables[j],
                        'similarity': sim
                    })

                except Exception as e:
                    print(f"✗ 计算失败: {e}")
                    import traceback
                    traceback.print_exc()

        # 演示2：相似表推荐
        print("\n" + "=" * 60)
        print("演示2：相似表推荐")
        print("=" * 60)

        # 修复：确保向量库已构建
        index_path = infer.config['vector_store']['path']
        if not os.path.exists(index_path):
            print("⚠️ 向量库缺失，正在自动构建...")
            try:
                from vector_store import VectorStore
                store = VectorStore(config_path="config.yml")
                store.build_vector_store()
            except Exception as e:
                print(f"✗ 自动构建失败: {e}")

        recommendations = infer.recommend_similar_tables(test_tables[0], top_k=3)
        if recommendations:
            print(f"\n为表 '{test_tables[0]}' 推荐相似表（Top-3）：")
            for rec in recommendations:
                print(f"  相似度 {rec['similarity']:.4f}: {rec['table_name']}")
        else:
            print("推荐功能不可用，请检查向量库")

        # 演示3：批量比较
        print("\n" + "=" * 60)
        print("演示3：批量比较")
        print("=" * 60)

        candidates = all_tables[1:6]
        batch_results = infer.batch_compare(test_tables[0], candidates)
        print(f"\n'{test_tables[0]}' 与其他表的相似度:")
        for table, sim in batch_results:
            print(f"  {table}: {sim:.4f}")

        # 硬案例诊断（增强）
        if _hard_case_buffer:
            print("\n" + "=" * 60)
            print("硬案例诊断:")
            print("=" * 60)
            similarities = [case['similarity'] for case in _hard_case_buffer]
            if len(similarities) > 1:
                sim_std = np.std(similarities)
                sim_range = np.max(similarities) - np.min(similarities)
                print(f"相似度标准差: {sim_std:.4f} {'✅正常' if sim_std > 0.05 else '⚠️过低'}")
                print(f"相似度范围: {sim_range:.4f} {'✅正常' if sim_range > 0.1 else '⚠️过小'}")
                print(f"平均相似度: {np.mean(similarities):.4f}")

        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n请确保已按顺序执行：")
        print("1. python generate_dataset.py")
        print("2. python build_knowledge_graph.py")
        print("3. python train.py --epochs 20")
        print("4. python vector_store.py --rebuild")


def main():
    """主入口（命令行增强）"""
    parser = argparse.ArgumentParser(description="表相似性推理引擎")
    parser.add_argument("--table_a", type=str, help="第一张表名")
    parser.add_argument("--table_b", type=str, help="第二张表名")
    parser.add_argument("--recommend", type=str, help="推荐相似表（输入表名）")
    parser.add_argument("--top_k", type=int, default=5, help="推荐数量")
    args = parser.parse_args()

    infer = TableSimilarityInference()

    # 模式1：计算指定表对
    if args.table_a and args.table_b:
        try:
            similarity = infer.compute_table_similarity(args.table_a, args.table_b)
            print(f"\n'{args.table_a}' <-> '{args.table_b}' 的相似度: {similarity:.4f}")
        except Exception as e:
            print(f"❌ 计算失败: {e}")

    # 模式2：推荐相似表
    elif args.recommend:
        try:
            results = infer.recommend_similar_tables(args.recommend, args.top_k)
            if results:
                print(f"\n与 '{args.recommend}' 最相似的表（Top-{args.top_k}）：")
                for rec in results:
                    print(f"  {rec['table_name']}: {rec['similarity']:.4f}")
            else:
                print("未找到相似表")
        except Exception as e:
            print(f"❌ 推荐失败: {e}")

    # 模式3：演示模式
    else:
        demo_inference()


if __name__ == "__main__":
    main()
