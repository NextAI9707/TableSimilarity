import os
import faiss
import numpy as np
import yaml
import pickle
import torch
import torch.nn.functional as F
from model import EnhancedTableSimilarityModel
from dataset import TableSimilarityDataset
import mysql.connector  # 新增MySQL支持
from typing import Dict, List


class VectorStore:
    """
    向量库存储：基于Faiss的相似表快速检索（MySQL适配版）
    """

    def __init__(self, config_path: str = "config.yml"):
        # 加载配置（关键修复：从config读取所有参数）
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 向量存储路径（从配置读取）
        self.index_path = self.config['vector_store']['path']

        # MySQL配置（关键修复：取代SQLite）
        mysql_cfg = self.config.get('mysql', {})
        self.mysql_host = mysql_cfg.get('host', 'localhost')
        self.mysql_user = mysql_cfg.get('user', 'root')
        self.mysql_password = mysql_cfg.get('password', '')
        self.mysql_port = mysql_cfg.get('port', 3306)
        self.mysql_database = mysql_cfg.get('database', 'table_similarity')

        # 维度从配置读取
        self.dim = self.config['model']['embedding_dims']['fused']

        self.index = faiss.IndexFlatIP(self.dim)
        self.table_mapping: Dict[int, str] = {}
        self.reverse_mapping: Dict[str, int] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型路径从配置读取（关键修复：非硬编码）
        model_cfg = self.config.get('training', {})
        self.model_path = model_cfg.get('save_path', 'models/best_model.pth')

        # 初始化增强模型
        self.model = EnhancedTableSimilarityModel(config_path).to(self.device)

        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ 加载增强模型: epoch={checkpoint.get('epoch', 'unknown')}")
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✅ 成功加载 {sum(p.numel() for p in self.model.parameters()):,} 个参数")
        else:
            print(f"⚠️ 警告：未找到模型文件 {self.model_path}，将使用随机初始化")

        self.model.eval()

    def _get_db_connection(self):
        """获取MySQL连接（新增方法）"""
        return mysql.connector.connect(
            host=self.mysql_host,
            user=self.mysql_user,
            password=self.mysql_password,
            port=self.mysql_port,
            database=self.mysql_database,
            charset='utf8mb4'
        )

    def build_vector_store(self):
        """构建增强模型的向量库（MySQL适配版）"""
        # 关键修复：使用MySQL查询表
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        conn.close()

        if not tables:
            raise ValueError(f"MySQL数据库 {self.mysql_database} 中没有表！")

        # 关键修复：Dataset初始化改为配置驱动
        dataset = TableSimilarityDataset(config_path="config.yml", mode="train")

        all_vectors = []
        table_names = []
        failed_tables = []

        print("\n🔧 开始编码表向量（增强模型）...")
        self.model.eval()
        with torch.no_grad():
            for idx, table_info in enumerate(tables):
                table_name = table_info[0]
                try:
                    # 加载单表
                    table = dataset._load_table(table_name)
                    struct = dataset._encode_structure(table)  # [10, 39]
                    content = dataset._encode_content(table)  # [32]

                    # 构建batch
                    struct_batch = struct.unsqueeze(0).to(self.device)  # [1, 10, 39]
                    content_batch = content.unsqueeze(0).to(self.device)  # [1, 32]

                    # 在 build_vector_store 方法中，替换模型调用行
                    fused_vec, _, _, _, _, _ = self.model(
                        struct_a=struct_batch,
                        content_a=content_batch,
                        struct_b=struct_batch,  # 虚拟第二表
                        content_b=content_batch
                    )

                    # 归一化并存储（模型已内部归一化，此处再次确保）
                    vector = F.normalize(fused_vec[0], p=2, dim=0).cpu().numpy()
                    all_vectors.append(vector)
                    table_names.append(table_name)

                    if idx % 50 == 0:
                        print(f"  已编码 {idx + 1}/{len(tables)} 个表")

                except Exception as e:
                    print(f"❌ 表 {table_name} 编码失败: {e}")
                    failed_tables.append(table_name)
                    continue

        if not all_vectors:
            raise RuntimeError(f"没有任何表成功编码！失败表: {failed_tables}")

        # 构建Faiss索引
        vectors = np.array(all_vectors).astype(np.float32)
        self.index.add(vectors)

        # 构建映射关系
        for idx, name in enumerate(table_names):
            self.table_mapping[idx] = name
            self.reverse_mapping[name] = idx

        # 保存所有文件
        mapping_dir = "data"
        os.makedirs(mapping_dir, exist_ok=True)

        # 保存Faiss索引
        faiss.write_index(self.index, self.index_path)

        # 保存映射
        with open(f"{mapping_dir}/table_mapping.pkl", 'wb') as f:
            pickle.dump(self.table_mapping, f)
        with open(f"{mapping_dir}/reverse_mapping.pkl", 'wb') as f:
            pickle.dump(self.reverse_mapping, f)

        print(f"✅ 向量库构建完成！共{len(table_names)}个表，维度{self.dim}")
        print(f"✅ 文件已保存: {self.index_path}")

    def load_vector_store(self):
        """加载已存在的向量库"""
        if not os.path.exists(self.index_path):
            print(f"⚠️ 向量库文件不存在: {self.index_path}")
            return False

        mapping_files = {
            'table_mapping': f"data/table_mapping.pkl",
            'reverse_mapping': f"data/reverse_mapping.pkl"
        }

        for name, path in mapping_files.items():
            if not os.path.exists(path):
                print(f"⚠️ 映射文件不存在: {path}")
                return False

        try:
            self.index = faiss.read_index(self.index_path)

            with open(mapping_files['table_mapping'], 'rb') as f:
                self.table_mapping = pickle.load(f)
            with open(mapping_files['reverse_mapping'], 'rb') as f:
                self.reverse_mapping = pickle.load(f)

            print(f"✅ 向量库加载成功！共{len(self.table_mapping)}个表")
            return True
        except Exception as e:
            print(f"✗ 加载失败: {e}")
            return False

    def search_similar_tables(self, table_name: str, top_k: int = 10):
        """搜索相似表（增强版：边界检查）"""
        if not self.reverse_mapping:
            print("⚠️ 反向映射未加载，请先构建向量库")
            return []

        if table_name not in self.reverse_mapping:
            print(f"表 '{table_name}' 不在向量库中")
            print(f"可用表示例: {list(self.reverse_mapping.keys())[:10]}...")  # 显示前10个
            return []

        query_id = self.reverse_mapping[table_name]

        # 边界检查
        k = min(top_k + 1, self.index.ntotal)
        if k <= 1:
            print("⚠️ 向量库中只有1个表，无法推荐")
            return []

        query_vector = self.index.reconstruct(query_id)

        distances, indices = self.index.search(
            np.array([query_vector]).astype(np.float32),
            k
        )

        results = []
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            if idx != query_id:  # 排除自己
                results.append({
                    'table_name': self.table_mapping[idx],
                    'similarity': float(distances[0][i]),
                    'rank': len(results) + 1
                })

        return results[:top_k]

    def batch_search_all_pairs(self, top_k: int = 10):
        """批量计算所有表对的相似度（修复版：边界检查）"""
        if self.index.ntotal == 0:
            print("⚠️ 向量库为空")
            return []

        k = min(top_k + 1, self.index.ntotal)
        if k <= 1:
            print("⚠️ 向量库中表数量不足")
            return []

        all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
        distances, indices = self.index.search(all_vectors, k)

        similar_pairs = []
        for i in range(len(indices)):
            table_a = self.table_mapping[i]
            for j in range(1, len(indices[i])):  # 从1开始跳过自己
                idx_b = int(indices[i][j])
                if idx_b < len(self.table_mapping):  # 修复：边界检查
                    table_b = self.table_mapping[idx_b]
                    similarity = float(distances[i][j])
                    similar_pairs.append({
                        'table_a': table_a,
                        'table_b': table_b,
                        'similarity': similarity
                    })

        similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_pairs


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="强制重建向量库")
    parser.add_argument("--test", type=str, help="测试查询表名")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    store = VectorStore()

    # 尝试加载，失败则构建
    if args.rebuild or not store.load_vector_store():
        print("\n" + "=" * 60)
        print("开始构建向量库...")
        print("=" * 60)
        store.build_vector_store()

    # 测试查询
    if args.test and store.reverse_mapping:
        print(f"\n查询与 '{args.test}' 最相似的表（Top-5）：")
        results = store.search_similar_tables(args.test, top_k=5)
        for r in results:
            print(f"  {r['table_name']}: {r['similarity']:.4f}")
    elif store.reverse_mapping:
        # 默认查询第一个表
        all_tables = list(store.reverse_mapping.keys())
        if all_tables:
            print(f"\n查询与 '{all_tables[0]}' 最相似的表（Top-5）：")
            results = store.search_similar_tables(all_tables[0], top_k=5)
            for r in results:
                print(f"  {r['table_name']}: {r['similarity']:.4f}")

    # 批量分析（可选）
    if store.reverse_mapping and len(store.reverse_mapping) > 1:
        print("\n全局相似度分析（前5对）：")
        all_pairs = store.batch_search_all_pairs(top_k=5)
        for pair in all_pairs[:5]:
            print(f"  {pair['table_a']} <-> {pair['table_b']}: {pair['similarity']:.4f}")

if __name__ == "__main__":
    main()
