import networkx as nx
import yaml
import os
import pickle
import json
from typing import Dict, List


def build_knowledge_graph(config_path: str = "config.yml"):
    """
    构建领域知识图谱：包含实体和关系（增强版）
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    G = nx.DiGraph()

    # 1. 定义核心实体和关系（模式层）
    business_domains = {
        "金融", "交易", "客户", "产品", "风控", "营销", "报表", "元数据"
    }

    data_themes = {
        "账户", "流水", "汇率", "订单", "用户", "商品", "支付", "结算",
        "利率", "额度", "信用", "地址", "日志", "配置"
    }

    # 扩展字段知识库（支持中英文映射）
    field_knowledge = {
        "currency_code": {
            "chinese_name": "货币代码",
            "description": "货币代码，如CNY代表人民币，USD代表美元",
            "related_concepts": "货币,汇率,国际结算,ISO4217",
            "business_rules": "ISO 4217标准,3位大写字母",
            "importance_score": 0.95
        },
        "exchange_rate": {
            "chinese_name": "汇率",
            "description": "货币兑换比率，用于不同货币间的价值转换",
            "related_concepts": "汇率,外汇,兑换,中间价,牌价",
            "business_rules": "通常保留4-6位小数,随市场波动",
            "importance_score": 0.90
        },
        "date": {
            "chinese_name": "日期",
            "description": "业务发生日期，用于时间序列分析",
            "related_concepts": "时间,周期,时效性,交易日",
            "business_rules": "格式YYYY-MM-DD,支持时间戳",
            "importance_score": 0.85
        },
        "code": {
            "chinese_name": "编码",
            "description": "编码类字段，用于唯一标识或分类",
            "related_concepts": "编码,标识,分类,主键",
            "business_rules": "具有业务含义,可能外键关联",
            "importance_score": 0.80
        },
        "type": {
            "chinese_name": "类型",
            "description": "类型字段，用于区分业务场景或分类",
            "related_concepts": "分类,场景,业务类型,枚举",
            "business_rules": "通常用整型或短字符串,有字典表对应",
            "importance_score": 0.75
        },
        # 中文字段名映射
        "货币代码": {
            "english_name": "currency_code",
            "description": "货币代码，如CNY代表人民币，USD代表美元",
            "related_concepts": "货币,汇率,国际结算",
            "business_rules": "ISO 4217标准,3位大写字母",
            "importance_score": 0.95
        },
        "汇率": {
            "english_name": "exchange_rate",
            "description": "货币兑换比率",
            "related_concepts": "汇率,外汇,兑换",
            "business_rules": "保留4-6位小数",
            "importance_score": 0.90
        }
    }

    # 2. 构建图谱（数据层）
    # 添加业务域节点
    for domain in business_domains:
        G.add_node(f"domain_{domain}", type="业务域", name=domain, level=1)

    # 添加数据主题节点
    for theme in data_themes:
        G.add_node(f"theme_{theme}", type="数据主题", name=theme, level=2)
        domain = "金融" if theme in ["账户", "流水", "汇率", "利率", "额度", "信用"] else \
            "交易" if theme in ["订单", "支付", "结算"] else \
                "客户" if theme in ["用户", "地址"] else "元数据"
        G.add_edge(f"domain_{domain}", f"theme_{theme}", relation="包含", weight=0.8)

    # 添加字段知识节点（增强版：包含重要性评分）
    for field_name, knowledge in field_knowledge.items():
        node_id = f"field_{field_name}"

        # 确保所有值都是字符串（GraphML兼容性）
        node_attrs = {
            'type': "字段",
            'name': field_name,
            'description': str(knowledge.get('description', '')),
            'related_concepts': str(knowledge.get('related_concepts', '')),
            'business_rules': str(knowledge.get('business_rules', '')),
            'importance_score': float(knowledge.get('importance_score', 0.5)),
            'chinese_name': str(knowledge.get('chinese_name', field_name))
        }

        G.add_node(node_id, **node_attrs)

        # 建立字段与主题的关联（基于关键词）
        field_lower = field_name.lower()
        if 'currency' in field_lower or 'code' in field_lower:
            G.add_edge(node_id, "theme_汇率", relation="属于", weight=0.9)
        elif 'rate' in field_lower or 'exchange' in field_lower:
            G.add_edge(node_id, "theme_汇率", relation="属于", weight=0.85)
        elif 'date' in field_lower or 'time' in field_lower:
            G.add_edge(node_id, "theme_流水", relation="属于", weight=0.8)
        elif 'code' in field_lower or 'id' in field_lower:
            G.add_edge(node_id, "theme_账户", relation="属于", weight=0.75)
        elif 'type' in field_lower or 'status' in field_lower:
            G.add_edge(node_id, "theme_配置", relation="属于", weight=0.7)

    # 3. 保存图谱前的验证
    print(f"即将保存图谱，节点数: {len(G.nodes())}, 边数: {len(G.edges())}")

    # 检查是否有非法数据类型
    for node_id, node_data in G.nodes(data=True):
        for key, value in node_data.items():
            if isinstance(value, (list, dict, set)):
                raise TypeError(f"节点 {node_id} 的属性 {key} 包含非法类型 {type(value)}: {value}")

    # 4. 保存图谱
    os.makedirs(os.path.dirname(config['knowledge_graph']['path']), exist_ok=True)
    nx.write_graphml(G, config['knowledge_graph']['path'])
    print(f"✅ 知识图谱已保存至: {config['knowledge_graph']['path']}")

    # 5. 额外保存为pickle（更快加载）
    pickle_path = config['knowledge_graph']['path'].replace('.graphml', '.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"✅ 知识图谱(pickle)已保存至: {pickle_path}")

    return G


def query_field_knowledge(G, field_name: str, language: str = "en") -> Dict:
    """
    查询字段知识（增强版：支持中英文）
    language: "en" 或 "zh"
    """
    if language == "en":
        node_id = f"field_{field_name}"
    else:
        # 查找中文名
        node_id = None
        for node in G.nodes():
            if G.nodes[node].get('type') == '字段' and \
                    G.nodes[node].get('chinese_name') == field_name:
                node_id = node
                break
        if node_id is None:
            return {"error": "字段未找到"}

    if node_id in G:
        node_data = G.nodes[node_id]
        return {
            'description': node_data.get('description', ''),
            'related_concepts': node_data.get('related_concepts', '').split(','),
            'business_rules': node_data.get('business_rules', '').split(','),
            'importance_score': node_data.get('importance_score', 0.5),
            'chinese_name': node_data.get('chinese_name', field_name)
        }

    return {"error": "字段未找到"}


def expand_field_semantics(G, field_name: str, top_k: int = 5) -> List[str]:
    """
    扩展字段语义：返回最相关的概念
    """
    node_id = f"field_{field_name}"
    if node_id not in G:
        return []

    # 获取邻居节点
    neighbors = list(G.neighbors(node_id))
    concepts = []

    for neighbor in neighbors:
        edge_data = G.get_edge_data(node_id, neighbor)
        if edge_data and 'weight' in edge_data:
            weight = edge_data['weight']
            neighbor_type = G.nodes[neighbor].get('type', '')
            neighbor_name = G.nodes[neighbor].get('name', '')
            concepts.append((neighbor_name, weight, neighbor_type))

    # 按权重排序
    concepts.sort(key=lambda x: x[1], reverse=True)
    return [f"{name}({type_})" for name, _, type_ in concepts[:top_k]]

def main():
    """构建并测试知识图谱"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="查询字段名")
    parser.add_argument("--lang", type=str, choices=["en", "zh"], default="en",
                        help="查询语言")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("开始构建知识图谱...")
    print("=" * 60)

    G = build_knowledge_graph()

    if args.query:
        print(f"\n查询字段 '{args.query}' 的知识:")
        result = query_field_knowledge(G, args.query, language=args.lang)
        print(json.dumps(result, indent=2, ensure_ascii=False))

        print(f"\n扩展语义:")
        concepts = expand_field_semantics(G, args.query)
        print(concepts)
if __name__ == "__main__":
    main()


数据处理dataset.py：
import mysql.connector
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import networkx as nx
import re
from typing import Dict, List, Tuple, Optional
import json
import os


class TableSimilarityDataset(Dataset):
    """
    物理表相似性数据集：完全配置化
    所有参数从 config.yml 读取，无硬编码
    支持通过知识图谱丰富字段注释
    """

    # 默认配置文件路径
    DEFAULT_CONFIG_PATH = "D:\PyCharmProject\TableSimilarityV1\config.yml"

    def __init__(self, config_path: str = None, mode: str = "train"):
        if config_path is None:
            config_path = self.DEFAULT_CONFIG_PATH

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"数据集配置未找到: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        data_cfg = self.config.get('data', {})
        db_cfg = self.config.get('database', {})

        # 加载MySQL配置（关键修改）
        mysql_cfg = self.config.get('mysql', {})
        self.mysql_host = mysql_cfg.get('host', 'localhost')
        self.mysql_user = mysql_cfg.get('user', 'root')
        self.mysql_password = mysql_cfg.get('password', '')
        self.mysql_port = mysql_cfg.get('port', 3306)
        self.mysql_database = mysql_cfg.get('database', 'table_similarity')

        # 创建SQLAlchemy引擎供pandas使用
        self.engine = create_engine(
            f"mysql+mysqlconnector://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

        self.kg_path = data_cfg.get('kg_path', 'data/knowledge_graph.graphml')
        self.sample_size = db_cfg.get('sample_size', 1000)

        # 知识图谱加载（保持不变）
        if os.path.exists(self.kg_path):
            self.kg = nx.read_graphml(self.kg_path)
        else:
            print(f"⚠️ 警告：知识图谱未找到: {self.kg_path}")
            self.kg = nx.Graph()

        # 加载标注文件（保持不变）
        annotation_path = data_cfg.get(f'{mode}_annotations', f'data/{mode}_annotations.json')
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"标注文件未找到: {annotation_path}")

        with open(annotation_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        # 后续配置保持不变
        self.type_mapping = db_cfg.get('type_mapping', {
            'VARCHAR': 1, 'CHAR': 1, 'TEXT': 1,
            'INT': 2, 'INTEGER': 2, 'BIGINT': 2,
            'DECIMAL': 3, 'NUMERIC': 3, 'FLOAT': 3,
            'DATE': 4, 'TIMESTAMP': 4, 'DATETIME': 4
        })

        tfidf_cfg = self.config.get('tfidf', {})
        self.tfidf = TfidfVectorizer(
            max_features=tfidf_cfg.get('max_features', 50),
            analyzer=tfidf_cfg.get('analyzer', 'char'),
            ngram_range=tuple(tfidf_cfg.get('ngram_range', [2, 3]))
        )
        sample_texts = [f"value_{i}" for i in range(1000)]
        self.tfidf.fit(sample_texts)

        print(f"✅ 数据集初始化完成: {mode}")
        print(f"   配置文件: {config_path}")
        print(f"   MySQL数据库: {self.mysql_database}@{self.mysql_host}:{self.mysql_port}")
        print(f"   标注文件: {annotation_path}")
        print(f"   样本数: {len(self.annotations)}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """获取单个样本"""
        annotation = self.annotations[idx]

        # 加载两个表
        table_a = self._load_table(annotation['table_a'])
        table_b = self._load_table(annotation['table_b'])

        # 编码结构特征（包含字段注释）
        struct_a = self._encode_structure(table_a, reference_table=table_b)
        struct_b = self._encode_structure(table_b, reference_table=table_a)

        # 编码内容特征
        content_a = self._encode_content(table_a)
        content_b = self._encode_content(table_b)

        # 相似度标签
        similarity = float(annotation['similarity'])

        return {
            'struct_a': struct_a,  # 移除F.normalize
            'struct_b': struct_b,
            'content_a': content_a,
            'content_b': content_b,
            'similarity': torch.tensor(similarity, dtype=torch.float32),
            'table_a': annotation['table_a'],
            'table_b': annotation['table_b']
        }

    def _load_table(self, table_name: str) -> Dict:
        """加载表结构和内容（适配MySQL）"""
        # MySQL连接
        conn = mysql.connector.connect(
            host=self.mysql_host,
            user=self.mysql_user,
            password=self.mysql_password,
            port=self.mysql_port,
            database=self.mysql_database,
            charset='utf8mb4'
        )

        # 使用SQLAlchemy引擎读取数据
        df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT {self.sample_size}", self.engine)

        cursor = conn.cursor()

        # MySQL获取列信息（替代PRAGMA）
        cursor.execute(f"SHOW COLUMNS FROM {table_name}")
        columns = cursor.fetchall()
        # MySQL返回格式: (Field, Type, Null, Key, Default, Extra)
        # 原代码使用 (name, type, comment)，其中comment在SQLite中本就为空
        columns_info = [(row[0], row[1], "") for row in columns]

        # MySQL获取表注释（替代sqlite_master）
        cursor.execute(
            "SELECT table_comment FROM information_schema.TABLES "
            "WHERE table_schema = %s AND table_name = %s",
            (self.mysql_database, table_name)
        )
        table_comment_result = cursor.fetchone()
        table_comment = table_comment_result[0] if table_comment_result else ""

        conn.close()

        return {
            'name': table_name,
            'comment': table_comment,
            'columns': columns_info,  # name, type, comment(空)
            'data': df
        }

    def _query_kg_from_graph(self, field_name: str, original_comment: str = "") -> Dict[str, str]:
        """
        从知识图谱查询字段信息并丰富注释

        Args:
            field_name: 字段名
            original_comment: 原始注释（从数据生成时获得）
        """
        # 精确匹配
        node_id = f"field_{field_name}"
        if node_id in self.kg:
            node_data = self.kg.nodes[node_id]
            kg_description = node_data.get('description', '')
            # 合并注释：原始注释 + KG注释
            enriched_description = f"{original_comment}; {kg_description}".strip('; ')
            return {
                'description': enriched_description,
                'related_concepts': node_data.get('related_concepts', ''),
                'business_rules': node_data.get('business_rules', '')
            }

        # 模糊匹配
        for node in self.kg.nodes():
            if (self.kg.nodes[node].get('type') == '字段' and
                    field_name.lower() in self.kg.nodes[node].get('name', '').lower()):
                node_data = self.kg.nodes[node]
                kg_description = node_data.get('description', '')
                enriched_description = f"{original_comment}; {kg_description}".strip('; ')
                return {
                    'description': enriched_description,
                    'related_concepts': node_data.get('related_concepts', ''),
                    'business_rules': node_data.get('business_rules', '')
                }

        # 无知识图谱匹配，返回原始注释
        return {
            'description': original_comment,
            'related_concepts': '',
            'business_rules': ''
        }

    def _kg_field_importance(self, table_name: str) -> Dict[str, float]:
        """从知识图谱获取字段全局重要性"""
        importance = {}
        for node in self.kg.nodes():
            if self.kg.nodes[node].get('type') == '字段':
                field_name = self.kg.nodes[node].get('name', '')
                concepts = self.kg.nodes[node].get('related_concepts', '')
                importance[field_name] = len(concepts.split(',')) * 0.1 + 0.5
        return importance

    def _compute_alignment_score(self, field_name: str, field_type: str,
                                 reference_table: Optional[Dict]) -> float:
        """计算字段与参考表的最佳字段匹配度"""
        if reference_table is None:
            return 1.0

        best_sim = 0.0
        for ref_name, ref_type, _ in reference_table['columns']:
            name_sim = 1.0 if field_name == ref_name else (
                    len(set(field_name) & set(ref_name)) / max(len(field_name), 1)
            )
            type_sim = 1.0 if field_type == ref_type else 0.5
            total_sim = name_sim * 0.6 + type_sim * 0.4
            best_sim = max(best_sim, total_sim)

        return best_sim + 0.1

    def _encode_structure(self, table: Dict, reference_table: Optional[Dict] = None) -> torch.Tensor:
        """统一编码：字段级特征 + 对齐感知 + 重要性加权"""
        field_embeddings = []
        max_fields = 10  # 从配置读取或默认

        global_importance = self._kg_field_importance(table['name'])

        for idx, (field_name, field_type, field_comment) in enumerate(table['columns']):
            if idx >= max_fields:
                break

            # 名称特征
            name_features = [
                len(field_name),
                sum(c.isupper() for c in field_name),
                field_name.count('_'),
                1 if 'id' in field_name.lower() else 0,
            ]

            # 类型编码
            type_code = self.type_mapping.get(field_type.split('(')[0].upper(), 0)

            # 【关键改进】查询知识图谱并丰富注释
            kg_info = self._query_kg_from_graph(field_name, field_comment)

            # 业务重要性评分
            kg_importance = len(kg_info['related_concepts'].split(',')) * 0.1

            # 对齐评分
            alignment_score = self._compute_alignment_score(field_name, field_type, reference_table)

            # 动态权重
            global_score = global_importance.get(field_name, 0.5)
            field_weight = kg_importance * alignment_score * global_score
            field_weight = min(field_weight, 1.0)

            # 【关键改进】使用丰富的注释构建富文本描述
            enriched_text = f"{field_name} TYPE{type_code} {kg_info['description']}"
            if kg_info['related_concepts']:
                enriched_text += f" CONCEPTS:{kg_info['related_concepts']}"
            if kg_info['business_rules']:
                enriched_text += f" RULES:{kg_info['business_rules']}"

            # 哈希编码
            desc_hash = self._text_to_hash(enriched_text, dim=32)

            # 组合特征：4 + 2 + 32 = 38维
            field_vec = name_features + [type_code, field_weight] + desc_hash  # [38]

            # 填充到39维
            if len(field_vec) < 39:
                field_vec += [0.0] * (39 - len(field_vec))
            elif len(field_vec) > 39:
                field_vec = field_vec[:39]

            field_embeddings.append(field_vec)

        # 在 _encode_structure 方法末尾（约第250行）
        struct_tensor = torch.zeros(max_fields, 39, dtype=torch.float32)
        for i, vec in enumerate(field_embeddings[:max_fields]):
            # 强制L2归一化每个字段向量
            vec_tensor = torch.tensor(vec[:39], dtype=torch.float32)
            struct_tensor[i] = F.normalize(vec_tensor, p=2, dim=-1)  # 归一化到单位球

        return struct_tensor  # [10, 39]

    # 替换整个 _text_to_hash 方法（约第215行）
    def _text_to_hash(self, text: str, dim: int) -> List[float]:
        """稳定哈希输出[-1,1]分布"""
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)

        vec = [0.0] * dim
        for i in range(dim):
            # 从哈希中提取位，映射到[-1,1]
            bit = (hash_int >> (i % 128)) & 1
            vec[i] = 1.0 if bit else -1.0
        return vec

    def _is_numeric_type(self, type_str: str) -> bool:
        """判断数值类型"""
        return any(t in type_str.upper() for t in ['INT', 'DECIMAL', 'NUMERIC', 'FLOAT', 'DOUBLE'])

    def _is_date_type(self, type_str: str) -> bool:
        """判断日期类型"""
        return any(t in type_str.upper() for t in ['DATE', 'TIME', 'TIMESTAMP', 'DATETIME'])

    def _encode_numeric_column(self, series: pd.Series) -> torch.Tensor:
        """数值列编码：丰富的统计特征"""
        if series.empty:
            return torch.zeros(32)

        stats = [
            series.mean() if len(series) > 0 else 0,
            series.std() if len(series) > 0 else 0,
            series.min() if len(series) > 0 else 0,
            series.max() if len(series) > 0 else 0,
            series.median() if len(series) > 0 else 0,
            series.quantile(0.25) if len(series) > 0 else 0,
            series.quantile(0.75) if len(series) > 0 else 0,
            series.skew() if len(series) > 0 else 0,
            series.kurtosis() if len(series) > 0 else 0,
            len(series.unique()) / len(series) if len(series) > 0 else 0,
            series.nunique() if len(series) > 0 else 0,
            (series == 0).sum() / len(series) if len(series) > 0 else 0,
            series.astype(str).apply(len).mean() if len(series) > 0 else 0,
        ]

        stats += [0.0] * (32 - len(stats))
        return torch.tensor(stats[:32], dtype=torch.float32)

    def _encode_string_column(self, series: pd.Series) -> torch.Tensor:
        """字符串列编码：采样 + TF-IDF"""
        if series.empty:
            return torch.zeros(32)

        sample = series.sample(min(100, len(series))).astype(str).tolist()

        # TF-IDF特征
        try:
            tfidf_matrix = self.tfidf.transform(sample)
            tfidf_vec = tfidf_matrix.mean(axis=0).A1
        except:
            tfidf_vec = np.zeros(50)

        # 统计特征
        lengths = [len(s) for s in sample]
        length_stats = [
            np.mean(lengths) if lengths else 0,
            np.std(lengths) if lengths else 0,
            len(set(sample)) / len(sample) if sample else 0,
            sum(1 for s in sample if s.isdigit()) / len(sample) if sample else 0,
            sum(1 for s in sample if s.isalpha()) / len(sample) if sample else 0,
        ]

        combined = list(tfidf_vec[:20]) + length_stats
        combined += [0.0] * (32 - len(combined))
        return torch.tensor(combined[:32], dtype=torch.float32)

    def _encode_date_column(self, series: pd.Series) -> torch.Tensor:
        """日期型列编码"""
        if series.empty or series.isnull().all():
            return torch.zeros(32)

        try:
            dates = pd.to_datetime(series, errors='coerce').dropna()
        except:
            return torch.zeros(32)

        if len(dates) == 0:
            return torch.zeros(32)

        # 格式特征
        sample_str = str(series.iloc[0]) if not series.empty else ""
        format_features = [
            len(sample_str),
            sample_str.count('-'),
            sample_str.count('/'),
            1 if ':' in sample_str else 0,
            1 if sample_str.isdigit() else 0,
        ]

        # 时间间隔
        sorted_dates = dates.sort_values()
        intervals = sorted_dates.diff().dt.total_seconds().dropna()

        if len(intervals) > 0:
            interval_features = [
                intervals.mean(),
                intervals.std(),
                len(set(intervals)),
                1 if intervals.nunique() == 1 else 0,
            ]
        else:
            interval_features = [0.0, 0.0, 0.0, 0.0]

        # 范围特征
        range_features = [
            (dates.max() - dates.min()).total_seconds() if len(dates) > 1 else 0,
            dates.min().year,
            dates.max().year,
            len(dates.unique()) / len(dates) if len(dates) > 0 else 0,
        ]

        features = format_features + interval_features + range_features
        features += [0.0] * (32 - len(features))
        return torch.tensor(features[:32], dtype=torch.float32)

    def _encode_content(self, table: Dict) -> torch.Tensor:
        """统一编码表内容：按列类型分别处理"""
        df = table['data']
        if df.empty:
            return torch.zeros(32)

        column_features = []

        for col_name, col_type, col_comment in table['columns']:
            if col_name not in df.columns or df[col_name].isnull().all():
                column_features.append(torch.zeros(32))
                continue

            series = df[col_name].dropna()
            if len(series) == 0:
                column_features.append(torch.zeros(32))
                continue

            # 根据数据类型选择编码策略
            if self._is_numeric_type(col_type):
                feature = self._encode_numeric_column(series)
            elif self._is_date_type(col_type):
                feature = self._encode_date_column(series)
            else:  # 字符串类型
                feature = self._encode_string_column(series)

            column_features.append(feature)

        # 在 _encode_content 方法末尾（约第360行）
        content_vec = torch.stack(column_features).mean(dim=0) if column_features else torch.zeros(32)
        content_vec = F.normalize(content_vec, p=2, dim=-1)  # 归一化
        return content_vec


def get_dataloader(config_path: str = None, mode: str = "train", batch_size: int = None):
    """
    获取数据加载器：完全配置化

    Args:
        config_path: 配置文件路径，默认为 "config.yml"
        mode: 数据集模式 ("train", "val", "test")
        batch_size: 批次大小，从配置读取或指定
    """
    # 设置默认配置路径
    if config_path is None:
        config_path = TableSimilarityDataset.DEFAULT_CONFIG_PATH

    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 创建数据集
    dataset = TableSimilarityDataset(config_path, mode=mode)

    # 获取批次大小
    if batch_size is None:
        batch_size = config.get('training', {}).get('batch_size', 32)

    # 数据过滤（课程学习）
    threshold = config.get('training', {}).get(f'{mode}_threshold', 0.0)
    if hasattr(dataset, 'filter_similarity_threshold'):
        dataset.filter_similarity_threshold = threshold

    # 创建数据加载器
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=0,
        drop_last=False  # 关键：防止小数据集时空加载器
    )
