"""
DBSCAN.py

DBSCAN聚类算法实现
"""

import time
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    rand_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ==========================================================
# 数据集配置
# ==========================================================

Dataset = namedtuple(
    "Dataset",
    [
        "path",
        "clusters",
        "label_col",
        "id_col",
        "header",
        "sep",
        "skip_rows",
        "comment"
    ]
)

DATASETS = {
    # UCI数据集
    "iris": Dataset("dataset/uci/iris.data", 3, -1, None, None, ",", 0, None),
    "bezdekIris": Dataset("dataset/uci/bezdekIris.data", 3, -1, None, None, ",", 0, None),
    "seeds": Dataset("dataset/uci/seeds_dataset.txt", 3, -1, None, None, r"\s+", 0, None),
    "glass": Dataset("dataset/uci/glass.data", 6, -1, 0, None, ",", 0, None),
    "wine": Dataset("dataset/uci/wine.data", 3, 0, None, None, ",", 0, None),
    "wdbc": Dataset("dataset/uci/wdbc.data", 2, 1, 0, None, ",", 0, None),
    "ecoli": Dataset("dataset/uci/ecoli.data", 8, -1, 0, None, r"\s+", 0, None),

    # 人工合成数据集
    "flame": Dataset("dataset/synthetic/flame.txt", 2, -1, None, None, ",", 0, None),
    "jain": Dataset("dataset/synthetic/jain.txt", 2, -1, None, None, ",", 0, None),
    "spiral": Dataset("dataset/synthetic/spiral.txt", 3, -1, None, None, ",", 0, None),
    "panelB": Dataset("dataset/synthetic/panelB.txt", 3, None, None, None, ",", 0, None),
    "panelC": Dataset("dataset/synthetic/panelC.txt", 3, None, None, None, ",", 0, None),
    "r15": Dataset("dataset/synthetic/R15.txt", 15, -1, None, None, ",", 0, None),
    "d31": Dataset("dataset/synthetic/D31.txt", 31, -1, None, None, ",", 0, None),
    "aggregation": Dataset("dataset/synthetic/aggregation.txt", 7, -1, None, None, ",", 0, None),
    "compound": Dataset("dataset/synthetic/compound.txt", 6, -1, None, None, ",", 0, None),
    "pathbased": Dataset("dataset/synthetic/pathbased.txt", 3, -1, None, None, ",", 0, None),
}


# ==========================================================
# 数据加载
# ==========================================================

def load_dataset(dataset_name):
    """
    加载数据集
    """
    config = DATASETS[dataset_name]

    df = pd.read_csv(
        config.path,
        sep=config.sep,
        header=config.header,
        skiprows=config.skip_rows,
        comment=config.comment
    )

    if config.label_col is not None:
        labels_true = df.iloc[:, config.label_col].to_numpy()
    else:
        labels_true = None

    drop_cols = []

    if config.id_col is not None:
        drop_cols.append(df.columns[config.id_col])

    if config.label_col is not None:
        drop_cols.append(df.columns[config.label_col])

    if drop_cols:
        features = df.drop(columns=drop_cols).values
    else:
        features = df.values

    return features.astype(np.float32), labels_true, config.clusters


# ==========================================================
# 常量定义
# ==========================================================

UNCLASSIFIED = 0
NOISE = -1


# ==========================================================
# DBSCAN核心算法
# ==========================================================

def get_distance_matrix(features):
    """
    计算欧氏距离矩阵
    """
    n_samples = features.shape[0]
    dists = np.zeros((n_samples, n_samples), dtype=np.float32)

    for i in range(n_samples):
        for j in range(n_samples):
            vi = features[i]
            vj = features[j]
            dists[i, j] = np.sqrt(np.dot(vi - vj, vi - vj))

    return dists


def find_neighbors(point_id, eps, dists):
    """
    寻找以点 point_id 为中心，eps 为半径的圆内的所有点的索引
    """
    indices = np.where(dists[point_id] <= eps)[0]
    return indices.tolist()


def expand_cluster(dists, labels, cluster_id, seeds, eps, min_samples):
    """
    扩展聚类
    """
    i = 0
    while i < len(seeds):
        current_point = seeds[i]

        # 如果该点被标记为NOISE则重新标记为当前聚类
        if labels[current_point] == NOISE:
            labels[current_point] = cluster_id
        # 如果该点未被标记过
        elif labels[current_point] == UNCLASSIFIED:
            # 标记为当前聚类
            labels[current_point] = cluster_id
            # 计算该点的邻域
            new_seeds = find_neighbors(current_point, eps, dists)

            # 如果邻域足够大，则将其加入到seeds队列中
            if len(new_seeds) >= min_samples:
                seeds = seeds + new_seeds

        i += 1

def density_based_clustering(features, eps, min_samples, scale=True):
    """
    DBSCAN聚类算法
    """
    # 数据标准化（默认开启）
    if scale:
        features = StandardScaler().fit_transform(features)
    start = time.time()

    n_samples = features.shape[0]

    # 计算距离矩阵
    dists = get_distance_matrix(features)

    # 初始化标签
    labels = np.full(n_samples, UNCLASSIFIED, dtype=int)

    cluster_id = 0

    # 遍历所有点
    for point_id in range(n_samples):
        # 如果当前点已经处理过，则跳过
        if labels[point_id] != UNCLASSIFIED:
            continue

        # 找到当前点的邻域
        neighbors = find_neighbors(point_id, eps, dists)

        # 如果邻域点数少于min_samples，标记为NOISE
        if len(neighbors) < min_samples:
            labels[point_id] = NOISE
        else:
            # 否则开始一个新的聚类
            cluster_id += 1
            labels[point_id] = cluster_id
            expand_cluster(dists, labels, cluster_id, neighbors, eps, min_samples)

    elapsed_time = time.time() - start
    print(f"耗时={elapsed_time:.4f} s")

    return labels, cluster_id


# ==========================================================
# 标签对齐
# ==========================================================

def align_labels(labels_true, labels_pred, noise_label=None):
    """
    使用匈牙利算法对齐标签。

    参数
    ----------
    labels_true : ndarray
        真实标签。

    labels_pred : ndarray
        预测标签。

    noise_label : int or None
        噪声类别标签。
        如果不为 None，则禁止该类别参与匈牙利匹配。

    返回
    -------
    labels_pred_aligned : ndarray
        对齐后的预测标签。
    """
    from sklearn.metrics import confusion_matrix
    from scipy.optimize import linear_sum_assignment

    labels = np.union1d(labels_true, labels_pred)

    cm = confusion_matrix(
        labels_true,
        labels_pred,
        labels=labels
    )

    row_ind, col_ind = linear_sum_assignment(-cm)

    mapping = {}

    for r, c in zip(row_ind, col_ind):
        pred_label = labels[c]
        true_label = labels[r]

        # 禁止噪声参与匹配
        if noise_label is not None and pred_label == noise_label:
            continue

        mapping[pred_label] = true_label

    labels_pred_aligned = np.array(
        [
            mapping.get(label, label)
            for label in labels_pred
        ]
    )

    return labels_pred_aligned
# ==========================================================
# 聚类评价指标
# ==========================================================

def clustering_indicators(labels_true, labels_pred):
    """
    计算聚类评价指标。

    对于 DBSCAN：
        1. 噪声点参与评价；
        2. 预测为噪声视为错误；
        3. 噪声类别不参与匈牙利匹配。
    """
    if isinstance(labels_true[0], str):
        labels_true = LabelEncoder().fit_transform(labels_true)

    labels_true = labels_true.copy()
    labels_pred = labels_pred.copy()

    noise_label = None

    #
    # 将噪声映射为新的类别
    #
    if np.any(labels_pred == NOISE):
        noise_label = (
            max(
                np.max(labels_true),
                np.max(labels_pred)
            ) + 1
        )

        labels_pred[
            labels_pred == NOISE
        ] = noise_label

    #
    # 标签对齐
    #
    labels_pred_aligned = align_labels(
        labels_true,
        labels_pred,
        noise_label
    )

    #
    # 计算指标
    #
    f_measure = f1_score(
        labels_true,
        labels_pred_aligned,
        average="macro"
    )

    accuracy = accuracy_score(
        labels_true,
        labels_pred_aligned
    )

    normalized_mutual_information = normalized_mutual_info_score(
        labels_true,
        labels_pred
    )

    rand_index = rand_score(
        labels_true,
        labels_pred
    )

    adjusted_rand_index = adjusted_rand_score(
        labels_true,
        labels_pred
    )

    return (
        f_measure,
        accuracy,
        normalized_mutual_information,
        rand_index,
        adjusted_rand_index
    )
# ==========================================================
# 可视化
# ==========================================================
def draw_cluster(features, labels_pred, eps=None, min_samples=None):
    """
    绘制聚类结果
    """
    features = np.asarray(features)

    if features.shape[1] > 2:
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
    else:
        features_2d = features

    plt.figure(figsize=(8, 6))

    # 分离噪声点和聚类点
    noise_mask = labels_pred == NOISE
    cluster_mask = ~noise_mask

    # 绘制聚类点
    if np.any(cluster_mask):
        # 获取聚类标签（排除噪声）
        cluster_labels = labels_pred[cluster_mask]
        unique_labels = np.unique(cluster_labels)

        for label in unique_labels:
            mask = (labels_pred == label) & cluster_mask
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], s=7, cmap="nipy_spectral", label=f"Cluster {label}")

    # 绘制噪声点（黑色）
    if np.any(noise_mask):
        plt.scatter(features_2d[noise_mask, 0], features_2d[noise_mask, 1], color="black", s=7, alpha=0.6, label="Noise")
    title = "DBSCAN Clustering Result"
    if eps is not None and min_samples is not None:
        title += f" (eps={eps}, min_samples={min_samples})"
    plt.title(title)
    if np.any(noise_mask):
        plt.legend()
    plt.show()

# ==========================================================
# 主程序
# ==========================================================

if __name__ == "__main__":

    # ==========================================================
    # 数据集和参数配置
    # ==========================================================

    # DBSCAN参数配置（针对不同数据集）
    DBSCAN_PARAMS = {
        "bezdekIris": {"eps": 0.14, "min_samples": 8},
        "iris": {"eps": 0.14, "min_samples": 8},
        "seeds": {"eps": 0.17, "min_samples": 8},
        "wine": {"eps": 0.42, "min_samples": 10},
        "wdbc": {"eps": 0.27, "min_samples": 7},
        "glass": {"eps": 0.45, "min_samples": 5},
        "aggregation": {"eps": 0.18, "min_samples": 4},
        "flame": {"eps": 0.28, "min_samples": 4},
        "jain": {"eps": 0.315, "min_samples": 4},
        "spiral": {"eps": 0.315, "min_samples": 4},
        "panelB": {"eps": 0.45, "min_samples": 4},
        "panelC": {"eps": 0.45, "min_samples": 4},
    }

    # 选择数据集
    _dataset_name = "spiral"

    print("加载数据集:   " f"数据集={_dataset_name}")

    _features, _labels_true, _num_clusters_truth = load_dataset(_dataset_name)

    print(f"样本数量={_features.shape[0]}   " f"属性数量(维度)={_features.shape[1]}   " f"真实类别数={_num_clusters_truth}")

    # 获取DBSCAN参数
    _params = DBSCAN_PARAMS.get(_dataset_name, {"eps": 0.3, "min_samples": 4})
    _eps = _params["eps"]
    _min_samples = _params["min_samples"]

    print(f"eps={_eps}   min_samples={_min_samples}")

    # 执行DBSCAN聚类
    _labels_pred, _num_clusters = density_based_clustering(_features, eps=_eps, min_samples=_min_samples, scale=False)

    print(f"聚类数量={_num_clusters}")

    if _labels_true is not None:
        # 有标签：计算聚类指标
        F1, ACC, NMI, RI, ARI = clustering_indicators(_labels_true, _labels_pred)
        print("聚类指标: "  f"F1={F1:.6f}  "  f"ACC={ACC:.6f}  "  f"NMI={NMI:.6f}  "  f"RI={RI:.6f}  "  f"ARI={ARI:.6f}")

    draw_cluster(_features, _labels_pred, eps=_eps, min_samples=_min_samples)