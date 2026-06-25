"""
DPC.py

Density Peaks Clustering（密度峰值聚类）实现
"""

import time
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    rand_score,
)
from sklearn.preprocessing import LabelEncoder


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
    # 鸢尾花数据集有两个版本。虽然差距细微，但是dpc算法对其结果完全不同，所以两个数据集都跑一下。
    # 经典版鸢尾花数据集（35行和38行数据抄些错误）
    "iris": Dataset("dataset/uci/iris.data", 3, -1, None, None, ",", 0, None),
    # 修正版鸢尾花数据集（修正了35行和38行的错误数据）
    "bezdekIris": Dataset("dataset/uci/bezdekIris.data", 3, -1, None, None, ",", 0, None),
    "seeds": Dataset("dataset/uci/seeds_dataset.txt", 3, -1, None, None, r"\s+", 0, None),
    "glass": Dataset("dataset/uci/glass.data", 6, -1, 0, None, ",", 0, None),
    "wine": Dataset("dataset/uci/wine.data", 3, 0, None, None, ",", 0, None),
    "wdbc": Dataset("dataset/uci/wdbc.data", 2, 1, 0, None, ",", 0, None),
    "ecoli": Dataset("dataset/uci/ecoli.data", 8, -1, 0, None, r"\s+", 0, None),

    "flame": Dataset("dataset/synthetic/Flame.txt", 2, -1, None, None, ",", 0, None),
    "jain": Dataset("dataset/synthetic/Jain.txt", 2, -1, None, None, ",", 0, None),
    "spiral": Dataset("dataset/synthetic/Spiral.txt", 3, -1, None, None, ",", 0, None),
    "panelB": Dataset("dataset/synthetic/panelB.txt", 3, None, None, None, ",", 0, None),
    "panelC": Dataset("dataset/synthetic/panelC.txt", 3, None, None, None, ",", 0, None),
    "r15": Dataset("dataset/synthetic/R15.txt", 15, -1, None, None, ",", 0, None),
    "d31": Dataset("dataset/synthetic/D31.txt", 31, -1, None, None, ",", 0, None),
    "aggregation": Dataset("dataset/synthetic/Aggregation.txt", 7, -1, None, None, ",", 0, None),
    "compound": Dataset("dataset/synthetic/Compound.txt", 6, -1, None, None, ",", 0, None),
    "pathbased": Dataset("dataset/synthetic/Pathbased.txt", 3, -1, None, None, ",", 0, None),
}


# ==========================================================
# 数据加载
# ==========================================================

def load_dataset(dataset_name):
    """
    加载数据集
    """

    config = DATASETS[dataset_name]

    df = pd.read_csv(config.path, sep=config.sep, header=config.header, skiprows=config.skip_rows, comment=config.comment)

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
# DPC核心算法
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

def select_dc(dists, percent=2.0):
    """
    自动选择截断距离dc
    要求平均每个点周围距离小于dc的点的数目占总点数的1%-2%
    """

    n_samples = dists.shape[0]
    distance_vector = np.sort(dists.ravel())
    position = int(n_samples * (n_samples - 1) * percent / 100)
    dc = distance_vector[position + n_samples]
    return dc


def get_density(dists, dc, method="gaussian"):
    """
    计算局部密度ρ。每个点的局部密度
    """
    if method is None:
        rho = np.sum(dists < dc, axis=1 ) - 1
    else:
        rho = np.sum(np.exp(-(dists / dc) ** 2), axis=1 ) - 1

    return rho


def get_deltas(dists, rho):
    """
    计算密度距离δ
    即对每个点，找到密度比它大的所有点，再在这些点中找到距离其最近的点的距离
    """

    n_samples = dists.shape[0]
    deltas = np.zeros(n_samples)
    nearest_neighbor = np.zeros(n_samples, dtype=int)
    # 将密度从大到小排序
    index_rho = np.argsort(-rho)

    for i, index in enumerate(index_rho):
        # 对于密度最大的点
        if i == 0:
            continue
        # 对于其他的点，找到密度比其大的点的序号
        index_higher_rho = index_rho[:i]
        # 获取这些点距离当前点的距离,并找最小值
        distances = dists[index, index_higher_rho]
        # 保存最近邻点的编号
        deltas[index] = np.min(distances)
        nearest_neighbor[index] = (index_higher_rho[np.argmin(distances)])

    deltas[index_rho[0]] = np.max(deltas)
    return deltas, nearest_neighbor


def find_centers(rho, deltas, num_clusters=None):
    """
    选取聚类中心：选取rho与delta乘积较大的点作为聚类中心
    """

    if num_clusters is None:
        # 没有手动设置聚类数K，通过阈值选取rho与delta都大的点
        rho_threshold = (np.min(rho) + np.max(rho) ) / 2
        delta_threshold = (np.min(deltas) + np.max(deltas) ) / 2
        centers = np.where((rho >= rho_threshold) & (deltas > delta_threshold))[0]
        return centers
    else:
        # 手动输入聚类数K，选取rho与delta乘积最大的K个点作为聚类中心，作为k个中心点
        rho_delta = rho * deltas
        centers = np.argsort(-rho_delta)[:num_clusters]
        return centers


def cluster_pd(rho, centers, nearest_neighbor):
    """
    聚类分配
    """

    if len(centers) == 0:
        raise ValueError("Cannot find cluster centers.")

    n_samples = len(rho)

    labels_pred = -np.ones(n_samples, dtype=int)

    # 首先对几个聚类中进行标号
    for i, center in enumerate(centers):
        labels_pred[center] = i

    # 将密度从大到小排序
    index_rho = np.argsort(-rho)

    for index in index_rho:
        # 从密度大的点进行标号
        if labels_pred[index] == -1:
            # 如果没有被标记过，那么聚类标号与距离其最近且密度比其大的点的标号相同
            labels_pred[index] = (labels_pred[nearest_neighbor[index]])

    return labels_pred


def density_peak_clustering(features, num_clusters=None, percent=2.0, density_method="gaussian"):
    """
    Density Peaks Clustering
    """

    start = time.time()

    dists = get_distance_matrix(features)

    dc = select_dc(dists, percent)

    rho = get_density(dists, dc, density_method)

    deltas, nearest_neighbor = get_deltas(dists, rho)

    centers = find_centers(rho, deltas, num_clusters)

    labels_pred = cluster_pd(rho, centers, nearest_neighbor)

    elapsed_time = (time.time() - start)

    print( f"聚类中心数量={len(centers)}   " f"耗时={elapsed_time:.4f} s")

    return labels_pred, centers, rho, deltas, dc


# ==========================================================
# 标签对齐
# ==========================================================

def align_labels(labels_true, labels_pred):
    """
    使用匈牙利算法进行标签匹配
    """

    cm = confusion_matrix(labels_true, labels_pred)

    row_ind, col_ind = linear_sum_assignment(-cm)

    mapping = { pred: true for true, pred in zip(row_ind, col_ind)}

    labels_pred_aligned = np.array([
            mapping[label]
            for label in labels_pred
        ]
    )

    return labels_pred_aligned


# ==========================================================
# 聚类评价指标
# ==========================================================

def clustering_indicators(labels_true, labels_pred):
    """
    计算聚类评价指标
    """

    if isinstance(labels_true[0], str):
        # 如果标签为文本类型，把文本标签转换为数字标签
        labels_true = (LabelEncoder().fit_transform(labels_true))
    # 标签对齐
    labels_pred_aligned = align_labels(labels_true, labels_pred)
    f_measure = f1_score(labels_true, labels_pred_aligned, average="macro")  # F值
    accuracy = accuracy_score(labels_true, labels_pred_aligned)  # ACC
    normalized_mutual_information = normalized_mutual_info_score(labels_true, labels_pred)  # NMI
    rand_index = rand_score(labels_true, labels_pred)  # RI
    adjusted_rand_index = adjusted_rand_score(labels_true, labels_pred)  # ARI
    return f_measure, accuracy, normalized_mutual_information, rand_index, adjusted_rand_index


# ==========================================================
# 可视化
# ==========================================================

def draw_decision(rho, deltas):
    """
    绘制决策图
    """

    plt.figure(figsize=(6, 5))

    plt.scatter(rho, deltas, s=16, color="black")

    plt.xlabel("rho")
    plt.ylabel("delta")
    plt.title("Decision Graph")
    plt.show()


def draw_cluster(features, centers, labels_pred):
    """
    绘制聚类结果
    """

    features = np.asarray(features)

    if features.shape[1] > 2:
        pca = PCA(n_components=2)
        features_2d = (pca.fit_transform(features))
        centers_2d = pca.transform(features[centers])
    else:
        features_2d = features
        centers_2d = features[centers]

    plt.figure(figsize=(8, 6))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels_pred, cmap="nipy_spectral", s=7)
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], marker="x", color="red", s=50)
    plt.title( "DPC Clustering Result")
    plt.show()


# ==========================================================
# 主程序
# ==========================================================

if __name__ == "__main__":

    _dataset_name = "bezdekIris"

    print("加载数据集:   " f"数据集={_dataset_name}")

    _features, _labels_true, _num_clusters = load_dataset(_dataset_name)

    print(f"样本数量={_features.shape[0]}   " f"属性数量(维度)={_features.shape[1]}")

    # None：自动选中心（论文原始方式）
    # _num_clusters：固定K
    _num_clusters = None

    _labels_pred, _centers, _rho, _deltas, _dc = density_peak_clustering( _features, _num_clusters)

    print(f"dc={_dc:.6f}")
    print(_centers)

    if _labels_true is not None:
        # 有标签：计算聚类指标
        F1, ACC, NMI, RI, ARI = clustering_indicators(_labels_true, _labels_pred)
        print("聚类指标: "     f"F1={F1:.6f}  "     f"ACC={ACC:.6f}  "     f"NMI={NMI:.6f}  "     f"RI={RI:.6f}  "     f"ARI={ARI:.6f}" )

    draw_decision( _rho, _deltas)

    draw_cluster( _features, _centers, _labels_pred)