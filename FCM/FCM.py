"""
FCM.py

模糊C均值聚类（Fuzzy C-Means, FCM）实现

"""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import rand_score
from sklearn.metrics import adjusted_rand_score

# ==========================================================
# 数据集配置
# ==========================================================

DATASETS = {
    # (文件路径, 聚类簇数, 是否有标签列, 表头第一行是否为特征名)
    "iris": ("dataset/iris.csv", 3, True, True),
    "wine": ("dataset/wine.csv", 3, True, True),
    "seeds": ("dataset/seeds.csv", 3, True, True),
    "wdbc": ("dataset/wdbc.csv", 2, True, True),
    "glass": ("dataset/glass.csv", 6, True, True),

    # 人工合成数据集（无标签）
    "aggregation": ("dataset/Aggregation.csv", 7, True, True),
    "flame": ("dataset/Flame.csv", 2, True, True),
    "jain": ("dataset/Jain.csv", 2, True, True),
    "spiral": ("dataset/Spiral.csv", 3, True, True),
    "compound": ("dataset/Compound.csv", 6, True, True),
    "pathbased": ("dataset/Pathbased.csv", 3, True, True),
    "r15": ("dataset/R15.csv", 15, True, True),
    "d31": ("dataset/D31.csv", 31, True, True),
}


# ==========================================================
# 数据加载
# ==========================================================

def load_dataset(dataset_name):
    """
    加载数据集

    Parameters
    ----------
    dataset_name : str
        数据集名称

    Returns
    -------
    features : ndarray
                    特征矩阵

    labels_real : list
                  真实标签

    num_clusters : int
        聚类簇数
    """

    file_path, num_clusters, has_labels, has_header = DATASETS[dataset_name]

    df = pd.read_csv(file_path)

    features = df.iloc[:, :-1].values
    labels_true = df.iloc[:, -1].tolist()

    return features, labels_true, num_clusters


# ==========================================================
# FCM核心算法
# ==========================================================

def initialize_membership_matrix(n, c, random_state=42):
    """
    初始化隶属度矩阵
    """
    rng = np.random.default_rng(random_state)
    membership_matrix = rng.random((n, c))
    membership_matrix /= np.sum(membership_matrix, axis=1, keepdims=True)
    return membership_matrix


def calculate_cluster_center(features, membership_matrix, c, m):
    """
    根据隶属度矩阵计算聚类中心
    """
    n = len(features)
    cluster_centers = []
    cluster_mem_val_list = list(zip(*membership_matrix))
    for j in range(c):
        x = cluster_mem_val_list[j]
        x_raised = [e ** m for e in x]
        denominator = sum(x_raised)
        temp_num = []
        for i in range(n):
            data_point = features[i]
            prod = [x_raised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, zip(*temp_num))
        center = [z / denominator for z in numerator]
        cluster_centers.append(center)
    return np.array(cluster_centers)


def update_membership_value(features, membership_matrix, cluster_centers, c, m):
    """
    更新隶属度矩阵
    """
    n = len(features)
    power = 2.0 / (m - 1)

    for i in range(n):
        x = features[i]
        distances = np.array([
            np.linalg.norm(x - cluster_centers[j])
            for j in range(c)
        ])
        distances = np.maximum(distances, 1e-10)

        for j in range(c):
            den = sum([
                (distances[j] / distances[k]) ** power
                for k in range(c)
            ])
            membership_matrix[i][j] = 1.0 / den

    return membership_matrix


def get_clusters(membership_matrix):
    """
    获取聚类标签
    """
    return np.argmax(membership_matrix, axis=1)


def fuzzy_c_means_clustering(features, n_samples, num_clusters, tol=1e-5, fuzzifier=2.0, max_iter=100):
    """
    FCM聚类主函数

    Parameters
    ----------
    features : ndarray
              特征矩阵(符号：X)
    num_clusters : int
                聚类簇数(符号：c)
    n_samples : int
                样本总数(符号：n)
    tol : float, default=1e-5
          终止阈值容差tolerance(符号：epsilon)
    fuzzifier : float, default=2
                模糊化因子fuzzifier(符号：m)
    max_iter : int
               最大迭代数(符号：T)

    Returns
    -------
    cluster_labels : ndarray
                    聚类标签

    cluster_centers : list
                  聚类中心

    membership_matrix : int
        隶属度矩阵
    """

    # n_samples = len(features)
    start = time.time()
    membership_matrix = initialize_membership_matrix(n_samples, num_clusters)
    cluster_centers = np.zeros((num_clusters, features.shape[1]))  # 提前初始化，避免警告
    iteration = 0
    while iteration <= max_iter:
        cluster_centers = calculate_cluster_center(features, membership_matrix, num_clusters, fuzzifier)
        old_membership_mat = membership_matrix.copy()
        membership_matrix = update_membership_value(features, membership_matrix, cluster_centers, num_clusters, fuzzifier)
        if np.linalg.norm(membership_matrix - old_membership_mat) < tol:
            break
        iteration += 1
    cluster_labels = get_clusters(membership_matrix)
    elapsed_time = time.time() - start

    print(f"迭代次数: {iteration}")
    print(f"运行时间: {elapsed_time:.4f} s")

    return cluster_labels, cluster_centers, membership_matrix


# ==========================================================
# 标签对齐
# ==========================================================

def align_labels(labels_true, labels_pred):
    """
    使用匈牙利算法进行标签匹配
    """
    cm = confusion_matrix(labels_true, labels_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {}
    for true_label, pred_label in zip(row_ind, col_ind):
        mapping[pred_label] = true_label

    labels_pred_aligned = np.array([
        mapping[label]
        for label in labels_pred
    ])

    return labels_pred_aligned


# ==========================================================
# 聚类评价指标
# ==========================================================

def clustering_indicators(labels_true, labels_pred):
    """
    计算聚类评价指标
    """

    if isinstance(labels_true[0], str):
        # 真实标签转化为数字标签
        labels_true = LabelEncoder().fit_transform(labels_true)

    labels_pred_aligned = align_labels(labels_true, labels_pred)

    f1 = f1_score(labels_true, labels_pred_aligned, average="macro")
    accuracy = accuracy_score(labels_true, labels_pred_aligned)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    ri = rand_score(labels_true, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)

    return f1, accuracy, nmi, ri, ari


# ==========================================================
# 聚类结果可视化
# ==========================================================

def draw_cluster(features, cluster_centers, labels_pred):
    """
    绘制聚类结果散点图
    """

    features = np.asarray(features)
    cluster_centers = np.asarray(cluster_centers)

    if features.shape[1] > 2:
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        cluster_centers_2d = pca.transform(cluster_centers)
    else:
        features_2d = features
        cluster_centers_2d = cluster_centers

    plt.figure(figsize=(8, 6))
    # 做散点图
    # plt.scatter(features_2d[:, 0], features_2d[:, 1], marker='o', c='black', s=7)  # 原图
    # plt.show()
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels_pred, cmap="nipy_spectral", s=7, marker="o")

    plt.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1], marker="x", color="m", s=30)
    # 设置x和y坐标轴刻度的标签字体和字号
    # plt.xticks(fontproperties='Times New Roman', fontsize=10.5)
    # plt.yticks(fontproperties='Times New Roman', fontsize=10.5)
    # plt.xlabel("x - label", fontdict={'family': 'Times New Roman', 'size': 10.5}, loc="right")
    # plt.ylabel("y - label", fontdict={'family': 'Times New Roman', 'size': 10.5}, loc="top")
    plt.title("FCM Clustering Result")

    plt.show()


# ==========================================================
# 主程序
# ==========================================================

if __name__ == "__main__":
    # 初始化本次数据集名称
    _dataset_name = "iris"
    # 加载数据集
    _features, _labels_true, _num_clusters = load_dataset(_dataset_name)
    # 进行FCM算法
    _labels_pred, _cluster_centers, _membership_matrix = fuzzy_c_means_clustering(_features, len(_features), num_clusters=_num_clusters, tol=1e-5, fuzzifier=2.0, max_iter=100)
    # 计算聚类指标
    F1, ACC, NMI, RI, ARI = clustering_indicators(_labels_true, _labels_pred)
    print( f"F1={F1:.6f}  " f"ACC={ACC:.6f}  " f"NMI={NMI:.6f}  " f"RI={RI:.6f}  " f"ARI={ARI:.6f}")
    # 输出散点图
    draw_cluster(_features, _cluster_centers, _labels_pred)
