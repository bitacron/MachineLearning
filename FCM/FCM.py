from pylab import *
import pandas as pd
import numpy as np
import operator
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import normalized_mutual_info_score  # NMI
from sklearn.metrics import rand_score  # RI
from sklearn.metrics import accuracy_score  # ACC
from sklearn.metrics import f1_score  # F-measure
from sklearn.metrics import adjusted_rand_score  # ARI
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

# 数据保存在.csv文件中
iris = pd.read_csv("dataset/iris.csv", header=0)  # 鸢尾花数据集 Iris  class=3
wine = pd.read_csv("dataset/wine.csv")  # 葡萄酒数据集 Wine  class=3
seeds = pd.read_csv("dataset/seeds.csv")  # 小麦种子数据集 seeds  class=3
wdbc = pd.read_csv("dataset/wdbc.csv")  # 威斯康星州乳腺癌数据集 Breast Cancer Wisconsin (Diagnostic)  class=2
glass = pd.read_csv("dataset/glass.csv")  # 玻璃辨识数据集 Glass Identification  class=6
# aggregation = pd.read_csv("dataset/Aggregation.csv")  # class=7
# flame = pd.read_csv("dataset/Flame.csv")  # class=2
# jain = pd.read_csv("dataset/Jain.csv")  # class=2
# spiral = pd.read_csv("dataset/Spiral.csv")  # class=3
# compound = pd.read_csv("dataset/Compound.csv")  # class=6
# pathbased = pd.read_csv("dataset/Pathbased.csv")  # class=3
# r15 = pd.read_csv("dataset/R15.csv")  # class=15
# d31 = pd.read_csv("dataset/D31.csv")  # class=6
df = seeds  # 设置要读取的数据集
# print(df)
columns = list(df.columns)  # 获取数据集的第一行，第一行通常为特征名，所以先取出
features = columns[:len(columns) - 1]  # 数据集的特征名（去除了最后一列，因为最后一列存放的是标签，不是数据）
dataset = df[features]  # 预处理之后的数据，去除掉了第一行的数据（因为其为特征名，如果数据第一行不是特征名，可跳过这一步）
original_labels = list(df[columns[-1]])  # 原始标签（最后一列）
attributes = len(df.columns) - 1  # 属性数量（数据集维度）


# 初始化模糊矩阵U
def initializeMembershipMatrix():
    num_samples = df.shape[0]
    membership_mat = np.random.rand(num_samples, c)
    membership_mat = membership_mat / np.sum(membership_mat, axis=1, keepdims=True)
    return membership_mat


# 计算类中心点
def calculateClusterCenter(membership_mat, c, m):
    cluster_mem_val = zip(*membership_mat)
    cluster_centers = list()
    cluster_mem_val_list = list(cluster_mem_val)
    for j in range(c):
        x = cluster_mem_val_list[j]
        x_raised = [e ** m for e in x]
        denominator = sum(x_raised)
        temp_num = list()
        for i in range(n):
            data_point = list(dataset.iloc[i])
            prod = [x_raised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, zip(*temp_num))
        center = [z / denominator for z in numerator]  # 每一维都要计算。
        cluster_centers.append(center)
    return cluster_centers


# 更新隶属度
def updateMembershipValue(membership_mat, cluster_centers, c):
    #    p = float(2/(m-1))
    data = []
    for i in range(n):
        x = list(dataset.iloc[i])  # 取出文件中的每一行数据
        data.append(x)
        distances = [np.linalg.norm(list(map(operator.sub, x, cluster_centers[j]))) for j in range(c)]
        for j in range(c):
            den = sum([math.pow(float(distances[j] / distances[k]), 2) for k in range(c)])
            membership_mat[i][j] = float(1 / den)
    return membership_mat, data


# 得到聚类结果
def getClusters(membership_mat):
    cluster_labels = list()
    for i in range(n):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels


# FCM算法
def fuzzyCMeansClustering(c, epsilon, m, T):
    start = time.time()  # 开始时间，计时
    membership_mat = initializeMembershipMatrix()  # 初始化隶属度矩阵
    t = 0
    cluster_labels = []
    cluster_centers = []
    while t <= T:  # 最大迭代次数
        cluster_centers = calculateClusterCenter(membership_mat, c, m)  # 根据隶属度矩阵计算聚类中心
        old_membership_mat = membership_mat.copy()  # 保留之前的隶属度矩阵，同于判断迭代条件
        membership_mat, data = updateMembershipValue(membership_mat, cluster_centers, c)  # 新一轮迭代的隶属度矩阵
        cluster_labels = getClusters(membership_mat)  # 获取标签
        if np.linalg.norm(membership_mat - old_membership_mat) < epsilon:
            break
        print("第", t, "次迭代")
        t += 1

    print("用时：{0}".format(time.time() - start))
    # print(membership_mat)
    return cluster_labels, cluster_centers, data, membership_mat

# 调用confusion_matrix函数，使用匈牙利算法，进行标签匹配（Label Matching）
def align_labels(labels_true, labels_pred):
    cm = confusion_matrix(labels_true, labels_pred)

    row_ind, col_ind = linear_sum_assignment(-cm)

    mapping = {}
    for true_label, pred_label in zip(row_ind, col_ind):
        mapping[pred_label] = true_label

    labels_pred_aligned = np.array(
        [mapping[label] for label in labels_pred]
    )

    return labels_pred_aligned


# 计算聚类指标
def clustering_indicators(labels_true, labels_pred):
    if type(labels_true[0]) != int:
        labels_true = LabelEncoder().fit_transform(df[columns[len(columns)-1]])  # 如果标签为文本类型，把文本标签转换为数字标签
    # 标签对齐
    labels_pred_aligned = align_labels(labels_true, labels_pred)
    f_measure = f1_score(labels_true, labels_pred_aligned, average='macro')  # F值
    accuracy = accuracy_score(labels_true, labels_pred_aligned)  # ACC
    normalized_mutual_information = normalized_mutual_info_score(labels_true, labels_pred)  # NMI
    rand_index = rand_score(labels_true, labels_pred)  # RI
    adjusted_rand_index = adjusted_rand_score(labels_true, labels_pred)  # ARI
    return f_measure, accuracy, normalized_mutual_information, rand_index, adjusted_rand_index


# 绘制聚类结果散点图
def draw_cluster(dataset, centers, labels):
    center_array = array(centers)
    if attributes > 2:
        dataset = PCA(n_components=2).fit_transform(dataset)  # 如果属性数量大于2，降维
        center_array = PCA(n_components=2).fit_transform(center_array)  # 如果属性数量大于2，降维
    else:
        dataset = array(dataset)
    # 做散点图
    label = array(labels)
    plt.scatter(dataset[:, 0], dataset[:, 1], marker='o', c='black', s=7)  # 原图
    # plt.show()
    colors = np.array(
        ["#FF0000", "#0000FF", "#00FF00", "#FFFF00", "#00FFFF", "#FF00FF", "#800000", "#008000", "#000080", "#808000",
         "#800080", "#008080", "#444444", "#FFD700", "#008080"])
    # 循换打印k个簇，每个簇使用不同的颜色
    for i in range(c):
        plt.scatter(dataset[nonzero(label == i), 0], dataset[nonzero(label == i), 1], c=colors[i], s=7, marker='o')
    plt.scatter(center_array[:, 0], center_array[:, 1], marker='x', color='m', s=30)  # 聚类中心
    # 设置x和y坐标轴刻度的标签字体和字号
    # plt.xticks(fontproperties='Times New Roman', fontsize=10.5)
    # plt.yticks(fontproperties='Times New Roman', fontsize=10.5)
    # plt.xlabel("x - label", fontdict={'family': 'Times New Roman', 'size': 10.5}, loc="right")
    # plt.ylabel("y - label", fontdict={'family': 'Times New Roman', 'size': 10.5}, loc="top")
    plt.show()


if __name__ == "__main__":
    c = 3  # 聚类簇数
    T = 100  # 最大迭代数
    n = len(dataset)  # 样本数
    m = 2.00  # 模糊参数
    epsilon = 1e-5
    labels, centers, data, membership = fuzzyCMeansClustering(c, epsilon, m, T)  # 运行FCM算法
    F_measure, ACC, NMI, RI, ARI = clustering_indicators(original_labels, labels)  # 计算聚类指标
    print("F_measure:", F_measure, "ACC:", ACC, "NMI", NMI, "RI", RI, "ARI", ARI)
    # print(membership)
    # print(centers)
    # print(dataset)
    draw_cluster(dataset, centers, labels)
