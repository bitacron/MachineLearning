import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import normalized_mutual_info_score  # NMI
from sklearn.metrics import rand_score  # RI
from sklearn.metrics import accuracy_score  # ACC
from sklearn.metrics import f1_score  # F-measure
from sklearn.metrics import adjusted_rand_score  # ARI

# 数据保存在.csv文件中
iris = pd.read_csv("dataset/iris.csv", header=0)  # 鸢尾花数据集 Iris  class=3
wine = pd.read_csv("dataset/wine.csv")  # 葡萄酒数据集 Wine  class=3
seeds = pd.read_csv("dataset/seeds.csv")  # 小麦种子数据集 seeds  class=3
wdbc = pd.read_csv("dataset/wdbc.csv")  # 威斯康星州乳腺癌数据集 Breast Cancer Wisconsin (Diagnostic)  class=2
glass = pd.read_csv("dataset/glass.csv")  # 玻璃辨识数据集 Glass Identification  class=6
jain = pd.read_csv("dataset/Jain.csv")  # 人工数据集
spiral = pd.read_csv("dataset/spiral.csv")  # 人工数据集
flame = pd.read_csv("dataset/Flame.csv")  # 人工数据集
aggregation = pd.read_csv("dataset/Aggregation.csv")  # 人工数据集
df = spiral  # 设置要读取的数据集
# print(df)
columns = list(df.columns)  # 获取数据集的第一行，第一行通常为特征名，所以先取出
features = columns[:len(columns) - 1]  # 数据集的特征名（去除了最后一列，因为最后一列存放的是标签，不是数据）
dataset = df[features]  # 预处理之后的数据，去除掉了第一行的数据（因为其为特征名，如果数据第一行不是特征名，可跳过这一步）
attributes = len(df.columns) - 1  # 属性数量（数据集维度）
original_labels = list(df[columns[-1]])  # 原始标签


# 计算数据点两两之间的距离
def getDistanceMatrix(datas):
    N, D = np.shape(datas)
    dists = np.zeros([N, N])

    for i in range(N):
        for j in range(N):
            vi = datas[i, :]
            vj = datas[j, :]
            dists[i, j] = np.sqrt(np.dot((vi - vj), (vi - vj)))
    return dists


# 找到密度计算的阈值dc
# 要求平均每个点周围距离小于dc的点的数目占总点数的1%-2%
def select_dc(dists):
    N = np.shape(dists)[0]
    tt = np.reshape(dists, N * N)
    percent = 2.0
    position = int(N * (N - 1) * percent / 100)
    dc = np.sort(tt)[position + N]
    return dc


# 计算每个点的局部密度
def get_density(dists, dc, method=None):
    N = np.shape(dists)[0]
    rho = np.zeros(N)

    for i in range(N):
        if method == None:
            rho[i] = np.where(dists[i, :] < dc)[0].shape[0] - 1
        else:
            rho[i] = np.sum(np.exp(-(dists[i, :] / dc) ** 2)) - 1
    return rho


# 计算每个数据点的密度距离
# 即对每个点，找到密度比它大的所有点
# 再在这些点中找到距离其最近的点的距离
def get_deltas(dists, rho):
    N = np.shape(dists)[0]
    deltas = np.zeros(N)
    nearest_neiber = np.zeros(N)
    # 将密度从大到小排序
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        # 对于密度最大的点
        if i == 0:
            continue

        # 对于其他的点
        # 找到密度比其大的点的序号
        index_higher_rho = index_rho[:i]
        # 获取这些点距离当前点的距离,并找最小值
        deltas[index] = np.min(dists[index, index_higher_rho])

        # 保存最近邻点的编号
        index_nn = np.argmin(dists[index, index_higher_rho])
        nearest_neiber[index] = index_higher_rho[index_nn].astype(int)

    deltas[index_rho[0]] = np.max(deltas)
    return deltas, nearest_neiber


# 选取rho与delta乘积较大的点作为聚类中心
def find_centers(rho, deltas, K=0):
    if K == 0:
        # 没有手动设置聚类数K，通过阈值选取rho与delta都大的点
        rho_threshold = (np.min(rho) + np.max(rho)) / 2
        delta_threshold = (np.min(deltas) + np.max(deltas)) / 2
        N = np.shape(rho)[0]
        centers = []
        for i in range(N):
            if rho[i] >= rho_threshold and deltas[i] > delta_threshold:
                centers.append(i)
        return np.array(centers)
    else:
        # 手动输入聚类数K，选取rho与delta乘积最大的K个点作为聚类中心，作为k个中心点
        rho_delta = rho * deltas
        centers = np.argsort(-rho_delta)
        return centers[:K]


def cluster_PD(rho, centers, nearest_neiber):
    K = np.shape(centers)[0]
    if K == 0:
        print("can not find centers")
        return

    N = np.shape(rho)[0]
    label = -1 * np.ones(N).astype(int)

    # 首先对几个聚类中进行标号
    for i, center in enumerate(centers):
        label[center] = i

    # 将密度从大到小排序
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        # 从密度大的点进行标号
        if label[index] == -1:
            # 如果没有被标记过
            # 那么聚类标号与距离其最近且密度比其大
            # 的点的标号相同
            label[index] = label[int(nearest_neiber[index])]
    return label


def draw_decision(rho, deltas):
    plt.cla()
    for i in range(np.shape(datas)[0]):
        plt.scatter(rho[i], deltas[i], s=16., color=(0, 0, 0))
        plt.annotate(str(i), xy=(rho[i], deltas[i]), xytext=(rho[i], deltas[i]))
        plt.xlabel("rho")
        plt.ylabel("deltas")
    # plt.savefig(filename+"_decision.jpg")
    plt.show()


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


def draw_cluster(datas, label, centers):
    plt.cla()
    K = np.shape(centers)[0]
    colors = np.array(["blue", "red", "green", "orange", "purple", "cyan", "magenta", "beige", "hotpink", "#88c999"])
    if attributes > 2:
        datas = PCA(n_components=2).fit_transform(datas)  # 如果属性数量大于2，降维
    for i in range(K):
        plt.scatter(datas[np.nonzero(label == i), 0], datas[np.nonzero(label == i), 1], c=colors[i], s=7)
        # plt.scatter(datas[centers[i], 0], datas[centers[i], 1], color="k", marker="+", s=200.)  # 聚类中心
    # plt.savefig(file_name + "_cluster.jpg")
    # 设置x和y坐标轴刻度的标签字体和字号
    plt.xticks(fontproperties='Times New Roman', fontsize=10.5)
    plt.yticks(fontproperties='Times New Roman', fontsize=10.5)
    plt.show()


if __name__ == "__main__":

    # 主程序
    datas = np.array(dataset).astype(np.float32)
    # 计算距离矩阵
    dists = getDistanceMatrix(datas)
    # 计算dc
    dc = select_dc(dists)
    print("dc", dc)
    # 计算局部密度ρ
    rho = get_density(dists, dc, method="Gaussion")
    # 计算密度距离δ
    deltas, nearest_neiber = get_deltas(dists, rho)

    # 绘制密度/距离分布图
    draw_decision(rho, deltas)

    # 获取聚类中心点
    centers = find_centers(rho, deltas)
    print("centers", centers)

    label = cluster_PD(rho, centers, nearest_neiber)
    plt.cla()
    K = np.shape(centers)[0]
    F_measure, ACC, NMI, RI, ARI = clustering_indicators(original_labels, label)  # 计算聚类指标
    print("F_measure:", F_measure, "ACC:", ACC, "NMI", NMI, "RI", RI, "ARI", ARI)
    draw_cluster(datas, label, centers)
