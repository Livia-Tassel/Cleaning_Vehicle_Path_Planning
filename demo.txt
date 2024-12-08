import csv
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
import random
from sklearn.cluster import SpectralClustering
import heapq
import matplotlib.font_manager as fm
import logging
import os

# 配置日志记录，将日志输出到log.txt文件
logging.basicConfig(
    filename='log.txt',
    filemode='w',  # 写入模式，覆盖之前的内容
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# 设置中文字体
def set_chinese_font():
    # 查找系统中的中文字体，例如 SimHei 或 Microsoft YaHei
    font_paths = [
        "C:\\Windows\\Fonts\\simhei.ttf",  # Windows 系统
        "/usr/share/fonts/truetype/arphic/SimHei.ttf",  # Linux 系统
        "/Library/Fonts/SIMHEI.TTF"  # macOS 系统
    ]
    for font_path in font_paths:
        try:
            prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = prop.get_name()
            break
        except FileNotFoundError:
            continue
    else:
        # 如果没有找到中文字体，使用默认字体并禁用中文
        plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

set_chinese_font()

# 定义 Graph 类来存储图的结构
class Graph:
    def __init__(self):
        self.graph = nx.Graph()  # 使用 networkx 创建无向图
        self.node_positions = {}  # 存储节点坐标
        self.edge_dirtiness = {}  # 存储每条边的脏度
        self.edge_decay = {}  # 存储每条边的脏度增长系数

    # 无向图构建
    def add_edge(self, start, end, length):
        """
        添加边到图中，并初始化脏度和变化系数
        """
        self.graph.add_edge(start, end, weight=length)
        # 使用 (min, max) 来处理无向边，保证键的一致性
        # 新添加的边都会拥有自己的 edge_key，并在 edge_dirtiness 字典中有对应的脏度值
        edge_key = (min(start, end), max(start, end))
        # 随机初始化脏度在 [10, 20] 之间
        self.edge_dirtiness[edge_key] = random.uniform(10, 20)
        # 随机初始化脏度增长系数 a 在 [0.5, 1.0] 之间，线性增长
        self.edge_decay[edge_key] = random.uniform(0.5, 1.0)

    def load_edges_from_csv(self, file_path):
        """
        从 CSV 文件加载边的数据
        """
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # 跳过标题行
            for row in reader:
                start, end, length = int(row[0]), int(row[1]), float(row[2])
                self.add_edge(start, end, length)

    def load_points_from_csv(self, file_path):
        """
        从 CSV 文件加载点的坐标数据
        """
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # 跳过标题行
            for row in reader:
                point_id = int(row[0])
                x, y = float(row[1]), float(row[2])
                self.node_positions[point_id] = (x, y)

    # 平滑曲线
    def smooth_line(self, start, end, curvature_factor=0.1):
        """
        计算平滑的直线，边缘稍微过渡自然
        """
        p0 = self.node_positions[start]
        p1 = self.node_positions[end]

        # 计算控制点，使得边略微弯曲
        control_point = (
            (p0[0] + p1[0]) / 2 + np.random.uniform(-curvature_factor, curvature_factor),
            (p0[1] + p1[1]) / 2 + np.random.uniform(-curvature_factor, curvature_factor)
        )

        # 生成插值的平滑线段
        t_values = np.linspace(0, 1, 100)  # 从0到1的参数值
        curve_points = []

        for t in t_values:
            x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * control_point[0] + t ** 2 * p1[0]
            y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * control_point[1] + t ** 2 * p1[1]
            curve_points.append((x, y))

        return curve_points

    # 根据边的脏度创建邻接矩阵，用于后续谱聚类分析
    def compute_adjacency_matrix(self):
        """
        计算邻接矩阵，用于谱聚类
        """
        nodes = list(self.graph.nodes())
        n = len(nodes)
        adj_matrix = np.zeros((n, n))

        node_index = {node: idx for idx, node in enumerate(nodes)}  # node 映射到索引

        # 为每对节点之间的边赋值
        for edge in self.graph.edges():
            u, v = edge
            edge_key = (min(u, v), max(u, v))
            dirtiness = self.edge_dirtiness.get(edge_key, 0)
            i, j = node_index[u], node_index[v]
            adj_matrix[i][j] = dirtiness
            adj_matrix[j][i] = dirtiness  # 无向图对称

        return adj_matrix, nodes

    # 根据邻接矩阵脏度信息对节点进行谱聚类，划分校园为n_clusters区域
    def cluster_nodes(self, n_clusters=3):
        """
        使用谱聚类对图中的节点进行区域划分
        """
        adj_matrix, nodes = self.compute_adjacency_matrix()

        # 使用 SpectralClustering 进行谱聚类
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
        labels = clustering.fit_predict(adj_matrix)

        # 返回每个节点的聚类标签
        node_clusters = {node: labels[i] for i, node in enumerate(nodes)}
        return node_clusters

    # 边脏度随时间步增加，最大100
    def update_dirtiness(self):
        """
        在每个时间步更新所有道路的脏度，脏度线性增加
        """
        for edge_key in self.edge_dirtiness:
            self.edge_dirtiness[edge_key] += self.edge_decay[edge_key]
            # 确保脏度最大为100
            if self.edge_dirtiness[edge_key] > 100:
                self.edge_dirtiness[edge_key] = 100

    # 模拟清洁车清扫降低脏度
    def decrease_dirtiness(self, edge_key, amount):
        """
        减少指定道路的脏度
        """
        if edge_key in self.edge_dirtiness:
            self.edge_dirtiness[edge_key] = max(0, self.edge_dirtiness[edge_key] - amount)

    # 根据当前图的状态动态绘制校园地图以及节点
    def draw(self, ax, node_clusters, time_step):
        """
        绘制图，使用 matplotlib 库进行可视化
        根据脏度随时间变化的计算来动态调整边的颜色和宽度
        """
        ax.clear()  # 清除当前图形
        ax.set_title(f"校园清洁模拟 - 时间步: {time_step}", fontsize=16, fontweight='bold')  # 使用传入的time_step

        # 设置背景颜色
        ax.set_facecolor('whitesmoke')  # 灰白色

        # 更新每条边的颜色和宽度
        for edge in self.graph.edges:
            start, end = edge
            edge_key = (min(start, end), max(start, end))
            dirtiness = self.edge_dirtiness[edge_key]  # 当前脏度

            # 通过脏度映射到颜色和边的宽度（脏度越高，颜色越深，边宽越大）
            color = plt.cm.Reds(dirtiness / 100)  # 使用红色 colormap
            width = 1 + (dirtiness / 100) * 4  # 使脏度越高边宽越大

            # 计算平滑的直线
            curve_points = self.smooth_line(start, end)
            x_values, y_values = zip(*curve_points)

            # 绘制曲线
            ax.plot(x_values, y_values, color=color, lw=width)

        # 绘制节点，节点根据聚类进行颜色分配
        node_size = [self.graph.degree(node) * 50 for node in self.graph.nodes()]  # 调整节点大小
        # 存储每个节点的颜色列表
        colors = []
        for node in self.graph.nodes():
            cluster_id = node_clusters[node]
            if cluster_id == 0:
                colors.append('skyblue')
            elif cluster_id == 1:
                colors.append('lightgreen')
            else:
                colors.append('salmon')
        # alpha=0.8为透明参数，ax为坐标轴对象
        nx.draw_networkx_nodes(self.graph, self.node_positions, node_size=node_size, node_color=colors, alpha=0.8, ax=ax)

        # 去除数字标签，不再调用 nx.draw_networkx_labels
        # nx.draw_networkx_labels(self.graph, self.node_positions, font_size=10, font_color="black", font_weight="bold",
        #                         bbox=dict(facecolor='none', edgecolor='none'), ax=ax)

        # 添加网格和轴标签
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim(-100, 1100)
        ax.set_ylim(-100, 1100)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)

        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='区域 1', markerfacecolor='skyblue', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='区域 2', markerfacecolor='lightgreen', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='区域 3', markerfacecolor='salmon', markersize=10)
        ]
        ax.legend(handles=legend_elements, loc='upper right')

# 定义 Cleaner 类
class Cleaner:
    def __init__(self, graph, speed=None, start_node=None, cleaner_id=0, unique_id=0):
        """
        初始化清洁车
        graph: 图对象
        speed: 清洁车的速度，单位：米/秒，随机在[1, 1.5]范围内
        start_node: 清洁车的起始位置
        cleaner_id: 清洁车所属区域ID
        unique_id: 清洁车的唯一ID
        """
        self.graph = graph  # 图对象
        self.speed = random.uniform(1, 1.5) if speed is None else speed  # 清洁车速度随机初始化
        self.current_node = start_node  # 当前节点
        self.path = []  # 清洁车的路径
        self.target_edge = None  # 当前目标边
        self.cleaner_id = cleaner_id  # 清洁车所属区域ID
        self.unique_id = unique_id  # 清洁车唯一ID

    # 设置清洁车路径
    def set_path(self, path):
        """ 设置清洁车的路径 """
        self.path = path

    # 行驶函数
    def move(self):
        """
        清洁车沿路径移动，每次移动到下一个节点，并更新经过路径的脏度。
        """
        if not self.path:
            logging.info(f"清洁车 {self.unique_id} 没有路径可执行。")
            return  # 如果没有路径，停止移动

        next_node = self.path.pop(0)  # 获取下一个节点
        edge_key = (min(self.current_node, next_node), max(self.current_node, next_node))

        # 清洁车经过的边脏度减少 [10-20]
        dirtiness_decrease = random.uniform(10, 20)  # 脏度减少范围
        self.graph.decrease_dirtiness(edge_key, dirtiness_decrease)
        logging.info(f"清洁车 {self.unique_id} 从节点 {self.current_node} 移动到节点 {next_node}，减少边 {edge_key} 脏度 {dirtiness_decrease:.2f}")

        # 更新当前节点
        self.current_node = next_node

        # 检查是否完成路径
        if not self.path:
            logging.info(f"清洁车 {self.unique_id} 完成当前路径，准备规划新路径。")
            self.target_edge = None  # 路径完成，重置目标边

    # 获取清洁车
    def get_position(self):
        """
        返回清洁车的当前位置（当前节点）
        """
        return self.graph.node_positions[self.current_node]

def astar(graph, start, end):
    """
    使用A*算法计算从起点到终点的最短路径
    """
    # 确保起点和终点存在于图中
    if start not in graph.node_positions or end not in graph.node_positions:
        raise ValueError("起点或终点不在图中")

    # A*启发函数(Heuristic)
    def heuristic(node, end):
        # 获取起点与终点的坐标
        # 使用欧几里得距离作为启发式函数
        x1, y1 = graph.node_positions[node]
        x2, y2 = graph.node_positions[end]
        return math.hypot(x2 - x1, y2 - y1)

    # 初始化优先队列(存储待扩展节点及相应代价信息)
    # heapq将元组(f, g, node)压入open_list中
    # f为总估计代价=g+h(g为起点到当前节点的实际代价即最短距离，h为启发值即heuristic(node, end))
    open_list = []
    # 首个节点(起点)
    heapq.heappush(open_list, (0 + heuristic(start, end), 0, start))  # (f, g, node)
    # came_from记录节点的前驱节点，以便回溯路径
    came_from = {}
    # g_costs记录起点到各节点的当前已知最小实际代价，初始化为 {start: 0}
    g_costs = {start: 0}
    # 闭合集合，用于记录已扩展的节点
    closed_set = set()

    while open_list:
        # heapq.heappop自 open_list 中取出 f 值最小的节点(即当前最优的扩展节点)
        # current_node为当前正在扩展的节点
        # g为从起点到 current_node 的实际代价
        _, g, current_node = heapq.heappop(open_list)

        # 如果当前节点已经被扩展，跳过
        if current_node in closed_set:
            continue

        # 添加到闭合集合，表示已经扩展过
        closed_set.add(current_node)

        # 若 current_node 为终点 end，则已经找到由起点到终点的路径
        if current_node == end:
            # 空列表 path
            path = []
            # 从终点开始，依次查找各节点的前驱节点，直到回溯到起点
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(start)
            path = path[::-1]
            # 验证路径完整性
            if path[0] == start and path[-1] == end:
                return path
            else:
                raise RuntimeError("路径回溯错误")

        # 获取当前节点的所有直接邻居
        for neighbor in graph.graph.neighbors(current_node):
            edge_key = (min(current_node, neighbor), max(current_node, neighbor))
            distance = graph.graph[current_node][neighbor].get("weight", None)
            if distance is None or distance < 0:
                raise ValueError(f"边 ({current_node}, {neighbor}) 的权重无效")
            # 起点到邻居节点 neighbor 的临时代价
            tentative_g = g + distance
            # 若邻居节点 neighbor 已经在闭合集合中，跳过
            if neighbor in closed_set:
                continue
            # 若邻居节点 neighbor 未被访问过(即不在 g_costs 中)或者找到更短的路径(tentative_g < g_costs[neighbor])
            if neighbor not in g_costs or tentative_g < g_costs[neighbor]:
                # 记录邻居节点的前驱为 current_node
                came_from[neighbor] = current_node
                # 记录从起点到邻居节点的新的最优实际代价
                g_costs[neighbor] = tentative_g
                # 更新总估计代价
                f_cost = tentative_g + heuristic(neighbor, end)
                # 将邻居节点按新的 f 值加入优先队列，以备后续扩展
                heapq.heappush(open_list, (f_cost, tentative_g, neighbor))

    return []  # 如果没有路径

# 规划每辆车清洁区域
# 更新清洁车路径以及目标边
def plan_cleaning_paths(graph, cleaners, node_clusters, targeted_edges):
    """
    为每个清洁车规划路径，确保覆盖其所属区域内的所有边
    优先选择脏度最高的未清扫道路
    """
    for cleaner in cleaners:
        # 只有当清洁车没有当前路径时，才规划新的路径
        if cleaner.path:
            logging.info(f"清洁车 {cleaner.unique_id} 正在执行路径，跳过规划。")
            continue  # 清洁车当前有路径，跳过规划

        # 获取清洁车所属区域的边
        cluster_id = cleaner.cleaner_id  # 使用cleaner_id作为区域ID
        edges_in_cluster = [
            edge for edge in graph.graph.edges()
            if node_clusters[edge[0]] == cluster_id or node_clusters[edge[1]] == cluster_id
        ]

        # 在清洁车负责区域筛选脏度>0且未被其他清洁车目标化的边
        edges_to_clean = [
            edge for edge in edges_in_cluster
            if graph.edge_dirtiness[(min(edge), max(edge))] > 0 and (min(edge), max(edge)) not in targeted_edges
        ]

        if not edges_to_clean:
            logging.info(f"清洁车 {cleaner.unique_id} 所属区域内所有边已清扫或被其他清洁车选定。")
            continue  # 该区域内所有边已清扫或被其他清洁车选定

        # 按照脏度降序排序
        edges_to_clean.sort(key=lambda edge: graph.edge_dirtiness[(min(edge), max(edge))], reverse=True)

        # 选择脏度最高的边作为目标
        target_edge = edges_to_clean[0]

        # 标记该边为已被选定
        targeted_edges.add(target_edge)

        # 确定目标节点（选择距离当前节点最近的端点）
        if cleaner.current_node == target_edge[0]:
            target_node = target_edge[1]
        elif cleaner.current_node == target_edge[1]:
            target_node = target_edge[0]
        else:
            # 选择距离当前节点较近的端点
            # 欧几里得距离
            distance0 = math.hypot(
                graph.node_positions[cleaner.current_node][0] - graph.node_positions[target_edge[0]][0],
                graph.node_positions[cleaner.current_node][1] - graph.node_positions[target_edge[0]][1]
            )
            distance1 = math.hypot(
                graph.node_positions[cleaner.current_node][0] - graph.node_positions[target_edge[1]][0],
                graph.node_positions[cleaner.current_node][1] - graph.node_positions[target_edge[1]][1]
            )
            target_node = target_edge[0] if distance0 < distance1 else target_edge[1]

        # 若清洁车当前路径已经指向目标边，则无需重新规划
        if cleaner.target_edge != target_edge:
            # 使用 A* 算法规划到目标边的路径
            path_to_target = astar(graph, cleaner.current_node, target_node)
            if path_to_target:
                # 若 A* 返回路径为 [current_node, next_node, ...]
                # 移除当前节点，避免重复
                if path_to_target[0] == cleaner.current_node:
                    path_to_target = path_to_target[1:]
                # 更新清洁车路径以及目标边
                cleaner.set_path(path_to_target)
                cleaner.target_edge = target_edge
                logging.info(f"清洁车 {cleaner.unique_id} 规划新路径: {path_to_target}, 目标边: {target_edge}")

# 根据动画帧重绘图形
def animate_simulation(time_step, graph, cleaners, node_clusters, ax, targeted_edges):
    """
    动画中每个帧的更新函数
    """
    # 每个时间步更新道路脏度
    graph.update_dirtiness()

    # 为各清洁车规划路径
    # 脏度最高作为优先级
    # A*规划路径
    plan_cleaning_paths(graph, cleaners, node_clusters, targeted_edges)

    # 移动清洁车
    for cleaner in cleaners:
        cleaner.move()

    # 绘制图形，反映脏度更新和清洁车位置
    graph.draw(ax, node_clusters, time_step)

    # 绘制清洁车的位置
    for cleaner in cleaners:
        pos = cleaner.get_position()
        ax.plot(pos[0], pos[1], marker='o', markersize=10, markeredgecolor='black',
                markerfacecolor='blue')

    # 添加时间步信息（已在 graph.draw 中设置，不再需要）
    # ax.set_title(f"校园清洁模拟 - 时间步: {time_step}", fontsize=16, fontweight='bold')

def main():
    # 创建图对象并加载数据
    graph = Graph()
    graph.load_edges_from_csv('Edges.csv')  # 加载边数据
    graph.load_points_from_csv('Points.csv')  # 加载点数据

    # 定义区域数
    n_clusters = 3  # 将地图分为3个区域

    # 节点聚类
    node_clusters = graph.cluster_nodes(n_clusters=n_clusters)

    # 初始化清洁车
    # 创建空列表 cleaners
    cleaners = []
    # 定义每个区域的清洁车数量，按照3-2-2分配
    cleaners_per_cluster = {0: 3, 1: 2, 2: 2}
    unique_cleaner_id = 0  # 用于分配唯一的清洁车ID

    # 用于跟踪已被选定的边，防止多辆清洁车选择同一边
    targeted_edges = set()

    for cluster_id in range(n_clusters):
        # 获取当前区域的节点
        nodes_in_cluster = [node for node, cluster in node_clusters.items() if cluster == cluster_id]
        if nodes_in_cluster:
            # 按照指定数量创建清洁车
            num_cleaners = cleaners_per_cluster.get(cluster_id, 1)
            for _ in range(num_cleaners):
                # 随机选择节点作为清洁车的起始位置
                start_node = random.choice(nodes_in_cluster)

                # 创建 Cleaner 实例
                cleaner = Cleaner(graph=graph, start_node=start_node, cleaner_id=cluster_id, unique_id=unique_cleaner_id)
                cleaners.append(cleaner)
                logging.info(f"初始化清洁车 {cleaner.unique_id}，所属区域 {cluster_id}，起始节点 {start_node}")
                unique_cleaner_id += 1  # 增加唯一ID

    # 创建动画的绘图窗口和坐标轴(12英寸宽、10英寸高)
    fig, ax = plt.subplots(figsize=(12, 10))

    # 定义动画每帧的更新函数
    def update(frame):
        animate_simulation(frame, graph, cleaners, node_clusters, ax, targeted_edges)

    # 动画的帧数范围1-100（共100帧），每帧会传递给 update 函数作为 frame 参数
    # interval=500：每帧之间的时间间隔，以毫秒为单位，设置为500ms，即每秒更新2帧
    ani = animation.FuncAnimation(fig, update, frames=range(1, 101), interval=500, repeat=False)

    # 显示图形
    plt.show()

if __name__ == "__main__":
    main()
