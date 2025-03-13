import networkx as nx
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import simpledialog, messagebox


class Graph:
    def __init__(self):
        self.graph = nx.Graph()
        self.edges = []

    def add_edge(self, u, v, weight):
        self.graph.add_edge(u, v, weight=weight)
        self.edges.append((u, v, weight))

    def from_adjacency_matrix(self, matrix):
        size = len(matrix)
        for i in range(size):
            for j in range(i + 1, size):
                if matrix[i][j] > 0:
                    self.add_edge(i, j, matrix[i][j])

    def from_incidence_matrix(self, matrix):
        edges = []
        for j in range(len(matrix[0])):
            nodes = [i for i in range(len(matrix)) if matrix[i][j] == 1]
            if len(nodes) == 2:
                edges.append(tuple(nodes))
        for u, v in edges:
            weight = random.randint(1, 10)
            self.add_edge(u, v, weight)


class MSTVisualizer:
    def __init__(self, graph):
        self.graph = graph.graph
        self.mst_edges = []
        self.fig, self.ax = plt.subplots()
        self.pos = nx.spring_layout(self.graph)
        self.steps = []
        self.ani = None

    def prim(self):
        mst_edges = []
        visited = set()
        edges = sorted((self.graph[u][v]['weight'], u, v) for u, v in self.graph.edges())
        start_node = edges[0][1]
        visited.add(start_node)
        while len(visited) < len(self.graph.nodes()):
            for weight, u, v in edges:
                if u in visited and v not in visited:
                    mst_edges.append((u, v))
                    visited.add(v)
                    break
        self.steps = mst_edges
        self.animate()

    def kruskal(self):
        mst_edges = []
        parent = {node: node for node in self.graph.nodes()}

        def find(node):
            if parent[node] != node:
                parent[node] = find(parent[node])
            return parent[node]

        edges = sorted((self.graph[u][v]['weight'], u, v) for u, v in self.graph.edges())
        for weight, u, v in edges:
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                mst_edges.append((u, v))
                parent[root_u] = root_v
        self.steps = mst_edges
        self.animate()

    def animate(self):
        self.mst_edges = []
        self.ani = FuncAnimation(self.fig, self.update, frames=len(self.steps), interval=2000, repeat=False)
        plt.show()

    def update(self, i):
        edge = self.steps[i]
        self.mst_edges.append(edge)
        self.ax.clear()
        nx.draw(self.graph, self.pos, with_labels=True, node_color='lightblue', edge_color='gray', ax=self.ax)
        nx.draw_networkx_edges(self.graph, self.pos, edgelist=self.mst_edges, edge_color='red', width=2, ax=self.ax)
        self.ax.set_title(f"Шаг {i + 1}: Добавлено ребро {edge}")


def add_edge_ui(graph):
    root = tk.Tk()
    root.withdraw()
    while True:
        u = simpledialog.askinteger("Добавление ребра", "Введите первую вершину (или -1 для выхода):")
        if u == -1:
            break
        v = simpledialog.askinteger("Добавление ребра", "Введите вторую вершину:")
        weight = simpledialog.askinteger("Добавление ребра", "Введите вес ребра:")
        graph.add_edge(u, v, weight)


def add_adjacency_matrix_ui(graph):
    root = tk.Tk()
    root.withdraw()
    size = simpledialog.askinteger("Матрица смежности", "Введите размер матрицы:")
    if not size or size <= 0:
        messagebox.showerror("Ошибка", "Неверный размер матрицы")
        return

    matrix = []
    for i in range(size):
        row = simpledialog.askstring("Матрица смежности", f"Введите {i + 1}-ю строку (через пробел):")
        matrix.append(list(map(int, row.split())))

    graph.from_adjacency_matrix(matrix)


def select_algorithm(visualizer):
    root = tk.Tk()
    root.withdraw()
    algo = simpledialog.askstring("Выбор алгоритма", "Выберите алгоритм: prim или kruskal")
    if algo == "prim":
        visualizer.prim()
    elif algo == "kruskal":
        visualizer.kruskal()
    else:
        print("Неверный выбор. Запуск по умолчанию: Прим")
        visualizer.prim()


def select_input_method():
    root = tk.Tk()
    root.withdraw()
    method = simpledialog.askstring("Выбор метода ввода",
                                    "Выберите метод ввода: edges (ребра) или adjacency (матрица смежности)")
    return method


# Создание графа вручную через UI
g = Graph()
input_method = select_input_method()
if input_method == "edges":
    add_edge_ui(g)
elif input_method == "adjacency":
    add_adjacency_matrix_ui(g)
else:
    print("Неверный выбор, используем ввод через ребра.")
    add_edge_ui(g)

visualizer = MSTVisualizer(g)
select_algorithm(visualizer)
