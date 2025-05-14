"""
Модуль с утилитами для работы с графами.
Содержит функции для загрузки, сохранения и создания графов из различных форматов.
"""

import networkx as nx
import numpy as np
import random

def is_weighted(graph):
    """Проверяет, является ли граф взвешенным"""
    return any('weight' in graph[u][v] for u, v in graph.edges())

def parse_matrix(text):
    """
    Преобразует текстовое представление матрицы в numpy массив.
    
    Пример входных данных:
    1 0 1
    0 1 0
    1 0 1
    """
    try:
        # Разбиваем текст на строки и удаляем пустые
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Преобразуем каждую строку в список чисел
        matrix = []
        for line in lines:
            row = [float(x) for x in line.split()]
            matrix.append(row)
            
        # Проверяем, что все строки имеют одинаковую длину
        if not all(len(row) == len(matrix[0]) for row in matrix):
            raise ValueError("Все строки матрицы должны иметь одинаковую длину")
            
        return np.array(matrix)
    except ValueError as e:
        raise ValueError(f"Ошибка при разборе матрицы: {str(e)}")

def create_graph_from_adjacency_matrix(matrix, is_directed=False, is_weighted=False):
    """
    Создает граф из матрицы смежности.
    
    Args:
        matrix: numpy массив с матрицей смежности
        is_directed: флаг, указывающий является ли граф ориентированным
        is_weighted: флаг, указывающий является ли граф взвешенным
    """
    # Проверяем, что матрица квадратная
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Матрица смежности должна быть квадратной")
    
    # Создаем граф нужного типа
    graph = nx.DiGraph() if is_directed else nx.Graph()
    
    # Добавляем вершины
    for i in range(matrix.shape[0]):
        graph.add_node(i)
    
    # Добавляем рёбра
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] != 0:
                if is_weighted:
                    graph.add_edge(i, j, weight=matrix[i][j])
                else:
                    graph.add_edge(i, j)
    
    return graph

def create_graph_from_incidence_matrix(matrix, is_directed=False, is_weighted=False):
    """
    Создает граф из матрицы инцидентности.
    
    Args:
        matrix: numpy массив с матрицей инцидентности
        is_directed: флаг, указывающий является ли граф ориентированным
        is_weighted: флаг, указывающий является ли граф взвешенным
    """
    # Создаем граф нужного типа
    graph = nx.DiGraph() if is_directed else nx.Graph()
    
    # Добавляем вершины
    for i in range(matrix.shape[0]):
        graph.add_node(i)
    
    # Для каждого столбца (ребра) находим связанные вершины
    for j in range(matrix.shape[1]):
        if is_directed:
            # Для ориентированного графа ищем -1 (начало) и 1 (конец)
            start = np.where(matrix[:, j] == -1)[0]
            end = np.where(matrix[:, j] == 1)[0]
            if len(start) == 1 and len(end) == 1:
                if is_weighted:
                    # Вес ребра берем из значений матрицы
                    weight = abs(matrix[start[0]][j])
                    graph.add_edge(start[0], end[0], weight=weight)
                else:
                    graph.add_edge(start[0], end[0])
        else:
            # Для неориентированного графа ищем ненулевые элементы
            vertices = np.where(matrix[:, j] != 0)[0]
            if len(vertices) == 2:
                if is_weighted:
                    # Вес ребра берем из значений матрицы
                    weight = abs(matrix[vertices[0]][j])
                    graph.add_edge(vertices[0], vertices[1], weight=weight)
                else:
                    graph.add_edge(vertices[0], vertices[1])
    
    return graph

def load_graph_from_file(filename):
    """
    Загружает граф из файла.
    
    Поддерживаемые форматы:
    1. Список рёбер:
        directed/undirected weighted/unweighted
        vertex1 vertex2 [weight]
        vertex3 vertex4 [weight]
        ...
    
    2. Матрица смежности:
        adjacency directed/undirected weighted/unweighted
        0 1 0
        1 0 1
        0 1 0
        
    3. Матрица инцидентности:
        incidence directed/undirected weighted/unweighted
        1 0 1
        1 1 0
        0 1 1
    """
    with open(filename, 'r') as f:
        # Читаем первую строку с параметрами
        params = f.readline().strip().split()
        if len(params) < 2:
            raise ValueError("Первая строка должна содержать как минимум два параметра")
            
        # Определяем формат и тип графа
        format_type = params[0].lower() if len(params) >= 3 else 'edges'
        is_directed = params[-2].lower() == 'directed'
        is_weighted = params[-1].lower() == 'weighted'
        
        # Читаем оставшиеся строки
        matrix_text = ''.join(f.readlines())
        
        if format_type == 'adjacency':
            # Загружаем из матрицы смежности
            matrix = parse_matrix(matrix_text)
            return create_graph_from_adjacency_matrix(matrix, is_directed, is_weighted)
        elif format_type == 'incidence':
            # Загружаем из матрицы инцидентности
            matrix = parse_matrix(matrix_text)
            return create_graph_from_incidence_matrix(matrix, is_directed, is_weighted)
        else:
            # Загружаем из списка рёбер
            graph = nx.DiGraph() if is_directed else nx.Graph()
            for line in matrix_text.split('\n'):
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                    
                v1, v2 = int(parts[0]), int(parts[1])
                if is_weighted:
                    if len(parts) != 3:
                        raise ValueError(f"Для взвешенного графа необходимо указать вес ребра: {line}")
                    weight = float(parts[2])
                    graph.add_edge(v1, v2, weight=weight)
                else:
                    graph.add_edge(v1, v2)
            return graph

def save_graph_to_file(graph, filename):
    """
    Сохраняет граф в файл.
    
    Формат файла аналогичен формату загрузки:
    directed/undirected weighted/unweighted
    vertex1 vertex2 [weight]
    vertex3 vertex4 [weight]
    ...
    """
    with open(filename, 'w') as f:
        # Записываем параметры графа
        graph_type = 'directed' if isinstance(graph, nx.DiGraph) else 'undirected'
        weight_type = 'weighted' if is_weighted(graph) else 'unweighted'
        f.write(f"{graph_type} {weight_type}\n")
        
        # Записываем рёбра
        for edge in graph.edges(data=True):
            if 'weight' in edge[2]:
                f.write(f"{edge[0]} {edge[1]} {edge[2]['weight']}\n")
            else:
                f.write(f"{edge[0]} {edge[1]}\n")

def generate_random_graph(num_vertices, num_edges, is_directed=False, is_weighted=False):
    """
    Генерирует случайный граф с заданными параметрами.
    
    Args:
        num_vertices (int): Количество вершин
        num_edges (int): Количество рёбер
        is_directed (bool): Флаг ориентированного графа
        is_weighted (bool): Флаг взвешенного графа
        
    Returns:
        nx.Graph или nx.DiGraph: Сгенерированный граф
    """
    if is_directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()
        
    # Добавляем вершины
    for i in range(num_vertices):
        graph.add_node(i)
        
    # Генерируем все возможные рёбра
    possible_edges = []
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            possible_edges.append((i, j))
            
    # Перемешиваем рёбра
    random.shuffle(possible_edges)
    
    # Добавляем рёбра
    for i, j in possible_edges[:num_edges]:
        if is_weighted:
            # Генерируем случайный целый вес от 1 до 10
            weight = random.randint(1, 10)
            graph.add_edge(i, j, weight=weight)
        else:
            graph.add_edge(i, j)
            
    return graph

def save_random_graph(filename, num_vertices, num_edges, is_directed=False, is_weighted=False, min_weight=1, max_weight=10):
    """
    Генерирует случайный граф и сохраняет его в файл.
    
    Args:
        filename: имя файла для сохранения
        num_vertices: количество вершин
        num_edges: количество рёбер
        is_directed: флаг, указывающий является ли граф ориентированным
        is_weighted: флаг, указывающий является ли граф взвешенным
        min_weight: минимальный вес ребра
        max_weight: максимальный вес ребра
    """
    graph = generate_random_graph(num_vertices, num_edges, is_directed, is_weighted)
    save_graph_to_file(graph, filename) 