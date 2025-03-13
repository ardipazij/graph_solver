import numpy as np
import networkx as nx

def parse_matrix(matrix_text):
    """Парсит матрицу из текстового представления"""
    try:
        rows = matrix_text.strip().split('\n')
        matrix = []
        for row in rows:
            matrix.append([float(x) for x in row.strip().split()])
        return np.array(matrix)
    except Exception as e:
        raise ValueError(f"Ошибка при парсинге матрицы: {str(e)}")

def create_graph_from_adjacency_matrix(matrix, directed=False, weighted=False):
    """Создает граф из матрицы смежности"""
    n = len(matrix)
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    for i in range(n):
        G.add_node(i)
        for j in range(n):
            if matrix[i][j] != 0:
                if weighted:
                    G.add_edge(i, j, weight=matrix[i][j])
                else:
                    G.add_edge(i, j)
    
    return G

def create_graph_from_incidence_matrix(matrix, directed=False, weighted=False):
    """Создает граф из матрицы инцидентности"""
    n_vertices, n_edges = matrix.shape
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    for i in range(n_vertices):
        G.add_node(i)
    
    for j in range(n_edges):
        edge_vertices = []
        for i in range(n_vertices):
            if matrix[i][j] != 0:
                edge_vertices.append(i)
        
        if len(edge_vertices) == 2:
            if weighted:
                G.add_edge(edge_vertices[0], edge_vertices[1], weight=abs(matrix[edge_vertices[0]][j]))
            else:
                G.add_edge(edge_vertices[0], edge_vertices[1])
    
    return G

def load_graph_from_file(file_path):
    """Загружает граф из текстового файла"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Удаляем пустые строки и комментарии
    lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    
    # Определяем формат файла
    if len(lines) == 0:
        raise ValueError("Файл пуст")
    
    # Проверяем, является ли первая строка числом (старый формат)
    try:
        n = int(lines[0])
        # Старый формат: первая строка - количество вершин, затем матрица смежности
        matrix_lines = lines[1:]
        matrix = []
        for line in matrix_lines:
            matrix.append([float(x) for x in line.split()])
        matrix = np.array(matrix)
        
        # Определяем тип графа по матрице
        directed = not np.allclose(matrix, matrix.T)
        weighted = np.any(matrix > 1)
        
        return create_graph_from_adjacency_matrix(matrix, directed, weighted)
    except ValueError:
        # Новый формат: первая строка - тип графа, затем матрица
        graph_type = lines[0].strip().lower()
        directed = 'directed' in graph_type
        weighted = 'weighted' in graph_type
        
        matrix_lines = lines[1:]
        matrix_text = '\n'.join(matrix_lines)
        matrix = parse_matrix(matrix_text)
        
        # Определяем тип матрицы по её размерности
        if matrix.shape[0] == matrix.shape[1]:
            return create_graph_from_adjacency_matrix(matrix, directed, weighted)
        else:
            return create_graph_from_incidence_matrix(matrix, directed, weighted)

def save_graph_to_file(graph, file_path):
    """Сохраняет граф в текстовый файл"""
    with open(file_path, 'w') as f:
        # Записываем тип графа
        graph_type = []
        if isinstance(graph, nx.DiGraph):
            graph_type.append('directed')
        if is_weighted(graph):
            graph_type.append('weighted')
        f.write(' '.join(graph_type) + '\n')
        
        # Записываем матрицу смежности
        matrix = nx.adjacency_matrix(graph).todense()
        for row in matrix:
            f.write(' '.join(map(str, row)) + '\n')

def is_weighted(graph):
    """Проверяет, является ли граф взвешенным"""
    return any('weight' in graph[u][v] for u, v in graph.edges()) 