"""
Модуль, содержащий реализации алгоритмов обхода графа.
"""

import networkx as nx
from PySide6.QtWidgets import QMessageBox, QInputDialog

class GraphAlgorithm:
    """Базовый класс для алгоритмов обхода графа"""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.visited = set()
        self.result = []
        self.current = None
        self.path = []

    def reset(self):
        """Сбрасывает состояние алгоритма"""
        self.visited.clear()
        self.result.clear()
        self.current = None
        self.path.clear()
        self.main_window.graph_widget.bfs_path.clear()
        self.main_window.graph_widget.bfs_current_edge = None
        self.main_window.graph_widget.visited_vertices.clear()
        self.main_window.graph_widget.bfs_current = None

class BFSAlgorithm(GraphAlgorithm):
    """Реализация алгоритма поиска в ширину (BFS)"""
    
    def __init__(self, main_window):
        super().__init__(main_window)
        self.queue = []
        self.parent = {}
        self.current_vertex = None
        self.current_neighbor = None
        self.step = 0

    def start(self, start_vertex):
        """Запускает алгоритм BFS"""
        self.reset()
        self.queue = [start_vertex]
        self.visited = {start_vertex}
        self.parent = {}
        self.current_vertex = None
        self.current_neighbor = None
        self.step = 0
        message = "Начинаем обход графа в ширину..."
        self.main_window.explanation_widget.clear()
        self.main_window.explanation_widget.append(message)
        self.main_window.highlight_pseudocode_line(0)  # Инициализация
        self.main_window.graph_widget.visited_vertices = self.visited
        self.main_window.graph_widget.bfs_current = start_vertex
        self.main_window.graph_widget.update()
        return message

    def next_step(self):
        """Выполняет следующий шаг алгоритма"""
        if not self.queue:
            message = "Обход завершен!"
            self.main_window.explanation_widget.append(message)
            self.main_window.highlight_pseudocode_line(16)  # Завершение
            self.main_window.graph_widget.bfs_current = None
            self.main_window.graph_widget.bfs_current_edge = None
            self.main_window.graph_widget.update()
            return True, message, self._get_state()

        self.current_vertex = self.queue.pop(0)
        self.result.append(self.current_vertex)
        self.visited.add(self.current_vertex)
        self.main_window.graph_widget.bfs_current = self.current_vertex
        message = f"Обрабатываем вершину {self.current_vertex}"
        self.main_window.explanation_widget.append(message)
        self.main_window.highlight_pseudocode_line(7)  # Основной цикл: обработка вершины
        
        # Получаем соседей текущей вершины
        neighbors = list(self.main_window.graph_widget.graph.neighbors(self.current_vertex))
        
        # Обрабатываем каждого соседа
        for neighbor in neighbors:
            self.main_window.graph_widget.bfs_current_edge = (self.current_vertex, neighbor)
            self.main_window.graph_widget.update()
            if neighbor not in self.visited:
                message = f"Найден непосещенный сосед: {neighbor}"
                self.main_window.explanation_widget.append(message)
                self.main_window.highlight_pseudocode_line(12)  # Обработка соседа
                self.visited.add(neighbor)
                self.queue.append(neighbor)
                self.parent[neighbor] = self.current_vertex
                self.path.append((self.current_vertex, neighbor))
                self.main_window.graph_widget.visited_vertices = self.visited
                self.main_window.graph_widget.bfs_path = self.path
            else:
                message = f"Сосед {neighbor} уже был посещен"
                self.main_window.explanation_widget.append(message)
        self.main_window.graph_widget.bfs_current_edge = None
        self.main_window.graph_widget.update()
        return False, message, self._get_state()

    def _get_state(self):
        """Возвращает текущее состояние алгоритма (только ключевые переменные)"""
        return {
            'visited': self.visited,
            'queue': self.queue,
            'parent': self.parent,
            'current_vertex': self.current_vertex
        }

class DFSAlgorithm(GraphAlgorithm):
    """Реализация алгоритма поиска в глубину (DFS)"""
    
    def __init__(self, main_window):
        super().__init__(main_window)
        self.stack = []
        self.current = None

    def start(self, start_vertex):
        """Начинает обход с указанной вершины"""
        self.reset()
        self.stack = [start_vertex]
        message = "Начинаем обход графа в глубину..."
        self.main_window.explanation_widget.clear()
        self.main_window.explanation_widget.append(message)
        self.main_window.highlight_pseudocode_line(0)  # Инициализация
        self.main_window.graph_widget.bfs_current = start_vertex
        self.main_window.graph_widget.update()
        return message

    def next_step(self):
        """Выполняет следующий шаг алгоритма DFS"""
        if not self.stack:
            message = "Обход завершен!"
            self.main_window.explanation_widget.append(message)
            self.main_window.highlight_pseudocode_line(22)  # Завершение
            self.main_window.graph_widget.bfs_current = None
            self.main_window.graph_widget.bfs_current_edge = None
            self.main_window.graph_widget.update()
            return True, message, self._get_state()

        # Извлекаем вершину из стека
        vertex = self.stack.pop()
        self.current = vertex
        self.main_window.graph_widget.bfs_current = vertex
        if vertex not in self.visited:
            self.visited.add(vertex)
            self.result.append(vertex)
            self.main_window.graph_widget.visited_vertices = self.visited
            message = f"Обрабатываем вершину {vertex}"
            self.main_window.explanation_widget.append(message)
            self.main_window.highlight_pseudocode_line(8)  # Основной цикл: обработка вершины
            # Получаем соседей в обратном отсортированном порядке
            neighbors = sorted(list(self.main_window.graph_widget.graph.neighbors(vertex)), reverse=True)
            # Обрабатываем каждого соседа
            for neighbor in neighbors:
                self.main_window.graph_widget.bfs_current_edge = (vertex, neighbor)
                self.main_window.graph_widget.update()
                if neighbor not in self.visited:
                    message = f"Найден непосещенный сосед: {neighbor}"
                    self.main_window.explanation_widget.append(message)
                    self.main_window.highlight_pseudocode_line(15)  # Обработка соседа
                    self.stack.append(neighbor)
                    self.path.append((vertex, neighbor))
                    self.main_window.graph_widget.bfs_path = self.path
                else:
                    message = f"Сосед {neighbor} уже был посещен"
                    self.main_window.explanation_widget.append(message)
        else:
            message = f"Вершина {vertex} уже была посещена"
            self.main_window.explanation_widget.append(message)
        self.main_window.graph_widget.bfs_current_edge = None
        self.main_window.graph_widget.update()
        return False, message, self._get_state()

    def _get_state(self):
        """Возвращает текущее состояние алгоритма (только ключевые переменные)"""
        return {
            'visited': self.visited,
            'stack': self.stack,
            'parent': getattr(self, 'parent', {}),
            'current': self.current
        }

class DijkstraAlgorithm(GraphAlgorithm):
    """Реализация алгоритма поиска кратчайшего пути (Дейкстра)"""
    
    def __init__(self, main_window):
        super().__init__(main_window)
        self.distances = {}  # расстояния до вершин
        self.previous = {}   # предыдущие вершины для восстановления пути
        self.unvisited = set()  # непосещенные вершины
        self.current_vertex = None
        self.end_vertex = None  # конечная вершина
        self.shortest_path = []  # кратчайший путь
        self.waiting_for_end = False  # флаг ожидания выбора конечной вершины
        self.current_neighbors = []  # текущие необработанные соседи
        self.processing_vertex = False  # флаг обработки текущей вершины
        self.current_neighbor = None  # текущий обрабатываемый сосед
        self.comparison_step = False  # флаг шага сравнения
        self.current_distance = None  # текущее вычисленное расстояние
        self.comparison_text = {}  # текст сравнения для отображения над вершинами
        
    def reset(self):
        """Сбрасывает состояние алгоритма"""
        super().reset()
        self.comparison_text = {}
        self.main_window.graph_widget.distances = {}
        self.main_window.graph_widget.comparison_text = {}

    def start(self, start_vertex):
        """Начинает поиск кратчайшего пути из указанной вершины"""
        self.reset()
        
        # Инициализация расстояний и множества непосещенных вершин
        for vertex in self.main_window.graph_widget.graph.nodes():
            self.distances[vertex] = float('inf')
            self.previous[vertex] = None
            self.unvisited.add(vertex)
            
        # Устанавливаем расстояние до начальной вершины
        self.distances[start_vertex] = 0
        self.current_vertex = start_vertex
        self.waiting_for_end = True
        
        # Обновляем отображение расстояний и текущей вершины
        self.main_window.graph_widget.distances = self.distances
        self.main_window.graph_widget.bfs_current = start_vertex
        self.main_window.graph_widget.waiting_for_vertex_selection = True
        
        message = "Выберите конечную вершину"
        self.main_window.explanation_widget.clear()
        self.main_window.explanation_widget.append(message)
        self.main_window.highlight_pseudocode_line(0)
        self.main_window.graph_widget.update()
        return message

    def set_end_vertex(self, end_vertex):
        """Устанавливает конечную вершину и начинает поиск пути"""
        self.end_vertex = end_vertex
        self.waiting_for_end = False
        self.main_window.graph_widget.waiting_for_vertex_selection = False
        message = f"Начинаем поиск кратчайшего пути от вершины {self.current_vertex} до вершины {end_vertex}"
        self.main_window.explanation_widget.append(message)
        self.main_window.graph_widget.update()
        return message

    def next_step(self):
        """Выполняет следующий шаг алгоритма"""
        if self.waiting_for_end:
            self.main_window.highlight_pseudocode_line(0)  # Инициализация
            return False, "Выберите конечную вершину", self._get_state()

        # Если есть текущий сосед и мы на шаге сравнения
        if self.current_neighbor is not None and self.comparison_step:
            self.main_window.highlight_pseudocode_line(17)  # Сравнение расстояний (было 15)
            return self._process_comparison()

        # Если есть необработанные соседи текущей вершины
        if self.current_neighbors:
            self.main_window.highlight_pseudocode_line(15)  # Обновление расстояний до соседей (было 12)
            return self._prepare_next_neighbor()

        # Сбрасываем подсветку ребра перед переходом к новой вершине
        self.main_window.graph_widget.bfs_current_edge = None
        self.main_window.graph_widget.update()

        # Если все соседи обработаны или это первый шаг
        if not self.unvisited:
            self.main_window.highlight_pseudocode_line(21)  # Восстановление пути (было 23)
            return self._finish_algorithm()

        # Находим следующую вершину для обработки
        min_vertex = self._find_min_distance_vertex()
        if min_vertex is None:
            self.main_window.highlight_pseudocode_line(11)  # Нет пути до оставшихся вершин (было 8)
            return True, "Не удалось найти путь до всех вершин", self._get_state()

        # Начинаем обработку новой вершины
        self.current_vertex = min_vertex
        self.unvisited.remove(min_vertex)
        self.visited.add(min_vertex)
        self.main_window.graph_widget.visited_vertices = self.visited
        self.main_window.graph_widget.bfs_current = min_vertex
        
        # Получаем всех соседей в отсортированном порядке
        self.current_neighbors = sorted(list(self.main_window.graph_widget.graph.neighbors(min_vertex)))
        
        message = f"Обрабатываем вершину {min_vertex} (расстояние: {self.distances[min_vertex] if self.distances[min_vertex] != float('inf') else '∞'})"
        self.main_window.explanation_widget.append(message)
        self.main_window.highlight_pseudocode_line(12)  # Помечаем вершину как посещённую (было 5)
        self.main_window.graph_widget.update()
        
        return False, message, self._get_state()

    def _prepare_next_neighbor(self):
        """Подготавливает обработку следующего соседа"""
        self.current_neighbor = self.current_neighbors.pop(0)
        self.main_window.highlight_pseudocode_line(15)  # Обновление расстояний до соседей (было 12)
        # Устанавливаем текущее ребро для подсветки
        self.main_window.graph_widget.bfs_current_edge = (self.current_vertex, self.current_neighbor)
        edge_data = self.main_window.graph_widget.graph.get_edge_data(self.current_vertex, self.current_neighbor)
        weight = edge_data.get('weight', 1)
        self.current_distance = self.distances[self.current_vertex] + weight
        message = [
            f"Рассматриваем путь до вершины {self.current_neighbor} через {self.current_vertex}:",
            f"Текущее расстояние до {self.current_neighbor}: {self.distances[self.current_neighbor] if self.distances[self.current_neighbor] != float('inf') else '∞'}",
            f"Новое расстояние через {self.current_vertex}: {self.current_distance}"
        ]
        self.comparison_step = True
        self.main_window.explanation_widget.append("\n".join(message))
        self.main_window.graph_widget.update()
        return False, "\n".join(message)

    def _process_comparison(self):
        """Обрабатывает шаг сравнения расстояний"""
        # Подсвечиваем строку сравнения
        self.main_window.highlight_pseudocode_line(17)
        current_dist_str = str(self.current_distance) if self.current_distance != float('inf') else '∞'
        neighbor_dist_str = str(self.distances[self.current_neighbor]) if self.distances[self.current_neighbor] != float('inf') else '∞'
        comparison = f"{self.distances[self.current_vertex]}+{self.main_window.graph_widget.graph[self.current_vertex][self.current_neighbor].get('weight', 1)}<{neighbor_dist_str}"
        self.comparison_text[self.current_neighbor] = comparison
        self.main_window.graph_widget.comparison_text = self.comparison_text
        message = [
            f"Сравниваем: {current_dist_str} < {neighbor_dist_str}",
        ]
        if self.current_distance < self.distances[self.current_neighbor]:
            old_distance = self.distances[self.current_neighbor]
            self.distances[self.current_neighbor] = self.current_distance
            self.previous[self.current_neighbor] = self.current_vertex
            message.append(f"Обновляем расстояние: {old_distance if old_distance != float('inf') else '∞'} → {self.current_distance}")
            self.main_window.graph_widget.distances = self.distances
            # Подсвечиваем строку обновления
            self.main_window.highlight_pseudocode_line(18)
        else:
            message.append(f"Оставляем текущее расстояние: {self.distances[self.current_neighbor] if self.distances[self.current_neighbor] != float('inf') else '∞'}")
            # Подсвечиваем строку 'иначе'
            self.main_window.highlight_pseudocode_line(20)
        self.main_window.explanation_widget.append("\n".join(message))
        self.comparison_text = {}
        self.main_window.graph_widget.comparison_text = {}
        self.comparison_step = False
        self.current_neighbor = None
        self.current_distance = None
        self.main_window.graph_widget.update()
        return False, "\n".join(message)

    def _find_min_distance_vertex(self):
        """Находит вершину с минимальным расстоянием среди непосещенных"""
        min_distance = float('inf')
        min_vertex = None
        for vertex in self.unvisited:
            if self.distances[vertex] < min_distance:
                min_distance = self.distances[vertex]
                min_vertex = vertex
        return min_vertex

    def _finish_algorithm(self):
        self.main_window.highlight_pseudocode_line(22)  # Восстановление пути: "Если previous[end] не null:"
        if self.end_vertex is not None:
            self.shortest_path = self._reconstruct_path()
            path_edges = [(self.shortest_path[i], self.shortest_path[i+1]) 
                        for i in range(len(self.shortest_path)-1)]
            self.main_window.graph_widget.bfs_path = path_edges
            distance = self.distances[self.end_vertex]
            if distance == float('inf'):
                message = f"Путь до вершины {self.end_vertex} не найден"
            else:
                path_str = " → ".join(map(str, self.shortest_path))
                message = f"Найден кратчайший путь длиной {distance}:\n{path_str}"
            self.main_window.graph_widget.bfs_current = None
            self.main_window.graph_widget.bfs_current_edge = None
            self.main_window.graph_widget.update()
            return True, message, self._get_state()
        return True, "Поиск завершен!", self._get_state()

    def _reconstruct_path(self):
        """Восстанавливает кратчайший путь от начальной до конечной вершины"""
        path = []
        current = self.end_vertex
        while current is not None:
            path.append(current)
            current = self.previous[current]
        return list(reversed(path))

    def _get_state(self):
        """Возвращает текущее состояние алгоритма"""
        return {
            'distances': self.distances,
            'previous': self.previous,
            'unvisited': self.unvisited,
            'visited': self.visited,
            'current_vertex': self.current_vertex,
            'end_vertex': self.end_vertex,
            'shortest_path': self.shortest_path,
            'current_neighbors': self.current_neighbors,
            'current_neighbor': self.current_neighbor,
            'comparison_step': self.comparison_step,
            'current_distance': self.current_distance,
            'path': self.shortest_path
        } 