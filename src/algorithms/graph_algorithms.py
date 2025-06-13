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

    def get_pseudocode(self):
        """Возвращает псевдокод алгоритма (список строк)"""
        raise NotImplementedError

    def get_highlight_map(self):
        """Возвращает словарь: этап -> номер строки псевдокода"""
        raise NotImplementedError

    def get_name(self):
        """Возвращает название алгоритма"""
        return self.__class__.__name__

    def get_description(self):
        """Возвращает краткое описание алгоритма"""
        return ""

    def start(self):
        pass
    def next_step(self):
        pass
    def _get_state(self):
        pass

class BFSAlgorithm(GraphAlgorithm):
    """Пошаговый BFS: каждый сосед обрабатывается отдельно с подсветкой и выводом информации"""
    def __init__(self, main_window):
        super().__init__(main_window)
        self.queue = []
        self.parent = {}
        self.result = []
        self.current_vertex = None
        self._bfs_substep = 0
        self._bfs_neighbors = []
        self._bfs_neighbor_idx = 0
        self.path_edges = []

    def reset(self):
        super().reset()
        self.queue = []
        self.parent = {}
        self.result = []
        self.current_vertex = None
        self._bfs_substep = 0
        self._bfs_neighbors = []
        self._bfs_neighbor_idx = 0
        self.path_edges = []
        self.main_window.graph_widget.visited_vertices = set()
        self.main_window.graph_widget.bfs_current_edge = None
        self.main_window.graph_widget.bfs_path = []

    def get_pseudocode(self):
        return [
            "1. Инициализация:",
            "   visited = ∅        // множество посещенных вершин",
            "   queue = [start]    // очередь вершин для обработки",
            "   result = []        // результат обхода",
            "   parent = {}        // словарь для хранения родительских вершин",
            "",
            "2. Основной цикл:",
            "   Пока queue не пуста:",
            "       vertex = queue.pop(0)    // берем первую вершину из очереди",
            "       result.append(vertex)    // добавляем в результат",
            "       visited.add(vertex)      // помечаем как посещенную",
            "",
            "       // Обработка соседей:",
            "       Для каждого соседа neighbor вершины vertex:",
            "           Если neighbor не посещен:",
            "               visited.add(neighbor)    // помечаем как посещенного",
            "               queue.append(neighbor)    // добавляем в очередь",
            "               parent[neighbor] = vertex // запоминаем родителя",
            "",
            "3. Завершение:",
            "   Возвращаем result, parent"
        ]

    def start(self, start_vertex):
        self.reset()
        self.queue = [start_vertex]
        self.visited = {start_vertex}
        self.parent = {}
        self.result = []
        self.current_vertex = None
        self.main_window.graph_widget.visited_vertices = set(self.visited)
        self.main_window.graph_widget.bfs_current = start_vertex
        self.main_window.graph_widget.bfs_current_edge = None
        self.main_window.graph_widget.update()
        self._bfs_substep = 0
        return False, f"Начинаем обход графа в ширину с вершины {start_vertex}", self._get_state(), 'init'

    def next_step(self):
        # Если очередь пуста — завершить обход
        if not self.queue and self._bfs_substep == 0:
            # Подсветить BFS-дерево (пути parent)
            path_edges = [(v, p) for v, p in self.parent.items()]
            self.main_window.graph_widget.bfs_path = path_edges
            self.main_window.graph_widget.bfs_current = None
            self.main_window.graph_widget.bfs_current_edge = None
            self.main_window.graph_widget.update()
            return True, "Обход завершен!", self._get_state(), 'finish'

        # Подшаг 0: взять вершину из очереди
        if self._bfs_substep == 0:
            self.current_vertex = self.queue.pop(0)
            self.result.append(self.current_vertex)
            self.visited.add(self.current_vertex)
            self.main_window.graph_widget.visited_vertices = set(self.visited)
            self.main_window.graph_widget.bfs_current = self.current_vertex
            self._bfs_neighbors = list(self.main_window.graph_widget.graph.neighbors(self.current_vertex))
            self._bfs_neighbor_idx = 0
            self.main_window.graph_widget.bfs_current_edge = None
            self.main_window.graph_widget.update()
            self._bfs_substep = 1
            return False, f"Взяли вершину {self.current_vertex} из очереди", self._get_state(), 'pop_vertex'

        # Подшаг 1: обработка соседей по одному
        if self._bfs_substep == 1:
            if self._bfs_neighbor_idx < len(self._bfs_neighbors):
                neighbor = self._bfs_neighbors[self._bfs_neighbor_idx]
                self.main_window.graph_widget.bfs_current_edge = (self.current_vertex, neighbor)
                self.main_window.graph_widget.update()
                if neighbor not in self.visited:
                    self.visited.add(neighbor)
                    self.queue.append(neighbor)
                    self.parent[neighbor] = self.current_vertex
                    self.path_edges.append((self.current_vertex, neighbor))
                    msg = f"Добавили {neighbor} в очередь и пометили как посещённого (родитель: {self.current_vertex})"
                else:
                    msg = f"Сосед {neighbor} уже был посещён"
                self._bfs_neighbor_idx += 1
                return False, msg, self._get_state(), 'neighbor_check'
            else:
                self.main_window.graph_widget.bfs_current_edge = None
                self.main_window.graph_widget.update()
                self._bfs_substep = 0
                return False, f"Завершили обработку всех соседей вершины {self.current_vertex}", self._get_state(), 'neighbor_loop'

    def _get_state(self):
        return {
            'visited': self.visited,
            'queue': self.queue,
            'parent': self.parent,
            'result': self.result,
            'current_vertex': self.current_vertex,
            'current_neighbor': self._bfs_neighbors[self._bfs_neighbor_idx-1] if self._bfs_neighbor_idx > 0 and self._bfs_neighbor_idx <= len(self._bfs_neighbors) else None
        }

    def get_highlight_map(self):
        return {
            'init': 0,
            'pop_vertex': 7,
            'neighbor_check': 13,
            'neighbor_loop': 17,
            'finish': 19
        }

class DFSAlgorithm(GraphAlgorithm):
    """Реализация алгоритма поиска в глубину (DFS) с подробной визуализацией"""
    def __init__(self, main_window):
        super().__init__(main_window)
        self.stack = []
        self.current = None
        self.current_path = []
        self.paths_checked = 0
        self.parent = {}
        self.last_backtrack = False
        self.visited = set()
        self.path = []  # для хранения рёбер обхода
        self._dfs_neighbor_idx = {}  # для хранения индекса соседа для каждой вершины

    def reset(self):
        super().reset()
        self.stack = []
        self.current = None
        self.current_path = []
        self.paths_checked = 0
        self.parent = {}
        self.last_backtrack = False
        self.visited = set()
        self.path = []
        self._dfs_neighbor_idx = {}  # для хранения индекса соседа для каждой вершины
        self.main_window.graph_widget.bfs_path = []
        self.main_window.graph_widget.bfs_current = None
        self.main_window.graph_widget.bfs_current_edge = None
        self.main_window.graph_widget.visited_vertices = set()
        self.main_window.graph_widget.update()

    def start(self, start_vertex):
        self.reset()
        self.stack = [(start_vertex, [start_vertex])]  # (vertex, path)
        self.current = start_vertex
        self.current_path = [start_vertex]
        self.main_window.graph_widget.bfs_current = start_vertex
        self.main_window.graph_widget.update()
        self.main_window.explanation_widget.clear()
        self.main_window.explanation_widget.append(f"Начинаем обход графа в глубину с вершины {start_vertex}")
        return False, f"Начинаем обход графа в глубину с вершины {start_vertex}", self._get_state(), 'init'

    def next_step(self):
        if not self.stack:
            self.main_window.graph_widget.bfs_current = None
            self.main_window.graph_widget.bfs_current_edge = None
            self.main_window.graph_widget.bfs_path = self.path
            self.main_window.graph_widget.update()
            explanation = "Обход завершён!"
            return True, explanation, self._get_state(), 'finish'

        if self.last_backtrack:
            self.main_window.graph_widget.bfs_current_edge = None
            self.main_window.graph_widget.update()
            self.last_backtrack = False

        vertex, path = self.stack[-1]
        self.current = vertex
        self.current_path = path
        self.main_window.graph_widget.bfs_current = vertex
        self.main_window.graph_widget.bfs_path = [(path[i], path[i+1]) for i in range(len(path)-1)]

        if vertex not in self.visited:
            self.visited.add(vertex)
            self.main_window.graph_widget.visited_vertices = set(self.visited)
            explanation = f"Обрабатываем вершину {vertex}"
        else:
            explanation = None

        neighbors = list(self.main_window.graph_widget.graph.neighbors(vertex))
        idx = self._dfs_neighbor_idx.get(vertex, 0)
        while idx < len(neighbors):
            neighbor = neighbors[idx]
            self._dfs_neighbor_idx[vertex] = idx + 1
            if neighbor not in self.visited:
                self.main_window.graph_widget.bfs_current_edge = (vertex, neighbor)
                self.main_window.graph_widget.update()
                new_path = path + [neighbor]
                self.stack.append((neighbor, new_path))
                self.parent[neighbor] = vertex
                self.path.append((vertex, neighbor))
                explanation = f"Переходим по ребру ({vertex}, {neighbor})"
                self.last_backtrack = False
                self.main_window.explanation_widget.append(explanation)
                return False, explanation, self._get_state(), 'neighbor_loop'
            idx += 1

        # Если все соседи просмотрены
        self.stack.pop()
        self.last_backtrack = True
        if len(path) > 1:
            explanation = f"Возврат назад из вершины {vertex} к {path[-2]}"
        else:
            explanation = f"Возврат назад из вершины {vertex} (начало обхода)"
        self.main_window.explanation_widget.append(explanation)
        return False, explanation, self._get_state(), 'backtrack'

    def _get_state(self):
        return {
            'current_path': self.current_path,
            'stack': [v for v, _ in self.stack],
            'visited': list(self.visited),
            'parent': self.parent,
            'paths_checked': self.paths_checked
        }

    def get_pseudocode(self):
        return [
            "1. Инициализация:",
            "   visited = ∅",
            "   stack = [start]",
            "   parent = {}",
            "",
            "2. Пока stack не пуст:",
            "   vertex, path = stack.pop()",
            "   Если vertex не посещена:",
            "       visited.add(vertex)",
            "       Для каждого соседа neighbor:",
            "           Если neighbor не посещён:",
            "               parent[neighbor] = vertex",
            "               stack.append((neighbor, path + [neighbor]))",
            "",
            "3. Завершение:",
            "   Все достижимые вершины посещены, parent содержит дерево обхода"
        ]

    def get_highlight_map(self):
        return {
            'init': 0,
            'main_loop': 4,
            'neighbor_loop': 8,
            'backtrack': 13,
            'finish': 15
        }

    def get_name(self):
        return "DFS (поиск в глубину)"

    def get_description(self):
        return "Алгоритм поиска в глубину (Depth-First Search) — обходит граф в глубину, строит дерево обхода и фиксирует времена входа/выхода."

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
        self.main_window.graph_widget.update()
        return False, message, self._get_state(), 'init'

    def set_end_vertex(self, end_vertex):
        """Устанавливает конечную вершину и начинает поиск пути"""
        self.end_vertex = end_vertex
        self.waiting_for_end = False
        self.main_window.graph_widget.waiting_for_vertex_selection = False
        message = f"Начинаем поиск кратчайшего пути от вершины {self.current_vertex} до вершины {end_vertex}"
        self.main_window.explanation_widget.append(message)
        self.main_window.graph_widget.update()
        return False, message, self._get_state(), 'init'

    def next_step(self):
        """Выполняет следующий шаг алгоритма"""
        if self.waiting_for_end:
            return False, "Выберите конечную вершину", self._get_state(), 'finish'

        # Если есть текущий сосед и мы на шаге сравнения
        if self.current_neighbor is not None and self.comparison_step:
            return self._process_comparison()

        # Если есть необработанные соседи текущей вершины
        if self.current_neighbors:
            return self._prepare_next_neighbor()

        # Сбрасываем подсветку ребра перед переходом к новой вершине
        self.main_window.graph_widget.bfs_current_edge = None
        self.main_window.graph_widget.update()

        # Если все вершины посещены, завершаем алгоритм
        if not self.unvisited:
            return self._finish_algorithm()

        # Находим следующую вершину для обработки
        min_vertex = self._find_min_distance_vertex()
        if min_vertex is None or self.distances[min_vertex] == float('inf'):
            return self._finish_algorithm()

        # Начинаем обработку новой вершины
        self.current_vertex = min_vertex
        self.unvisited.remove(min_vertex)
        self.visited.add(min_vertex)
        self.main_window.graph_widget.visited_vertices = self.visited
        self.main_window.graph_widget.bfs_current = min_vertex

        # Получаем всех соседей в отсортированном порядке
        self.current_neighbors = sorted(list(self.main_window.graph_widget.graph.neighbors(min_vertex)))
        
        message = f"Обрабатываем вершину {min_vertex} (расстояние: {self.distances[min_vertex]})"
        self.main_window.explanation_widget.append(message)
        self.main_window.graph_widget.update()
        
        return False, message, self._get_state(), 'main_loop'

    def _prepare_next_neighbor(self):
        """Подготавливает обработку следующего соседа"""
        self.current_neighbor = self.current_neighbors.pop(0)
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
        return False, "\n".join(message), self._get_state(), 'main_loop'

    def _process_comparison(self):
        """Обрабатывает шаг сравнения расстояний"""
        # Подсвечиваем строку сравнения
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
            self.main_window.graph_widget.update()
        else:
            message.append(f"Оставляем текущее расстояние: {self.distances[self.current_neighbor] if self.distances[self.current_neighbor] != float('inf') else '∞'}")
            # Подсвечиваем строку 'иначе'
            self.main_window.graph_widget.update()
        self.comparison_text = {}
        self.main_window.graph_widget.comparison_text = {}
        self.comparison_step = False
        self.current_neighbor = None
        self.current_distance = None
        self.main_window.graph_widget.update()
        return False, "\n".join(message), self._get_state(), 'main_loop'

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
        self.main_window.explanation_widget.append("Восстановление пути: Если previous[end] не null:")
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
            return True, message, self._get_state(), 'finish'
        return True, "Поиск завершен!", self._get_state(), 'finish'

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

    def get_pseudocode(self):
        return [
            "1. Инициализация:",
            "   distances = {v: ∞ для всех вершин v}  // расстояния до вершин",
            "   previous = {v: null для всех вершин v} // предыдущие вершины",
            "   unvisited = все вершины графа         // непосещенные вершины",
            "   distances[start] = 0                   // расстояние до начальной вершины",
            "   path = []                             // путь до конечной вершины",
            "",
            "2. Основной цикл:",
            "   Пока есть непосещенные вершины:",
            "       v = вершина с min расстоянием среди непосещенных",
            "       Если v не найдена, выход         // нет пути до оставшихся вершин",
            "       Помечаем v как посещенную",
            "",
            "       // Обновляем расстояния до соседей:",
            "       Для каждого соседа u вершины v:",
            "           d = distances[v] + вес ребра (v,u)",
            "           Если d < distances[u]:",
            "               distances[u] = d          // найден более короткий путь",
            "               previous[u] = v           // запоминаем предыдущую вершину",
            "           Иначе:",
            "               # оставляем текущее расстояние",
            "",
            "3. Восстановление пути:",
            "   Если previous[end] не null:",
            "       current = end",
            "       Пока current не null:",
            "           path.append(current)",
            "           current = previous[current]",
            "       path.reverse()",
            "",
            "4. Завершение:",
            "   Возвращаем distances[end], path"
        ]

    def get_highlight_map(self):
        return {
            'init': 0,
            'main_loop': 7,
            'select_vertex': 9,
            'no_path': 10,
            'mark_visited': 11,
            'neighbor_loop': 13,
            'calc_distance': 14,
            'compare': 15,
            'update_distance': 16,
            'update_previous': 17,
            'else': 18,
            'path_recovery': 21,
            'finish': 27
        }

    def get_name(self):
        return "Dijkstra (поиск кратчайшего пути)"

    def get_description(self):
        return "Алгоритм Дейкстры — эффективный способ поиска кратчайших путей в графе без отрицательных весов рёбер."

class BellmanFordAlgorithm(GraphAlgorithm):
    """Реализация алгоритма поиска кратчайшего пути (Беллман-Форд) с подсветкой финального пути и строк псевдокода"""
    def __init__(self, main_window):
        super().__init__(main_window)
        self.distances = {}
        self.previous = {}
        self.vertices = []
        self.edges = []
        self.iteration = 0
        self.edge_index = 0
        self.max_iterations = 0
        self.end_vertex = None
        self.start_vertex = None
        self.waiting_for_end = False
        self.negative_cycle = False
        self.shortest_path = []

    def get_pseudocode(self):
        return [
            "1. Инициализация:",
            "   distances = {v: ∞ для всех вершин v}  // расстояния до вершин",
            "   previous = {v: null для всех вершин v} // предыдущие вершины",
            "   distances[start] = 0                   // расстояние до начальной вершины",
            "   path = []                             // путь до конечной вершины",
            "",
            "2. Основной цикл (V-1 раз):",
            "   Для каждой итерации от 1 до V-1:",
            "       Для каждого ребра (u, v) с весом w:",
            "           Если distances[u] + w < distances[v]:",
            "               distances[v] = distances[u] + w  // найден более короткий путь",
            "               previous[v] = u                  // запоминаем предыдущую вершину",
            "",
            "3. Проверка на отрицательные циклы:",
            "   Для каждого ребра (u, v) с весом w:",
            "       Если distances[u] + w < distances[v]:",
            "           Обнаружен отрицательный цикл!",
            "           Кратчайший путь не существует.",
            "",
            "4. Восстановление пути:",
            "   Если previous[end] не null:",
            "       current = end",
            "       Пока current не null:",
            "           path.append(current)",
            "           current = previous[current]",
            "       path.reverse()",
            "",
            "5. Завершение:",
            "   Возвращаем distances[end], path"
        ]

    def get_highlight_map(self):
        return {
            'init': 0,
            'main_loop': 7,
            'edge_loop': 9,
            'update_distance': 11,
            'update_previous': 12,
            'cycle_check': 15,
            'cycle_found': 17,
            'path_recovery': 21,
            'finish': 27
        }

    def get_name(self):
        return "Bellman-Ford (поиск кратчайшего пути)"

    def get_description(self):
        return "Алгоритм Беллмана-Форда — эффективный способ поиска кратчайших путей в графе с отрицательными весами рёбер."

    def reset(self):
        super().reset()
        self.distances = {}
        self.previous = {}
        self.vertices = []
        self.edges = []
        self.iteration = 0
        self.edge_index = 0
        self.max_iterations = 0
        self.end_vertex = None
        self.start_vertex = None
        self.waiting_for_end = False
        self.negative_cycle = False
        self.shortest_path = []
        self.main_window.graph_widget.distances = {}
        self.main_window.graph_widget.bfs_current_edge = None
        self.main_window.graph_widget.bfs_current = None
        self.main_window.graph_widget.bfs_path = []

    def start(self, start_vertex):
        self.reset()
        graph = self.main_window.graph_widget.graph
        self.vertices = list(graph.nodes())
        if isinstance(graph, nx.DiGraph):
            self.edges = list(graph.edges(data=True))
        else:
            edges = list(graph.edges(data=True))
            undirected_edges = []
            for u, v, data in edges:
                undirected_edges.append((u, v, data))
                undirected_edges.append((v, u, data))
            self.edges = undirected_edges
        self.max_iterations = len(self.vertices) - 1
        for v in self.vertices:
            self.distances[v] = float('inf')
            self.previous[v] = None
        self.distances[start_vertex] = 0
        self.start_vertex = start_vertex
        self.waiting_for_end = True
        self.main_window.graph_widget.distances = self.distances.copy()
        self.main_window.graph_widget.bfs_current = start_vertex
        self.main_window.graph_widget.waiting_for_vertex_selection = True
        self.main_window.explanation_widget.clear()
        self.main_window.explanation_widget.append("Выберите конечную вершину")
        self.main_window.graph_widget.update()
        return False, "Выберите конечную вершину", self._get_state(), 'init'

    def set_end_vertex(self, end_vertex):
        self.end_vertex = end_vertex
        self.waiting_for_end = False
        self.iteration = 0
        self.edge_index = 0
        self.negative_cycle = False
        self.shortest_path = []
        self.main_window.graph_widget.waiting_for_vertex_selection = False
        self.main_window.explanation_widget.append(f"Начинаем поиск кратчайшего пути от вершины {self.start_vertex} до вершины {end_vertex}")
        self.main_window.graph_widget.update()
        return False, f"Начинаем поиск кратчайшего пути от вершины {self.start_vertex} до вершины {end_vertex}", self._get_state(), 'init'

    def next_step(self):
        """Выполняет следующий шаг алгоритма"""
        if self.waiting_for_end:
            return False, "Выберите конечную вершину", self._get_state(), 'finish'

        if self.negative_cycle:
            return True, "Обнаружен отрицательный цикл! Кратчайший путь не существует.", self._get_state(), 'finish'

        # Если все итерации завершены
        if self.iteration >= self.max_iterations:
            # Проверка на отрицательные циклы
            self.main_window.explanation_widget.append("Проверка на отрицательные циклы:")
            for u, v, data in self.edges:
                weight = data.get('weight', 1)
                if self.distances[u] != float('inf') and self.distances[u] + weight < self.distances[v]:
                    self.negative_cycle = True
                    self.main_window.graph_widget.bfs_current_edge = (u, v)
                    self.main_window.graph_widget.update()
                    return True, f"Обнаружен отрицательный цикл по ребру ({u}, {v})! Кратчайший путь не существует.", self._get_state(), 'finish'
            
            # Если отрицательных циклов нет, восстанавливаем путь
            return self._finish_algorithm()

        # Обработка текущего ребра
        if self.edge_index < len(self.edges):
            u, v, data = self.edges[self.edge_index]
            weight = data.get('weight', 1)
            
            self.main_window.graph_widget.bfs_current_edge = (u, v)
            self.main_window.graph_widget.bfs_current = u
            
            message = f"Проверяем ребро ({u}, {v}) с весом {weight}: "
            
            if self.distances[u] != float('inf'):
                new_distance = self.distances[u] + weight
                if new_distance < self.distances[v]:
                    old_distance = self.distances[v]
                    self.distances[v] = new_distance
                    self.previous[v] = u
                    message += f"Обновляем расстояние: {old_distance if old_distance != float('inf') else '∞'} → {new_distance}"
                else:
                    message += f"Без изменений. Текущее расстояние: {self.distances[v] if self.distances[v] != float('inf') else '∞'}"
            else:
                message += f"Без изменений. Текущее расстояние: {self.distances[v] if self.distances[v] != float('inf') else '∞'}"
            
            self.main_window.graph_widget.distances = self.distances.copy()
            self.main_window.explanation_widget.append(message)
            self.main_window.graph_widget.update()
            
            self.edge_index += 1
            return False, message, self._get_state(), 'main_loop'
        
        # Переход к следующей итерации
        self.iteration += 1
        self.edge_index = 0
        message = f"Завершена итерация {self.iteration}"
        self.main_window.explanation_widget.append(message)
        self.main_window.graph_widget.bfs_current_edge = None
        self.main_window.graph_widget.bfs_current = None
        self.main_window.graph_widget.update()
        
        return False, message, self._get_state(), 'main_loop'

    def _finish_algorithm(self):
        self.main_window.explanation_widget.append("Восстановление пути: Если previous[end] не null:")
        if self.end_vertex is not None:
            self.shortest_path = self._reconstruct_path()
            path_edges = [(self.shortest_path[i], self.shortest_path[i+1]) for i in range(len(self.shortest_path)-1)]
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
            return True, message, self._get_state(), 'finish'
        return True, "Поиск завершён!", self._get_state(), 'finish'

    def _reconstruct_path(self):
        """Восстанавливает кратчайший путь от начальной до конечной вершины"""
        path = []
        current = self.end_vertex
        visited = set()
        while current is not None and current not in visited:
            path.append(current)
            visited.add(current)
            current = self.previous.get(current, None)
        path.reverse()
        # Проверяем, что путь начинается с начальной вершины (расстояние = 0)
        if path and self.distances.get(path[0], None) == 0:
            # Приводим индексы к int, если это возможно
            try:
                path = [int(v) for v in path]
            except Exception:
                pass
            return path
        return []

    def _get_state(self):
        """Возвращает текущее состояние алгоритма"""
        return {
            'distances': self.distances.copy(),
            'previous': self.previous.copy(),
            'vertices': self.vertices,
            'edges': self.edges,
            'iteration': self.iteration,
            'edge_index': self.edge_index,
            'max_iterations': self.max_iterations,
            'start_vertex': self.start_vertex,
            'end_vertex': self.end_vertex,
            'negative_cycle': self.negative_cycle,
            'shortest_path': self.shortest_path,
            'path': self.shortest_path
        }

class MaxPathAlgorithm(GraphAlgorithm):
    """Поиск максимального простого пути между двумя вершинами (как в graphonline), с подробной визуализацией процесса поиска"""
    def __init__(self, main_window):
        super().__init__(main_window)
        self.max_path = []
        self.max_weight = float('-inf')
        self.start_vertex = None
        self.end_vertex = None
        self.waiting_for_end = False
        self.finished = False
        self.current_path = []
        self.current_weight = 0
        self.stack = []
        self.visited = set()
        self.step_mode = False
        self.paths_checked = 0
        self.last_step = None
        self.last_backtrack = False

    def reset(self):
        super().reset()
        self.max_path = []
        self.max_weight = float('-inf')
        self.start_vertex = None
        self.end_vertex = None
        self.waiting_for_end = False
        self.finished = False
        self.current_path = []
        self.current_weight = 0
        self.stack = []
        self.visited = set()
        self.step_mode = False
        self.paths_checked = 0
        self.last_step = None
        self.last_backtrack = False
        self.main_window.graph_widget.bfs_path = []
        self.main_window.graph_widget.bfs_current = None
        self.main_window.graph_widget.bfs_current_edge = None
        self.main_window.graph_widget.visited_vertices = set()
        self.main_window.graph_widget.update()

    def start(self, start_vertex):
        self.reset()
        self.start_vertex = int(start_vertex)
        self.waiting_for_end = True
        self.main_window.graph_widget.waiting_for_vertex_selection = True
        self.main_window.explanation_widget.clear()
        self.main_window.explanation_widget.append("Выберите конечную вершину")
        self.main_window.graph_widget.bfs_current = self.start_vertex
        self.main_window.graph_widget.update()
        return False, "Выберите конечную вершину", self._get_state(), 'init'

    def set_end_vertex(self, end_vertex):
        self.end_vertex = int(end_vertex)
        self.waiting_for_end = False
        self.finished = False
        self.current_path = [self.start_vertex]
        self.current_weight = 0
        self.visited = set([self.start_vertex])
        # stack: (current, path, weight, visited, neighbors_iter, parent)
        neighbors_iter = iter(self.main_window.graph_widget.graph.neighbors(self.start_vertex))
        self.stack = [(self.start_vertex, [self.start_vertex], 0, set([self.start_vertex]), neighbors_iter, None)]
        self.main_window.graph_widget.waiting_for_vertex_selection = False
        self.main_window.explanation_widget.append(f"Начинаем поиск максимального пути от вершины {self.start_vertex} до вершины {self.end_vertex}")
        self.main_window.graph_widget.update()
        self.step_mode = True
        self.last_step = None
        self.last_backtrack = False
        return False, f"Начинаем поиск максимального пути от вершины {self.start_vertex} до вершины {self.end_vertex}", self._get_state(), 'init'

    def next_step(self):
        """Выполняет следующий шаг алгоритма"""
        if self.waiting_for_end:
            return False, "Выберите конечную вершину", self._get_state(), 'finish'

        if self.finished:
            msg = self._final_message()
            return True, msg, self._get_state(), 'finish'

        if not self.stack:
            self.finished = True
            msg = self._final_message()
            return True, msg, self._get_state(), 'finish'

        # Снимаем подсветку предыдущего ребра, если был backtrack
        if self.last_backtrack:
            self.main_window.graph_widget.bfs_current_edge = None
            self.main_window.graph_widget.update()
            self.last_backtrack = False

        current, path, weight, visited, neighbors_iter, parent = self.stack[-1]
        self.current_path = path.copy()  # Создаем копию пути
        self.current_weight = weight
        self.visited = visited.copy()    # Создаем копию множества посещенных вершин
        self.main_window.graph_widget.bfs_current = current
        self.main_window.graph_widget.bfs_path = [(path[i], path[i+1]) for i in range(len(path)-1)]

        try:
            neighbor = next(neighbors_iter)
            if neighbor not in visited:
                # Подсветка ребра
                self.main_window.graph_widget.bfs_current_edge = (current, neighbor)
                self.main_window.graph_widget.update()
                edge_weight = self.main_window.graph_widget.graph[current][neighbor].get('weight', 1)
                new_path = path + [neighbor]
                new_visited = visited.copy()
                new_visited.add(neighbor)
                new_neighbors_iter = iter(self.main_window.graph_widget.graph.neighbors(neighbor))
                self.stack.append((neighbor, new_path, weight + edge_weight, new_visited, new_neighbors_iter, current))
                
                explanation = f"Переходим по ребру ({current}, {neighbor}) с весом {edge_weight}"
                self.last_step = (current, neighbor)
                self.last_backtrack = False
                self.main_window.explanation_widget.append(explanation)
                return False, explanation, self._get_state(), 'main_loop'
            else:
                explanation = f"Сосед {neighbor} уже был в текущем пути — пропускаем"
                self.last_step = None
                self.main_window.explanation_widget.append(explanation)
                return False, explanation, self._get_state(), 'main_loop'
        except StopIteration:
            # Все соседи просмотрены — backtrack
            if current == self.end_vertex:
                self.paths_checked += 1
                if weight > self.max_weight:
                    self.max_weight = weight
                    self.max_path = path.copy()
                    explanation = f"Найден новый максимальный путь длиной {weight}!"
                else:
                    explanation = f"Достигли конечной вершины, но путь длиной {weight} не максимальный (текущий максимум: {self.max_weight})"
            else:
                explanation = f"Возврат назад из вершины {current}"
            
            self.stack.pop()
            self.last_backtrack = True
            self.main_window.explanation_widget.append(explanation)
            return False, explanation, self._get_state(), 'backtrack'

    def _final_message(self):
        if self.max_path:
            self.main_window.graph_widget.bfs_path = [(self.max_path[i], self.max_path[i+1]) for i in range(len(self.max_path)-1)]
            self.main_window.graph_widget.bfs_current = None
            self.main_window.graph_widget.bfs_current_edge = None
            self.main_window.graph_widget.update()
            return f"Найден максимальный путь длиной {self.max_weight}:\n{' → '.join(map(str, self.max_path))}"
        else:
            self.main_window.graph_widget.bfs_path = []
            self.main_window.graph_widget.bfs_current = None
            self.main_window.graph_widget.bfs_current_edge = None
            self.main_window.graph_widget.update()
            return "Путь не найден"

    def _get_state(self):
        """Возвращает текущее состояние алгоритма"""
        current_stack = []
        for vertex, path, weight, visited, _, _ in self.stack:
            current_stack.append({
                'vertex': vertex,
                'path': path,
                'weight': weight,
                'visited': list(visited)
            })
            
        return {
            'max_path': self.max_path,
            'max_weight': self.max_weight,
            'current_path': self.current_path,
            'current_weight': self.current_weight,
            'stack': current_stack,
            'paths_checked': self.paths_checked,
            'visited': sorted(list(self.visited)),
            'start_vertex': self.start_vertex,
            'end_vertex': self.end_vertex,
            'last_step': self.last_step,
            'waiting_for_end': self.waiting_for_end,
            'finished': self.finished
        }

    def get_pseudocode(self):
        return [
            "1. Запускаем поиск всех простых путей из start в end (DFS)",
            "2. Для каждого пути считаем сумму весов рёбер",
            "3. Сохраняем путь с максимальной суммой",
            "4. Возвращаем максимальный путь и его длину"
        ]

    def get_highlight_map(self):
        return {
            'init': 0,
            'main_loop': 4,
            'finish': 3
        }

    def get_name(self):
        return "MaxPath (поиск максимального простого пути)"

    def get_description(self):
        return "Поиск самого длинного простого пути между двумя вершинами (экспоненциальный перебор, как в graphonline)."

class KruskalAlgorithm(GraphAlgorithm):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.edges_sorted = []
        self.parent = {}
        self.rank = {}
        self.mst_edges = []
        self.edge_idx = 0

    def reset(self):
        super().reset()
        self.edges_sorted = []
        self.parent = {}
        self.rank = {}
        self.mst_edges = []
        self.edge_idx = 0
        self.main_window.graph_widget.bfs_path = []
        self.main_window.graph_widget.bfs_current_edge = None
        self.main_window.graph_widget.bfs_current = None
        self.main_window.graph_widget.update()

    def find(self, v):
        if self.parent[v] != v:
            self.parent[v] = self.find(self.parent[v])
        return self.parent[v]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u == root_v:
            return False
        if self.rank[root_u] < self.rank[root_v]:
            self.parent[root_u] = root_v
        else:
            self.parent[root_v] = root_u
            if self.rank[root_u] == self.rank[root_v]:
                self.rank[root_u] += 1
        return True

    def get_pseudocode(self):
        return [
            "1. edges = отсортировать_по_весу(ребра)",
            "2. для v в вершинах: parent[v] = v",
            "3. mst = []",
            "4. для (u, v) в edges:",
            "5.     если find(u) != find(v):",
            "6.         mst.append((u, v))",
            "7.         union(u, v)",
            "8.     иначе: пропустить (u, v)",
            "9. если len(mst) == n-1: завершить"
        ]

    def get_highlight_map(self):
        return {
            'init': 0,
            'main_loop': 4,
            'finish': 9
        }

    def get_name(self):
        return "Kruskal (алгоритм Краскала)"

    def get_description(self):
        return "Алгоритм Краскала — эффективный способ поиска минимального остовного дерева в графе."

    def start(self):
        self.reset()
        graph = self.main_window.graph_widget.graph
        self.edges_sorted = sorted(graph.edges(data=True), key=lambda e: e[2].get('weight', 1))
        for v in graph.nodes():
            self.parent[v] = v
            self.rank[v] = 0
        self.edge_idx = 0
        self.mst_edges = []
        self.main_window.graph_widget.bfs_path = []
        self.main_window.graph_widget.bfs_current_edge = None
        self.main_window.graph_widget.bfs_current = None
        self.main_window.graph_widget.update()
        return False, "Начинаем алгоритм Краскала", self._get_state(), 'init'

    def next_step(self):
        if len(self.mst_edges) == len(self.parent) - 1:
            self.main_window.graph_widget.bfs_path = self.mst_edges
            self.main_window.graph_widget.bfs_current_edge = None
            self.main_window.graph_widget.bfs_current = None
            self.main_window.graph_widget.update()
            # Формируем отображение множеств для визуализации
            sets = {}
            for v in self.parent:
                root = self.find(v)
                if root not in sets:
                    sets[root] = []
                sets[root].append(v)
            self.main_window.graph_widget.kruskal_sets = sets
            # Считаем итоговый вес остова
            total_weight = sum(self.main_window.graph_widget.graph[u][v].get('weight', 1) for u, v in self.mst_edges)
            message = f"Построено минимальное остовное дерево!\nВес остова: {total_weight}"
            if self.main_window.directed_checkbox.isChecked():
                message += "\nОриентация дуг была проигнорирована, так как остовное дерево можно строить только для неориентированных графов"
            return True, message, self._get_state(), 'finish'
        if self.edge_idx >= len(self.edges_sorted):
            self.main_window.graph_widget.bfs_path = self.mst_edges
            self.main_window.graph_widget.bfs_current_edge = None
            self.main_window.graph_widget.bfs_current = None
            self.main_window.graph_widget.update()
            sets = {}
            for v in self.parent:
                root = self.find(v)
                if root not in sets:
                    sets[root] = []
                sets[root].append(v)
            self.main_window.graph_widget.kruskal_sets = sets
            total_weight = sum(self.main_window.graph_widget.graph[u][v].get('weight', 1) for u, v in self.mst_edges)
            message = f"Рёбра закончились, остов построен!\nВес остова: {total_weight}"
            if self.main_window.directed_checkbox.isChecked():
                message += "\nОриентация дуг была проигнорирована, так как остовное дерево можно строить только для неориентированных графов"
            return True, message, self._get_state(), 'finish'
        u, v, data = self.edges_sorted[self.edge_idx]
        self.main_window.graph_widget.bfs_current_edge = (u, v)
        self.main_window.graph_widget.bfs_current = u
        self.main_window.graph_widget.update()
        self.edge_idx += 1
        if self.find(u) != self.find(v):
            self.union(u, v)
            self.mst_edges.append((u, v))
            self.main_window.graph_widget.bfs_path = self.mst_edges
            # Формируем отображение множеств для визуализации
            sets = {}
            for vv in self.parent:
                root = self.find(vv)
                if root not in sets:
                    sets[root] = []
                sets[root].append(vv)
            self.main_window.graph_widget.kruskal_sets = sets
            self.main_window.graph_widget.update()
            message = f"Добавили ребро ({u}, {v}) в остов"
        else:
            # Формируем отображение множеств для визуализации
            sets = {}
            for vv in self.parent:
                root = self.find(vv)
                if root not in sets:
                    sets[root] = []
                sets[root].append(vv)
            self.main_window.graph_widget.kruskal_sets = sets
            message = f"Пропустили ребро ({u}, {v}), чтобы не образовать цикл"
        return False, message, self._get_state(), 'main_loop'

    def _get_state(self):
        return {
            'mst_edges': self.mst_edges,
            'parent': self.parent.copy(),
            'edge_idx': self.edge_idx,
            'edges_sorted': [(u, v, d.get('weight', 1)) for u, v, d in self.edges_sorted]
        }

class PrimAlgorithm(GraphAlgorithm):
    """Пошаговая визуализация алгоритма Прима (минимальное остовное дерево)"""
    def __init__(self, main_window):
        super().__init__(main_window)
        self.in_tree = set()
        self.edges_in_tree = []
        self.candidates = []
        self.current_edge = None
        self.step = 0
        self.finished = False

    def reset(self):
        super().reset()
        self.in_tree = set()
        self.edges_in_tree = []
        self.candidates = []
        self.current_edge = None
        self.step = 0
        self.finished = False
        self.main_window.graph_widget.bfs_path = []
        self.main_window.graph_widget.bfs_current_edge = None
        self.main_window.graph_widget.bfs_current = None
        self.main_window.graph_widget.update()

    def get_pseudocode(self):
        return [
            "1. Выбрать произвольную вершину v₀, добавить в дерево:",
            "   V_T = {v₀}, X_T = ∅",
            "2. Пока V_T ≠ V:",
            "   Найти кратчайшее ребро {u, v}, где u ∈ V_T, v ∉ V_T",
            "   Добавить v в V_T, ребро {u, v} в X_T",
            "3. Если V_T = V, то T = (V_T, X_T) — минимальное покрывающее дерево"
        ]

    def get_highlight_map(self):
        return {
            'init': 0,
            'main_loop': 2,
            'add_edge': 4,
            'finish': 6
        }

    def get_name(self):
        return "Prim (алгоритм Прима)"

    def get_description(self):
        return "Алгоритм Прима — эффективный способ поэтапного построения минимального остовного дерева."

    def start(self, start_vertex=None):
        self.reset()
        graph = self.main_window.graph_widget.graph
        vertices = list(graph.nodes())
        if not vertices:
            return True, "Граф пуст", self._get_state(), 'finish'
        if start_vertex is None:
            start_vertex = vertices[0]
        self.in_tree = {start_vertex}
        self.edges_in_tree = []
        self.candidates = []
        self.current_edge = None
        self.step = 0
        self.finished = False
        self.main_window.graph_widget.visited_vertices = set(self.in_tree)
        self.main_window.graph_widget.bfs_path = []
        self.main_window.graph_widget.bfs_current = start_vertex
        self.main_window.graph_widget.bfs_current_edge = None
        self.main_window.graph_widget.update()
        return False, f"Начинаем с вершины {start_vertex}", self._get_state(), 'init'

    def next_step(self):
        if self.finished:
            return True, "Построено минимальное остовное дерево!", self._get_state(), 'finish'
        graph = self.main_window.graph_widget.graph
        if len(self.in_tree) == len(graph.nodes()):
            self.finished = True
            self.main_window.graph_widget.bfs_path = self.edges_in_tree
            self.main_window.graph_widget.bfs_current_edge = None
            self.main_window.graph_widget.bfs_current = None
            self.main_window.graph_widget.update()
            total_weight = sum(graph[u][v].get('weight', 1) for u, v in self.edges_in_tree)
            msg = f"Построено минимальное остовное дерево!\nВес остова: {total_weight}"
            return True, msg, self._get_state(), 'finish'
        # Находим все рёбра, ведущие из дерева наружу
        candidates = []
        for u in self.in_tree:
            for v in graph.neighbors(u):
                if v not in self.in_tree:
                    weight = graph[u][v].get('weight', 1)
                    candidates.append((weight, u, v))
        if not candidates:
            self.finished = True
            return True, "Граф несвязный — остов не построен", self._get_state(), 'finish'
        # Выбираем минимальное ребро
        candidates.sort()
        weight, u, v = candidates[0]
        self.current_edge = (u, v)
        self.candidates = candidates
        self.main_window.graph_widget.bfs_current_edge = (u, v)
        self.main_window.graph_widget.bfs_current = v
        self.main_window.graph_widget.update()
        # Добавляем вершину и ребро в дерево
        self.in_tree.add(v)
        self.edges_in_tree.append((u, v))
        self.main_window.graph_widget.visited_vertices = set(self.in_tree)
        self.main_window.graph_widget.bfs_path = self.edges_in_tree
        self.main_window.graph_widget.update()
        msg = f"Добавили ребро ({u}, {v}) с весом {weight} в остов"
        return False, msg, self._get_state(), 'add_edge'

    def _get_state(self):
        return {
            'in_tree': set(self.in_tree),
            'edges_in_tree': list(self.edges_in_tree),
            'candidates': list(self.candidates),
            'current_edge': self.current_edge
        } 