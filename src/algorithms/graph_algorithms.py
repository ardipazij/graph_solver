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
    """Реализация алгоритма поиска в ширину (BFS)"""
    
    def __init__(self, main_window):
        super().__init__(main_window)
        self.queue = []
        self.parent = {}
        self.current_vertex = None
        self.current_neighbor = None
        self.step = 0

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

    def get_highlight_map(self):
        return {
            'init': 0,
            'main_loop': 6,
            'pop_vertex': 8,
            'append_result': 9,
            'add_visited': 10,
            'neighbor_loop': 12,
            'neighbor_check': 13,
            'add_neighbor_visited': 14,
            'add_neighbor_queue': 15,
            'set_parent': 16,
            'finish': 18
        }

    def get_name(self):
        return "BFS (поиск в ширину)"

    def get_description(self):
        return "Алгоритм поиска в ширину (Breadth-First Search) — находит кратчайшие пути в невзвешенном графе."

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
        self.main_window.graph_widget.visited_vertices = self.visited
        self.main_window.graph_widget.bfs_current = start_vertex
        self.main_window.graph_widget.update()
        return False, message, self._get_state(), 'init'

    def next_step(self):
        """Выполняет следующий шаг алгоритма"""
        if not self.queue:
            message = "Обход завершен!"
            self.main_window.explanation_widget.append(message)
            self.main_window.graph_widget.bfs_current = None
            self.main_window.graph_widget.bfs_current_edge = None
            self.main_window.graph_widget.update()
            return True, message, self._get_state(), 'finish'
        self.current_vertex = self.queue.pop(0)
        self.result.append(self.current_vertex)
        self.visited.add(self.current_vertex)
        self.main_window.graph_widget.bfs_current = self.current_vertex
        message = f"Обрабатываем вершину {self.current_vertex}"
        self.main_window.explanation_widget.append(message)
        # Получаем соседей текущей вершины
        neighbors = list(self.main_window.graph_widget.graph.neighbors(self.current_vertex))
        for neighbor in neighbors:
            self.main_window.graph_widget.bfs_current_edge = (self.current_vertex, neighbor)
            self.main_window.graph_widget.update()
            if neighbor not in self.visited:
                message = f"Найден непосещенный сосед: {neighbor}"
                self.main_window.explanation_widget.append(message)
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
        return False, message, self._get_state(), 'main_loop'

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
        self.main_window.graph_widget.bfs_current = start_vertex
        self.main_window.graph_widget.update()
        return False, message, self._get_state(), 'init'

    def next_step(self):
        """Выполняет следующий шаг алгоритма DFS"""
        if not self.stack:
            message = "Обход завершен!"
            self.main_window.explanation_widget.append(message)
            self.main_window.graph_widget.bfs_current = None
            self.main_window.graph_widget.bfs_current_edge = None
            self.main_window.graph_widget.update()
            return True, message, self._get_state(), 'finish'

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
            # Получаем соседей в обратном отсортированном порядке
            neighbors = sorted(list(self.main_window.graph_widget.graph.neighbors(vertex)), reverse=True)
            # Обрабатываем каждого соседа
            for neighbor in neighbors:
                self.main_window.graph_widget.bfs_current_edge = (vertex, neighbor)
                self.main_window.graph_widget.update()
                if neighbor not in self.visited:
                    message = f"Найден непосещенный сосед: {neighbor}"
                    self.main_window.explanation_widget.append(message)
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
        return False, message, self._get_state(), 'main_loop'

    def _get_state(self):
        """Возвращает текущее состояние алгоритма (только ключевые переменные)"""
        return {
            'visited': self.visited,
            'stack': self.stack,
            'parent': getattr(self, 'parent', {}),
            'current': self.current
        }

    def get_pseudocode(self):
        return [
            "1. Инициализация:",
            "   visited = ∅        // множество посещенных вершин",
            "   stack = [start]    // стек вершин для обработки",
            "   result = []        // результат обхода",
            "   parent = {}        // словарь для хранения родительских вершин",
            "   discovery_time = {} // время обнаружения вершин",
            "   finish_time = {}   // время завершения обработки вершин",
            "   time = 0           // текущее время",
            "",
            "2. Основной цикл:",
            "   Пока stack не пуст:",
            "       vertex = stack.pop()     // берем последнюю вершину из стека",
            "       Если vertex не посещена:",
            "           time += 1",
            "           discovery_time[vertex] = time  // запоминаем время обнаружения",
            "           result.append(vertex)    // добавляем в результат",
            "           visited.add(vertex)      // помечаем как посещенную",
            "",
            "           // Обработка соседей:",
            "           Для каждого соседа neighbor вершины vertex:",
            "               Если neighbor не посещен:",
            "                   stack.append(neighbor)    // добавляем в стек",
            "                   parent[neighbor] = vertex // запоминаем родителя",
            "           time += 1",
            "           finish_time[vertex] = time  // запоминаем время завершения",
            "",
            "3. Завершение:",
            "   Возвращаем result, parent, discovery_time, finish_time"
        ]

    def get_highlight_map(self):
        return {
            'init': 0,
            'main_loop': 8,
            'pop_vertex': 10,
            'if_not_visited': 11,
            'inc_time': 12,
            'set_discovery': 13,
            'append_result': 14,
            'add_visited': 15,
            'neighbor_loop': 17,
            'neighbor_check': 18,
            'push_stack': 19,
            'set_parent': 20,
            'inc_time2': 21,
            'set_finish': 22,
            'finish': 24
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
            return False, "Выберите конечную вершину", self._get_state(), 'init'

        # Если есть текущий сосед и мы на шаге сравнения
        if self.current_neighbor is not None and self.comparison_step:
            return self._process_comparison()

        # Если есть необработанные соседи текущей вершины
        if self.current_neighbors:
            return self._prepare_next_neighbor()

        # Сбрасываем подсветку ребра перед переходом к новой вершине
        self.main_window.graph_widget.bfs_current_edge = None
        self.main_window.graph_widget.update()

        # Если все соседи обработаны или это первый шаг
        if not self.unvisited:
            return self._finish_algorithm()

        # Находим следующую вершину для обработки
        min_vertex = self._find_min_distance_vertex()
        if min_vertex is None:
            return True, "Не удалось найти путь до всех вершин", self._get_state(), 'no_path'

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
        return False, "\n".join(message), self._get_state(), 'neighbor_loop'

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
        return False, "\n".join(message), self._get_state(), 'compare'

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
            return True, message, self._get_state(), 'path_recovery'
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
        if self.waiting_for_end:
            return False, "Выберите конечную вершину", self._get_state(), 'init'
        if self.negative_cycle:
            msg = "Обнаружен отрицательный цикл! Кратчайший путь не существует."
            self.main_window.graph_widget.bfs_current = None
            self.main_window.graph_widget.bfs_current_edge = None
            self.main_window.graph_widget.update()
            return True, msg, self._get_state(), 'cycle_found'
        # Основной цикл (V-1 итераций)
        if self.iteration < self.max_iterations:
            if self.edge_index == 0:
                self.main_window.explanation_widget.append("2. Основной цикл (V-1 раз):")
            if self.edge_index >= len(self.edges):
                self.iteration += 1
                self.edge_index = 0
                self.main_window.explanation_widget.append(f"Завершена итерация {self.iteration}")
                self.main_window.graph_widget.bfs_current_edge = None
                self.main_window.graph_widget.bfs_current = None
                self.main_window.graph_widget.update()
                return False, f"Завершена итерация {self.iteration}", self._get_state(), 'finish'
            u, v, data = self.edges[self.edge_index]
            weight = data.get('weight', 1)
            self.main_window.graph_widget.bfs_current_edge = (u, v)
            self.main_window.graph_widget.bfs_current = u
            self.main_window.explanation_widget.append(f"Для каждого ребра ({u}, {v}) с весом {weight}:")
            msg = f"Проверяем ребро ({u}, {v}) с весом {weight}: "
            if self.distances[u] != float('inf'):
                if self.distances[u] + weight < self.distances[v]:
                    old = self.distances[v]
                    self.distances[v] = self.distances[u] + weight
                    self.main_window.explanation_widget.append(f"Обновляем расстояние: {old if old != float('inf') else '∞'} → {self.distances[v]} (previous[{v}] = {u})")
                    self.previous[v] = u
                    msg += f"Обновляем расстояние: {old if old != float('inf') else '∞'} → {self.distances[v]} (previous[{v}] = {u})"
                else:
                    self.main_window.explanation_widget.append(f"Без изменений. Текущее расстояние: {self.distances[v] if self.distances[v] != float('inf') else '∞'}")
                    msg += f"Без изменений. Текущее расстояние: {self.distances[v] if self.distances[v] != float('inf') else '∞'}"
            else:
                self.main_window.explanation_widget.append(f"Без изменений. Текущее расстояние: {self.distances[v] if self.distances[v] != float('inf') else '∞'}")
                msg += f"Без изменений. Текущее расстояние: {self.distances[v] if self.distances[v] != float('inf') else '∞'}"
            self.main_window.graph_widget.distances = self.distances.copy()
            self.main_window.graph_widget.update()
            self.edge_index += 1
            return False, msg, self._get_state(), 'main_loop'
        # Проверка на отрицательные циклы
        if self.iteration == self.max_iterations:
            self.main_window.explanation_widget.append("3. Проверка на отрицательные циклы:")
            for u, v, data in self.edges:
                weight = data.get('weight', 1)
                if self.distances[u] != float('inf') and self.distances[u] + weight < self.distances[v]:
                    self.negative_cycle = True
                    self.main_window.explanation_widget.append(f"Обнаружен отрицательный цикл по ребру ({u}, {v})!")
                    self.main_window.graph_widget.bfs_current_edge = (u, v)
                    self.main_window.graph_widget.update()
                    return True, "Обнаружен отрицательный цикл! Кратчайший путь не существует.", self._get_state(), 'cycle_found'
            # Восстановление пути
            self.main_window.explanation_widget.append("4. Восстановление пути:")
            self.shortest_path = self._reconstruct_path()
            path_edges = [(self.shortest_path[i], self.shortest_path[i+1]) for i in range(len(self.shortest_path)-1)]
            self.main_window.graph_widget.bfs_path = path_edges
            distance = self.distances[self.end_vertex]
            if distance == float('inf'):
                self.main_window.explanation_widget.append("5. Завершение")
                msg = f"Путь до вершины {self.end_vertex} не найден"
            else:
                self.main_window.explanation_widget.append("5. Завершение")
                path_str = " → ".join(map(str, self.shortest_path))
                msg = f"Найден кратчайший путь длиной {distance}:\n{path_str}"
            self.main_window.graph_widget.bfs_current = None
            self.main_window.graph_widget.bfs_current_edge = None
            self.main_window.graph_widget.update()
            self.iteration += 1  # чтобы не заходить сюда снова
            return True, msg, self._get_state(), 'finish'
        self.main_window.explanation_widget.append("5. Завершение")
        return True, "Поиск завершён!", self._get_state(), 'finish'

    def _reconstruct_path(self):
        """Восстанавливает кратчайший путь от начальной до конечной вершины"""
        path = []
        current = self.end_vertex
        while current is not None:
            path.append(current)
            current = self.previous[current]
        path.reverse()
        if path and path[0] == self.start_vertex:
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
    """Реализация алгоритма поиска максимального пути"""
    
    def __init__(self, main_window):
        super().__init__(main_window)
        self.distances = {}  # расстояния до вершин
        self.previous = {}   # предыдущие вершины для восстановления пути
        self.unvisited = set()  # непосещенные вершины
        self.current_vertex = None
        self.end_vertex = None  # конечная вершина
        self.shortest_path = []  # максимальный путь
        self.waiting_for_end = False  # флаг ожидания выбора конечной вершины
        self.current_neighbors = []  # текущие необработанные соседи
        self.processing_vertex = False  # флаг обработки текущей вершины
        self.current_neighbor = None  # текущий обрабатываемый сосед
        self.comparison_step = False  # флаг шага сравнения
        self.current_distance = None  # текущее вычисленное расстояние
        self.comparison_text = {}  # текст сравнения для отображения над вершинами
        self.max_iterations = 0  # максимальное число итераций
        self.iteration = 0  # текущая итерация
        
    def reset(self):
        """Сбрасывает состояние алгоритма"""
        super().reset()
        self.comparison_text = {}
        self.main_window.graph_widget.distances = {}
        self.main_window.graph_widget.comparison_text = {}
        self.iteration = 0
        self.max_iterations = 0

    def start(self, start_vertex):
        """Начинает поиск максимального пути из указанной вершины"""
        self.reset()
        
        # Инициализация расстояний и множества непосещенных вершин
        for vertex in self.main_window.graph_widget.graph.nodes():
            self.distances[vertex] = float('-inf')  # Изменено на -inf для поиска максимума
            self.previous[vertex] = None
            self.unvisited.add(vertex)
            
        # Устанавливаем расстояние до начальной вершины
        self.distances[start_vertex] = 0
        self.current_vertex = start_vertex
        self.waiting_for_end = True
        self.max_iterations = len(self.main_window.graph_widget.graph.nodes())
        
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
        message = f"Начинаем поиск максимального пути от вершины {self.current_vertex} до вершины {end_vertex}"
        self.main_window.explanation_widget.append(message)
        self.main_window.graph_widget.update()
        return False, message, self._get_state(), 'init'

    def next_step(self):
        """Выполняет следующий шаг алгоритма"""
        if self.waiting_for_end:
            return False, "Выберите конечную вершину", self._get_state(), 'init'

        # Проверка на превышение максимального числа итераций
        if self.iteration >= self.max_iterations:
            return self._finish_algorithm()

        # Если есть текущий сосед и мы на шаге сравнения
        if self.current_neighbor is not None and self.comparison_step:
            return self._process_comparison()

        # Если есть необработанные соседи текущей вершины
        if self.current_neighbors:
            return self._prepare_next_neighbor()

        # Сбрасываем подсветку ребра перед переходом к новой вершине
        self.main_window.graph_widget.bfs_current_edge = None
        self.main_window.graph_widget.update()

        # Если все соседи обработаны или это первый шаг
        if not self.unvisited:
            return self._finish_algorithm()

        # Находим следующую вершину для обработки
        max_vertex = self._find_max_distance_vertex()  # Изменено на поиск максимума
        if max_vertex is None:
            return self._finish_algorithm()

        # Начинаем обработку новой вершины
        self.current_vertex = max_vertex
        self.unvisited.remove(max_vertex)
        self.visited.add(max_vertex)
        self.main_window.graph_widget.visited_vertices = self.visited
        self.main_window.graph_widget.bfs_current = max_vertex
        
        # Получаем всех соседей в отсортированном порядке
        self.current_neighbors = sorted(list(self.main_window.graph_widget.graph.neighbors(max_vertex)))
        self.iteration += 1
        
        message = f"Обрабатываем вершину {max_vertex} (расстояние: {self.distances[max_vertex] if self.distances[max_vertex] != float('-inf') else '-∞'})"
        self.main_window.explanation_widget.append(message)
        self.main_window.graph_widget.update()
        
        return False, message, self._get_state(), 'main_loop'

    def _prepare_next_neighbor(self):
        """Подготавливает обработку следующего соседа"""
        if not self.current_neighbors:
            return self.next_step()
            
        self.current_neighbor = self.current_neighbors.pop(0)
        # Устанавливаем текущее ребро для подсветки
        self.main_window.graph_widget.bfs_current_edge = (self.current_vertex, self.current_neighbor)
        edge_data = self.main_window.graph_widget.graph.get_edge_data(self.current_vertex, self.current_neighbor)
        weight = edge_data.get('weight', 1)
        self.current_distance = self.distances[self.current_vertex] + weight
        message = [
            f"Рассматриваем путь до вершины {self.current_neighbor} через {self.current_vertex}:",
            f"Текущее расстояние до {self.current_neighbor}: {self.distances[self.current_neighbor] if self.distances[self.current_neighbor] != float('-inf') else '-∞'}",
            f"Новое расстояние через {self.current_vertex}: {self.current_distance}"
        ]
        self.comparison_step = True
        self.main_window.explanation_widget.append("\n".join(message))
        self.main_window.graph_widget.update()
        return False, "\n".join(message), self._get_state(), 'neighbor_loop'

    def _process_comparison(self):
        """Обрабатывает шаг сравнения расстояний"""
        # Подсвечиваем строку сравнения
        current_dist_str = str(self.current_distance) if self.current_distance != float('-inf') else '-∞'
        neighbor_dist_str = str(self.distances[self.current_neighbor]) if self.distances[self.current_neighbor] != float('-inf') else '-∞'
        comparison = f"{self.distances[self.current_vertex]}+{self.main_window.graph_widget.graph[self.current_vertex][self.current_neighbor].get('weight', 1)}>{neighbor_dist_str}"
        self.comparison_text[self.current_neighbor] = comparison
        self.main_window.graph_widget.comparison_text = self.comparison_text
        message = [
            f"Сравниваем: {current_dist_str} > {neighbor_dist_str}",
        ]
        if self.current_distance > self.distances[self.current_neighbor]:  # Изменено на >
            old_distance = self.distances[self.current_neighbor]
            self.distances[self.current_neighbor] = self.current_distance
            self.previous[self.current_neighbor] = self.current_vertex
            message.append(f"Обновляем расстояние: {old_distance if old_distance != float('-inf') else '-∞'} → {self.current_distance}")
            self.main_window.graph_widget.distances = self.distances
            self.main_window.graph_widget.update()
        else:
            message.append(f"Оставляем текущее расстояние: {self.distances[self.current_neighbor] if self.distances[self.current_neighbor] != float('-inf') else '-∞'}")
            self.main_window.graph_widget.update()
        self.comparison_text = {}
        self.main_window.graph_widget.comparison_text = {}
        self.comparison_step = False
        self.current_neighbor = None
        self.current_distance = None
        self.main_window.graph_widget.update()
        return False, "\n".join(message), self._get_state(), 'compare'

    def _find_max_distance_vertex(self):
        """Находит вершину с максимальным расстоянием среди непосещенных"""
        max_distance = float('-inf')
        max_vertex = None
        for vertex in self.unvisited:
            if self.distances[vertex] > max_distance:  # Изменено на >
                max_distance = self.distances[vertex]
                max_vertex = vertex
        return max_vertex

    def _finish_algorithm(self):
        """Завершает алгоритм и восстанавливает путь"""
        self.main_window.explanation_widget.append("Восстановление пути:")
        if self.end_vertex is not None:
            if self.previous[self.end_vertex] is None and self.end_vertex != self.current_vertex:
                message = f"Путь до вершины {self.end_vertex} не найден"
                self.main_window.graph_widget.bfs_current = None
                self.main_window.graph_widget.bfs_current_edge = None
                self.main_window.graph_widget.update()
                return True, message, self._get_state(), 'no_path'

            self.shortest_path = self._reconstruct_path()
            if not self.shortest_path:
                message = f"Путь до вершины {self.end_vertex} не найден"
            else:
                path_edges = [(self.shortest_path[i], self.shortest_path[i+1]) 
                            for i in range(len(self.shortest_path)-1)]
                self.main_window.graph_widget.bfs_path = path_edges
                distance = self.distances[self.end_vertex]
                path_str = " → ".join(map(str, self.shortest_path))
                message = f"Найден максимальный путь длиной {distance}:\n{path_str}"
            
            self.main_window.graph_widget.bfs_current = None
            self.main_window.graph_widget.bfs_current_edge = None
            self.main_window.graph_widget.update()
            return True, message, self._get_state(), 'path_recovery'
        return True, "Поиск завершен!", self._get_state(), 'finish'

    def _reconstruct_path(self):
        """Восстанавливает максимальный путь от начальной до конечной вершины"""
        if self.end_vertex is None or self.previous[self.end_vertex] is None:
            return []
            
        path = []
        current = self.end_vertex
        visited = set()  # Для предотвращения циклов
        
        while current is not None and current not in visited:
            path.append(current)
            visited.add(current)
            current = self.previous[current]
            
        if current is None:  # Если дошли до начальной вершины
            return list(reversed(path))
        return []  # Если обнаружен цикл

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
            'path': self.shortest_path,
            'iteration': self.iteration,
            'max_iterations': self.max_iterations
        }

    def get_pseudocode(self):
        return [
            "1. Инициализация:",
            "   distances = {v: -∞ для всех вершин v}  // расстояния до вершин",
            "   previous = {v: null для всех вершин v} // предыдущие вершины",
            "   unvisited = все вершины графа         // непосещенные вершины",
            "   distances[start] = 0                   // расстояние до начальной вершины",
            "   path = []                             // путь до конечной вершины",
            "",
            "2. Основной цикл:",
            "   Пока есть непосещенные вершины:",
            "       v = вершина с max расстоянием среди непосещенных",
            "       Если v не найдена, выход         // нет пути до оставшихся вершин",
            "       Помечаем v как посещенную",
            "",
            "       // Обновляем расстояния до соседей:",
            "       Для каждого соседа u вершины v:",
            "           d = distances[v] + вес ребра (v,u)",
            "           Если d > distances[u]:",
            "               distances[u] = d          // найден более длинный путь",
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
        return "MaxPath (поиск максимального пути)"

    def get_description(self):
        return "Алгоритм поиска максимального пути в графе. Основан на модификации алгоритма Дейкстры." 