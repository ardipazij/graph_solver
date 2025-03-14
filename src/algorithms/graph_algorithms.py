"""
Модуль, содержащий реализации алгоритмов обхода графа.
"""

import networkx as nx
from PyQt6.QtWidgets import QMessageBox, QInputDialog

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

    def start(self, start_vertex):
        """Начинает обход с указанной вершины"""
        self.reset()
        self.queue = [start_vertex]
        self.visited = {start_vertex}
        self.main_window.explanation_widget.clear()
        self.main_window.explanation_widget.append("Начинаем обход графа в ширину...")
        self.main_window.highlight_pseudocode_line(0)
        self.main_window.graph_widget.visited_vertices = self.visited
        self.main_window.graph_widget.bfs_current = start_vertex
        self.main_window.graph_widget.update()

    def next_step(self):
        """Выполняет следующий шаг алгоритма BFS"""
        if not self.queue:
            self.main_window.explanation_widget.append("Обход завершен!")
            self.main_window.highlight_pseudocode_line(16)
            self.main_window.graph_widget.bfs_current = None
            self.main_window.graph_widget.bfs_current_edge = None
            self.main_window.graph_widget.update()
            return True

        # Извлекаем вершину из очереди
        vertex = self.queue.pop(0)
        self.current = vertex
        self.result.append(vertex)
        self.main_window.graph_widget.bfs_current = vertex
        self.main_window.explanation_widget.append(f"Обрабатываем вершину {vertex}")
        self.main_window.highlight_pseudocode_line(7)
        
        # Получаем соседей в отсортированном порядке
        neighbors = sorted(list(self.main_window.graph_widget.graph.neighbors(vertex)))
        
        # Обрабатываем каждого соседа
        for neighbor in neighbors:
            self.main_window.graph_widget.bfs_current_edge = (vertex, neighbor)
            self.main_window.graph_widget.update()
            
            if neighbor not in self.visited:
                self.main_window.explanation_widget.append(
                    f"Найден непосещенный сосед: {neighbor}"
                )
                self.main_window.highlight_pseudocode_line(12)
                self.visited.add(neighbor)
                self.queue.append(neighbor)
                self.path.append((vertex, neighbor))
                self.main_window.graph_widget.visited_vertices = self.visited
                self.main_window.graph_widget.bfs_path = self.path
            else:
                self.main_window.explanation_widget.append(
                    f"Сосед {neighbor} уже был посещен"
                )
        
        self.main_window.graph_widget.bfs_current_edge = None
        self.main_window.graph_widget.update()
        return False

class DFSAlgorithm(GraphAlgorithm):
    """Реализация алгоритма поиска в глубину (DFS)"""
    
    def __init__(self, main_window):
        super().__init__(main_window)
        self.stack = []

    def start(self, start_vertex):
        """Начинает обход с указанной вершины"""
        self.reset()
        self.stack = [start_vertex]
        self.main_window.explanation_widget.clear()
        self.main_window.explanation_widget.append("Начинаем обход графа в глубину...")
        self.main_window.highlight_pseudocode_line(0)
        self.main_window.graph_widget.bfs_current = start_vertex
        self.main_window.graph_widget.update()

    def next_step(self):
        """Выполняет следующий шаг алгоритма DFS"""
        if not self.stack:
            self.main_window.explanation_widget.append("Обход завершен!")
            self.main_window.highlight_pseudocode_line(16)
            self.main_window.graph_widget.bfs_current = None
            self.main_window.graph_widget.bfs_current_edge = None
            self.main_window.graph_widget.update()
            return True

        # Извлекаем вершину из стека
        vertex = self.stack.pop()
        self.current = vertex
        self.main_window.graph_widget.bfs_current = vertex
        
        if vertex not in self.visited:
            self.visited.add(vertex)
            self.result.append(vertex)
            self.main_window.graph_widget.visited_vertices = self.visited
            self.main_window.explanation_widget.append(f"Обрабатываем вершину {vertex}")
            self.main_window.highlight_pseudocode_line(7)
            
            # Получаем соседей в обратном отсортированном порядке
            neighbors = sorted(list(self.main_window.graph_widget.graph.neighbors(vertex)), reverse=True)
            
            # Обрабатываем каждого соседа
            for neighbor in neighbors:
                self.main_window.graph_widget.bfs_current_edge = (vertex, neighbor)
                self.main_window.graph_widget.update()
                
                if neighbor not in self.visited:
                    self.main_window.explanation_widget.append(
                        f"Найден непосещенный сосед: {neighbor}"
                    )
                    self.main_window.highlight_pseudocode_line(12)
                    self.stack.append(neighbor)
                    self.path.append((vertex, neighbor))
                    self.main_window.graph_widget.bfs_path = self.path
                else:
                    self.main_window.explanation_widget.append(
                        f"Сосед {neighbor} уже был посещен"
                    )
        
        self.main_window.graph_widget.bfs_current_edge = None
        self.main_window.graph_widget.update()
        return False 