"""
Модуль, содержащий виджет для визуализации графа.
Обеспечивает отрисовку графа и обработку взаимодействия пользователя.
"""

from PyQt6.QtWidgets import QWidget, QMessageBox, QInputDialog
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QPainter, QPen, QColor
import networkx as nx
import numpy as np
from utils.graph_utils import is_weighted

class GraphWidget(QWidget):
    """
    Виджет для визуализации графа и взаимодействия с ним.
    Поддерживает добавление вершин и рёбер, перемещение вершин и визуализацию алгоритмов.
    """
    
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.graph = nx.Graph()
        self.vertex_positions = {}
        self.selected_vertex = None
        self.dragging = False
        self.adding_edge = False
        self.adding_vertex = False
        self.edge_start = None
        self.visited_vertices = set()  # для подсветки посещенных вершин
        self.bfs_current = None  # текущая обрабатываемая вершина в BFS
        self.bfs_path = []  # путь обхода
        self.bfs_current_edge = None  # текущее рассматриваемое ребро
        self.distances = {}  # расстояния для алгоритма Дейкстры
        self.comparison_text = {}  # текст сравнения для отображения над вершинами
        self.waiting_for_vertex_selection = False  # ожидание выбора вершины
        self.vertex_selection_callback = None  # callback для выбора вершины
        self.layout_type = 'circular'  # тип размещения по умолчанию
        self.setMinimumSize(600, 400)
        self.setMouseTracking(True)
        self.main_window = main_window
        self.last_mouse_pos = None
        self.scale_factor = 1.0
        self.center_offset = QPoint(0, 0)

    def resizeEvent(self, event):
        """Обработчик изменения размера виджета"""
        super().resizeEvent(event)
        if self.graph.nodes():
            self.adjust_layout()

    def adjust_layout(self):
        """Корректирует размещение вершин, чтобы граф помещался в окне"""
        if not self.vertex_positions:
            return

        # Находим границы текущего размещения
        min_x = min(pos.x() for pos in self.vertex_positions.values())
        max_x = max(pos.x() for pos in self.vertex_positions.values())
        min_y = min(pos.y() for pos in self.vertex_positions.values())
        max_y = max(pos.y() for pos in self.vertex_positions.values())

        # Вычисляем размеры графа
        graph_width = max_x - min_x
        graph_height = max_y - min_y

        # Вычисляем масштаб с учетом отступов
        padding = 40  # отступ от краев
        scale_x = (self.width() - 2 * padding) / graph_width if graph_width > 0 else 1
        scale_y = (self.height() - 2 * padding) / graph_height if graph_height > 0 else 1
        self.scale_factor = min(scale_x, scale_y)

        # Вычисляем смещение для центрирования
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        widget_center_x = self.width() / 2
        widget_center_y = self.height() / 2

        # Обновляем позиции всех вершин
        new_positions = {}
        for vertex, pos in self.vertex_positions.items():
            # Центрируем и масштабируем координаты
            new_x = (pos.x() - center_x) * self.scale_factor + widget_center_x
            new_y = (pos.y() - center_y) * self.scale_factor + widget_center_y
            new_positions[vertex] = QPoint(int(new_x), int(new_y))

        self.vertex_positions = new_positions
        self.update()

    def set_layout(self, layout_type):
        """Устанавливает новый тип размещения вершин"""
        self.layout_type = layout_type
        self.apply_layout()

    def apply_layout(self):
        """Применяет текущий алгоритм размещения вершин"""
        if not self.graph.nodes():
            return

        try:
            # Получаем размещение от networkx с оптимизированными параметрами
            if self.layout_type == 'circular':
                pos = nx.circular_layout(self.graph, scale=2.0)
            elif self.layout_type == 'spring':
                pos = nx.spring_layout(
                    self.graph,
                    k=2.0/np.sqrt(len(self.graph.nodes())),
                    iterations=100,
                    scale=2.0,
                    weight=None
                )
            elif self.layout_type == 'spectral':
                if len(self.graph) < 3:  # Для маленьких графов используем круговое размещение
                    pos = nx.circular_layout(self.graph, scale=2.0)
                else:
                    try:
                        pos = nx.spectral_layout(self.graph, scale=2.0)
                    except:
                        pos = nx.spring_layout(self.graph, scale=2.0)
            elif self.layout_type == 'shell':
                if len(self.graph) < 3:
                    pos = nx.circular_layout(self.graph, scale=2.0)
                else:
                    pos = nx.shell_layout(self.graph, scale=2.0)
            elif self.layout_type == 'kamada_kawai':
                if len(self.graph) < 3:
                    pos = nx.circular_layout(self.graph, scale=2.0)
                else:
                    try:
                        dist = dict(nx.shortest_path_length(self.graph))
                        pos = nx.kamada_kawai_layout(
                            self.graph,
                            dist=dist,
                            scale=2.0,
                            weight=None
                        )
                    except:
                        pos = nx.spring_layout(
                            self.graph,
                            k=2.0/np.sqrt(len(self.graph.nodes())),
                            iterations=100,
                            scale=2.0
                        )
            else:
                pos = nx.circular_layout(self.graph, scale=2.0)

            # Преобразуем координаты в QPoint с учетом отступов
            padding = 60  # увеличиваем отступ от краев
            width = self.width() - 2 * padding
            height = self.height() - 2 * padding
            center = QPoint(self.width() // 2, self.height() // 2)

            # Нормализуем координаты
            coords = np.array(list(pos.values()))
            min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
            min_y, max_y = coords[:, 1].min(), coords[:, 1].max()
            
            scale_x = width / (max_x - min_x) if max_x != min_x else 1
            scale_y = height / (max_y - min_y) if max_y != min_y else 1
            scale = min(scale_x, scale_y)

            self.vertex_positions = {}
            for node, (x, y) in pos.items():
                # Нормализуем и масштабируем координаты
                norm_x = (x - min_x) / (max_x - min_x) if max_x != min_x else 0.5
                norm_y = (y - min_y) / (max_y - min_y) if max_y != min_y else 0.5
                
                # Преобразуем в координаты экрана
                screen_x = int(norm_x * width + padding)
                screen_y = int(norm_y * height + padding)
                
                self.vertex_positions[node] = QPoint(screen_x, screen_y)

        except Exception as e:
            print(f"Ошибка при размещении вершин: {e}")
            # В случае ошибки используем простое круговое размещение
            radius = min(width, height) / 3
            n = len(self.graph.nodes())
            self.vertex_positions = {}
            for i, node in enumerate(self.graph.nodes()):
                angle = 2 * np.pi * i / n
                x = int(center.x() + radius * np.cos(angle))
                y = int(center.y() + radius * np.sin(angle))
                self.vertex_positions[node] = QPoint(x, y)

        self.update()

    def set_graph(self, graph):
        """Устанавливает новый граф и применяет текущий алгоритм размещения"""
        if isinstance(graph, nx.Graph) and self.main_window.directed_checkbox.isChecked():
            self.graph = nx.DiGraph()
            for edge in graph.edges(data=True):
                self.graph.add_edge(edge[0], edge[1], **edge[2])
        else:
            self.graph = nx.Graph()
            for edge in graph.edges(data=True):
                self.graph.add_edge(edge[0], edge[1], **edge[2])

        self.apply_layout()

    def paintEvent(self, event):
        """Отрисовывает граф"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Рисуем рёбра
        for edge in self.graph.edges():
            self._draw_edge(painter, edge)
            
        # Рисуем вершины
        for vertex in self.graph.nodes():
            self._draw_vertex(painter, vertex)
            
        # Рисуем временную линию при добавлении ребра
        if self.adding_edge and self.edge_start is not None and self.last_mouse_pos:
            painter.setPen(QPen(Qt.GlobalColor.blue, 2, Qt.PenStyle.DashLine))
            start_pos = self.vertex_positions[self.edge_start]
            painter.drawLine(start_pos, self.last_mouse_pos)

    def _draw_edge(self, painter, edge):
        """Отрисовывает ребро графа"""
        start = self.vertex_positions[edge[0]]
        end = self.vertex_positions[edge[1]]
        
        angle = np.arctan2(end.y() - start.y(), end.x() - start.x())
        start_x = start.x() + 20 * np.cos(angle)
        start_y = start.y() + 20 * np.sin(angle)
        end_x = end.x() - 20 * np.cos(angle)
        end_y = end.y() - 20 * np.sin(angle)
        
        # Проверяем ребро в обоих направлениях
        if edge == self.bfs_current_edge or (edge[1], edge[0]) == self.bfs_current_edge:
            painter.setPen(QPen(QColor(255, 165, 0), 3))
        elif (edge[0], edge[1]) in self.bfs_path or (edge[1], edge[0]) in self.bfs_path:
            painter.setPen(QPen(QColor(144, 238, 144), 3))
        else:
            painter.setPen(QPen(Qt.GlobalColor.black, 2))
        
        painter.drawLine(QPoint(int(start_x), int(start_y)), 
                        QPoint(int(end_x), int(end_y)))
        
        if self.main_window.directed_checkbox.isChecked():
            self._draw_arrow(painter, QPoint(int(end_x), int(end_y)), angle)
            
        if is_weighted(self.graph):
            weight = self.graph[edge[0]][edge[1]].get('weight', '')
            if weight:
                mid_point = QPoint((start.x() + end.x()) // 2, (start.y() + end.y()) // 2)
                painter.drawText(mid_point, str(weight))

    def _draw_arrow(self, painter, end_point, angle):
        """Отрисовывает стрелку для ориентированного графа"""
        arrow_size = 20
        arrow_point1 = QPoint(
            int(end_point.x() - arrow_size * np.cos(angle - np.pi/6)),
            int(end_point.y() - arrow_size * np.sin(angle - np.pi/6))
        )
        arrow_point2 = QPoint(
            int(end_point.x() - arrow_size * np.cos(angle + np.pi/6)),
            int(end_point.y() - arrow_size * np.sin(angle + np.pi/6))
        )
        painter.drawLine(end_point, arrow_point1)
        painter.drawLine(end_point, arrow_point2)

    def _draw_vertex(self, painter, vertex):
        """Отрисовывает вершину графа"""
        pos = self.vertex_positions[vertex]
        
        if self.adding_edge and (vertex == self.edge_start or vertex == self.selected_vertex):
            painter.setPen(QPen(Qt.GlobalColor.red, 2))
        else:
            painter.setPen(QPen(Qt.GlobalColor.black, 2))
        
        if vertex == self.bfs_current:
            painter.setBrush(QColor(255, 165, 0))  # оранжевый
        elif vertex in self.visited_vertices:
            painter.setBrush(QColor(144, 238, 144))  # светло-зеленый
        else:
            painter.setBrush(QColor(200, 200, 200))  # серый
            
        painter.drawEllipse(pos, 20, 20)
        painter.drawText(pos.x() - 5, pos.y() + 5, str(vertex))
        
        # Отображаем расстояние над вершиной
        if vertex in self.distances:
            distance = self.distances[vertex]
            distance_text = "∞" if distance == float('inf') else str(distance)
            painter.drawText(pos.x() - 15, pos.y() - 25, distance_text)
        
        # Отображаем текст сравнения над расстоянием, если есть
        if vertex in self.comparison_text:
            comparison = self.comparison_text[vertex]
            painter.drawText(pos.x() - 20, pos.y() - 40, comparison)

    def mousePressEvent(self, event):
        """Обработчик нажатия кнопки мыши"""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            self.last_mouse_pos = pos
            
            vertex = self.find_vertex_at(pos)
            
            # Проверяем, ожидается ли выбор вершины для алгоритма
            if self.waiting_for_vertex_selection and vertex is not None:
                if self.vertex_selection_callback:
                    self.vertex_selection_callback(vertex)
                return
            
            if self.adding_vertex and self.main_window.add_vertex_btn.isChecked():
                self._add_new_vertex(pos)
            else:
                if self.adding_edge and self.main_window.add_edge_btn.isChecked():
                    self._handle_edge_creation(vertex)
                else:
                    self._handle_vertex_selection(vertex)

    def mouseMoveEvent(self, event):
        """Обработчик перемещения мыши"""
        if self.dragging and self.selected_vertex is not None:
            self.vertex_positions[self.selected_vertex] = event.position().toPoint()
            self.update()
        elif self.adding_edge and self.edge_start is not None:
            self.last_mouse_pos = event.position().toPoint()
            self.selected_vertex = self.find_vertex_at(event.position().toPoint())
            self.update()

    def mouseReleaseEvent(self, event):
        """Обработчик отпускания кнопки мыши"""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.dragging:
                self.dragging = False
                self.selected_vertex = None
            elif self.adding_edge and self.edge_start is not None and self.selected_vertex is not None:
                self._finalize_edge_creation()

    def find_vertex_at(self, pos):
        """Находит вершину в заданной позиции"""
        for vertex, vertex_pos in self.vertex_positions.items():
            if (pos - vertex_pos).manhattanLength() < 20:
                return vertex
        return None

    def _add_new_vertex(self, pos):
        """Добавляет новую вершину в указанной позиции"""
        new_vertex = max(self.graph.nodes()) + 1 if self.graph.nodes() else 0
        self.graph.add_node(new_vertex)
        self.vertex_positions[new_vertex] = pos
        self.update()

    def _handle_edge_creation(self, vertex):
        """Обрабатывает создание ребра"""
        if vertex is not None:
            if self.edge_start is None:
                self.edge_start = vertex
                self.selected_vertex = None
            else:
                self.selected_vertex = vertex
        self.update()

    def _handle_vertex_selection(self, vertex):
        """Обрабатывает выбор вершины"""
        self.selected_vertex = vertex
        if self.selected_vertex is not None:
            self.dragging = True

    def _finalize_edge_creation(self):
        """Завершает создание ребра"""
        if self.selected_vertex != self.edge_start:
            self._handle_edge_addition()
        self.edge_start = None
        self.selected_vertex = None
        self.update()

    def _handle_edge_addition(self):
        """Обрабатывает добавление ребра в граф"""
        if self.graph.has_edge(self.edge_start, self.selected_vertex):
            self._handle_existing_edge()
        else:
            self._add_new_edge()

    def _handle_existing_edge(self):
        """Обрабатывает случай существующего ребра"""
        if self.main_window.directed_checkbox.isChecked():
            self._handle_directed_edge()
        if self.main_window.weighted_checkbox.isChecked():
            self._update_edge_weight()

    def _handle_directed_edge(self):
        """Обрабатывает изменение направления ребра"""
        reply = QMessageBox.question(
            self, 'Изменить направление ребра',
            'Ребро уже существует. Хотите изменить его направление?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            weight = self.graph[self.edge_start][self.selected_vertex].get('weight', None)
            self.graph.remove_edge(self.edge_start, self.selected_vertex)
            if weight is not None:
                self.graph.add_edge(self.selected_vertex, self.edge_start, weight=weight)
            else:
                self.graph.add_edge(self.selected_vertex, self.edge_start)

    def _update_edge_weight(self):
        """Обновляет вес существующего ребра"""
        current_weight = self.graph[self.edge_start][self.selected_vertex].get('weight', 1.0)
        weight, ok = QInputDialog.getDouble(
            self, 'Редактировать вес ребра', 'Введите новый вес ребра:',
            value=current_weight, min=-1000.0, max=1000.0, decimals=2
        )
        if ok:
            self.graph[self.edge_start][self.selected_vertex]['weight'] = weight

    def _add_new_edge(self):
        """Добавляет новое ребро в граф"""
        if self.main_window.weighted_checkbox.isChecked():
            weight, ok = QInputDialog.getDouble(
                self, 'Вес ребра', 'Введите вес ребра:',
                value=1.0, min=-1000.0, max=1000.0, decimals=2
            )
            if ok:
                self.graph.add_edge(self.edge_start, self.selected_vertex, weight=weight)
        else:
            self.graph.add_edge(self.edge_start, self.selected_vertex)

    def start_adding_vertex(self):
        """Включает режим добавления вершины"""
        if self.adding_vertex:
            self.adding_vertex = False
            self.selected_vertex = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.adding_vertex = True
            self.adding_edge = False
            self.selected_vertex = None
            self.setCursor(Qt.CursorShape.CrossCursor)

    def start_adding_edge(self):
        """Включает режим добавления ребра"""
        if self.adding_edge:
            self.adding_edge = False
            self.selected_vertex = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.adding_edge = True
            self.adding_vertex = False
            self.selected_vertex = None
            self.setCursor(Qt.CursorShape.CrossCursor)

    def stop_adding(self):
        """Выключает режимы добавления"""
        self.adding_vertex = False
        self.adding_edge = False
        self.selected_vertex = None
        self.edge_start = None
        self.setCursor(Qt.CursorShape.ArrowCursor) 