"""
Модуль, содержащий виджет для визуализации графа.
Обеспечивает отрисовку графа и обработку взаимодействия пользователя.
"""

from PySide6.QtWidgets import QWidget, QMessageBox, QInputDialog
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPainter, QPen, QColor
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
        # Для выделения начальной и конечной вершины кратчайшего пути
        self.dijkstra_start_vertex = None
        self.dijkstra_end_vertex = None

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
        """Применяет выбранный алгоритм размещения"""
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
                    # Создаем группировку вершин для shell_layout
                    nlist = self._create_shell_groups()
                    pos = nx.shell_layout(self.graph, nlist=nlist, scale=2.0)
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

    def _create_shell_groups(self):
        """Создает группы вершин для размещения по оболочкам на основе структуры графа.
        Возвращает список групп вершин для использования в shell_layout."""
        
        nodes = list(self.graph.nodes())
        if len(nodes) <= 2:
            return [nodes]
            
        # Вычисляем степени для всех вершин
        if isinstance(self.graph, nx.DiGraph):
            # Для ориентированного графа учитываем входящие и исходящие ребра
            degree_dict = dict(self.graph.degree())
            in_degree = dict(self.graph.in_degree())
            out_degree = dict(self.graph.out_degree())
            # Комбинированная метрика центральности
            centrality = {n: degree_dict[n] + 0.5 * in_degree[n] + 0.3 * out_degree[n] 
                        for n in nodes}
        else:
            # Для неориентированного графа используем обычную степень
            centrality = dict(self.graph.degree())
        
        # Сортируем вершины по центральности (от высокой к низкой)
        sorted_nodes = sorted(nodes, key=lambda n: centrality[n], reverse=True)
        
        # Определяем количество оболочек адаптивно, избегая хардкодинга
        num_nodes = len(sorted_nodes)
        if num_nodes <= 5:
            # Для маленьких графов используем 1-2 оболочки
            if num_nodes <= 3:
                shells = [sorted_nodes]  # Одна оболочка
            else:
                # Центральная вершина + остальные
                shells = [[sorted_nodes[0]], sorted_nodes[1:]]
        else:
            # Для графов побольше используем логарифмическую шкалу
            import math
            num_shells = min(int(math.log2(num_nodes)) + 1, 5)  # Не более 5 оболочек
            
            shells = []
            # Первая (внутренняя) оболочка - самые центральные вершины
            central_count = max(1, num_nodes // (2 ** num_shells))
            shells.append(sorted_nodes[:central_count])
            
            remaining = sorted_nodes[central_count:]
            if not remaining:
                return shells
                
            # Распределяем оставшиеся вершины по оболочкам
            if num_shells == 2:
                # Для двух оболочек просто добавляем все оставшиеся
                shells.append(remaining)
            else:
                # Для более чем двух оболочек распределяем с возрастающим размером
                chunk_size = len(remaining) // (num_shells - 1)
                if chunk_size == 0:
                    shells.append(remaining)
                else:
                    for i in range(num_shells - 2):
                        start_idx = i * chunk_size
                        end_idx = (i + 1) * chunk_size
                        shells.append(remaining[start_idx:end_idx])
                    # Последняя оболочка забирает все оставшиеся вершины
                    shells.append(remaining[(num_shells - 2) * chunk_size:])
        
        # Убираем пустые оболочки
        shells = [shell for shell in shells if shell]
        return shells

    def set_graph(self, graph):
        """Устанавливает новый граф и применяет текущий алгоритм размещения"""
        # Сохраняем все вершины
        all_vertices = list(graph.nodes())
        
        if isinstance(graph, nx.Graph) and self.main_window.directed_checkbox.isChecked():
            self.graph = nx.DiGraph()
            # Добавляем все вершины
            for vertex in all_vertices:
                self.graph.add_node(vertex)
            # Копируем все рёбра
            for edge in graph.edges(data=True):
                self.graph.add_edge(edge[0], edge[1], **edge[2])
        else:
            self.graph = nx.Graph()
            # Добавляем все вершины
            for vertex in all_vertices:
                self.graph.add_node(vertex)
            # Копируем все рёбра
            for edge in graph.edges(data=True):
                self.graph.add_edge(edge[0], edge[1], **edge[2])

        self.apply_layout()
        # Сброс выделения начальной/конечной вершины
        self.dijkstra_start_vertex = None
        self.dijkstra_end_vertex = None
        self.update()

    def paintEvent(self, event):
        """Отрисовывает граф"""
        from PySide6.QtWidgets import QApplication
        palette = QApplication.palette()
        bg_color = palette.window().color()
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), bg_color)
        self.setPalette(p)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._palette = palette  # сохраняем для использования в других методах
        # Рисуем рёбра
        for edge in self.graph.edges():
            self._draw_edge(painter, edge)
        # Рисуем вершины
        for vertex in self.graph.nodes():
            self._draw_vertex(painter, vertex)
        # Рисуем временную линию при добавлении ребра
        if self.adding_edge and self.edge_start is not None and self.last_mouse_pos:
            painter.setPen(QPen(palette.highlight().color(), 2, Qt.PenStyle.DashLine))
            start_pos = self.vertex_positions[self.edge_start]
            painter.drawLine(start_pos, self.last_mouse_pos)

    def _draw_edge(self, painter, edge):
        """Отрисовывает ребро графа"""
        palette = getattr(self, '_palette', None)
        if palette is None:
            from PySide6.QtWidgets import QApplication
            palette = QApplication.palette()
        start = self.vertex_positions[edge[0]]
        end = self.vertex_positions[edge[1]]
        angle = np.arctan2(end.y() - start.y(), end.x() - start.x())
        start_x = start.x() + 20 * np.cos(angle)
        start_y = start.y() + 20 * np.sin(angle)
        end_x = end.x() - 20 * np.cos(angle)
        end_y = end.y() - 20 * np.sin(angle)
        # Путь всегда зелёный
        if (edge[0], edge[1]) in self.bfs_path or (edge[1], edge[0]) in self.bfs_path:
            painter.setPen(QPen(QColor(0, 180, 0), 3))
        elif edge == self.bfs_current_edge or (edge[1], edge[0]) == self.bfs_current_edge:
            painter.setPen(QPen(palette.highlight().color(), 3))
        else:
            painter.setPen(QPen(palette.windowText().color(), 2))
        painter.drawLine(QPoint(int(start_x), int(start_y)), QPoint(int(end_x), int(end_y)))
        if self.main_window.directed_checkbox.isChecked():
            self._draw_arrow(painter, QPoint(int(end_x), int(end_y)), angle)
        if is_weighted(self.graph):
            weight = self.graph[edge[0]][edge[1]].get('weight', '')
            if weight:
                mid_point = QPoint((start.x() + end.x()) // 2, (start.y() + end.y()) // 2)
                painter.setPen(QPen(palette.windowText().color(), 2))
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
        palette = getattr(self, '_palette', None)
        if palette is None:
            from PySide6.QtWidgets import QApplication
            palette = QApplication.palette()
        pos = self.vertex_positions[vertex]
        # Начальная и конечная вершины
        is_start = False
        is_end = False
        # Для Дейкстры
        if getattr(self, 'dijkstra_start_vertex', None) is not None:
            is_start = (vertex == self.dijkstra_start_vertex)
        if getattr(self, 'dijkstra_end_vertex', None) is not None:
            is_end = (vertex == self.dijkstra_end_vertex)
        # Для BFS/DFS (если нужно)
        if not is_start and hasattr(self, 'bfs_path') and self.bfs_path:
            if vertex == self.bfs_path[0][0]:
                is_start = True
            if vertex == self.bfs_path[-1][1]:
                is_end = True
        if is_start:
            painter.setPen(QPen(palette.windowText().color(), 2))
            painter.setBrush(QColor(255, 215, 0))  # жёлтый
        elif is_end:
            painter.setPen(QPen(palette.windowText().color(), 2))
            painter.setBrush(QColor(220, 0, 0))  # красный
        elif self.adding_edge and (vertex == self.edge_start or vertex == self.selected_vertex):
            painter.setPen(QPen(palette.link().color(), 2))
            painter.setBrush(palette.base().color())
        else:
            painter.setPen(QPen(palette.windowText().color(), 2))
            if vertex == self.bfs_current:
                painter.setBrush(palette.highlight().color())
            elif vertex in self.visited_vertices:
                # Эффект свечения: полупрозрачный круг большего радиуса
                glow_color = QColor(0, 180, 0, 100)  # зелёный с alpha
                painter.setPen(Qt.NoPen)
                painter.setBrush(glow_color)
                painter.drawEllipse(pos, 20 + 10, 20 + 10)
            else:
                painter.setBrush(palette.base().color())
        painter.drawEllipse(pos, 20, 20)
        painter.setPen(QPen(palette.windowText().color(), 2))
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
        # Сброс выделения начальной/конечной вершины
        self.dijkstra_start_vertex = None
        self.dijkstra_end_vertex = None
        # Останавливаем алгоритм при добавлении вершины
        self.main_window.stop_animation()
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
            # Останавливаем алгоритм при изменении направления ребра
            self.main_window.stop_animation()
            self.update()

    def _update_edge_weight(self):
        """Обновляет вес существующего ребра"""
        current_weight = self.graph[self.edge_start][self.selected_vertex].get('weight', 1.0)
        weight, ok = QInputDialog.getDouble(
            self, 'Редактировать вес ребра', 'Введите новый вес ребра:',
            current_weight, -1000.0, 1000.0, 2
        )
        if ok:
            self.graph[self.edge_start][self.selected_vertex]['weight'] = weight
            # Останавливаем алгоритм при изменении веса ребра
            self.main_window.stop_animation()
            self.update()

    def _add_new_edge(self):
        """Добавляет новое ребро в граф"""
        if self.main_window.weighted_checkbox.isChecked():
            weight, ok = QInputDialog.getDouble(
                self, 'Вес ребра', 'Введите вес ребра:',
                1.0, -1000.0, 1000.0, 2
            )
            if ok:
                self.graph.add_edge(self.edge_start, self.selected_vertex, weight=weight)
        else:
            self.graph.add_edge(self.edge_start, self.selected_vertex)
        # Сброс выделения начальной/конечной вершины
        self.dijkstra_start_vertex = None
        self.dijkstra_end_vertex = None
        # Останавливаем алгоритм при добавлении ребра
        self.main_window.stop_animation()
        self.update()

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

    def reset_visual_state(self):
        """Сбрасывает все визуальные состояния графа (цвета, выделения, расстояния, сравнения)"""
        self.visited_vertices.clear()
        self.bfs_current = None
        self.bfs_path.clear()
        self.bfs_current_edge = None
        self.distances = {}
        self.comparison_text = {}
        # Сброс выделения начальной/конечной вершины
        self.dijkstra_start_vertex = None
        self.dijkstra_end_vertex = None
        self.update() 