import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QTextEdit, QMessageBox, QCheckBox, QInputDialog,
                           QMenu)
from PyQt6.QtCore import Qt, QPoint, QPointF
from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath
import networkx as nx
import numpy as np
from graph_utils import (parse_matrix, create_graph_from_adjacency_matrix,
                        create_graph_from_incidence_matrix, load_graph_from_file,
                        save_graph_to_file)

def is_weighted(graph):
    """Проверяет, является ли граф взвешенным"""
    return any('weight' in graph[u][v] for u, v in graph.edges())

class GraphWidget(QWidget):
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
        self.setMinimumSize(600, 400)
        self.setMouseTracking(True)
        self.main_window = main_window

    def set_graph(self, graph):
        """Устанавливает новый граф и распределяет вершины по кругу"""
        # Сохраняем текущие позиции вершин
        old_positions = self.vertex_positions.copy()
        
        # Проверяем, нужно ли преобразовать граф в ориентированный
        if isinstance(graph, nx.Graph) and self.main_window.directed_checkbox.isChecked():
            # Создаем новый ориентированный граф и копируем все рёбра с их весами
            self.graph = nx.DiGraph()
            for edge in graph.edges(data=True):
                self.graph.add_edge(edge[0], edge[1], **edge[2])
        else:
            # Создаем новый неориентированный граф и копируем все рёбра с их весами
            self.graph = nx.Graph()
            for edge in graph.edges(data=True):
                self.graph.add_edge(edge[0], edge[1], **edge[2])
            
        n = len(self.graph.nodes())
        if n > 0:
            # Если есть старые позиции, используем их для существующих вершин
            if old_positions:
                for vertex in self.graph.nodes():
                    if vertex in old_positions:
                        self.vertex_positions[vertex] = old_positions[vertex]
                    else:
                        # Для новых вершин распределяем по кругу
                        angle = 2 * np.pi * len(self.vertex_positions) / n
                        radius = min(self.width(), self.height()) * 0.4
                        center = QPoint(self.width() // 2, self.height() // 2)
                        x = center.x() + radius * np.cos(angle)
                        y = center.y() + radius * np.sin(angle)
                        self.vertex_positions[vertex] = QPoint(int(x), int(y))
            else:
                # Если нет старых позиций, распределяем все вершины по кругу
                radius = min(self.width(), self.height()) * 0.4
                center = QPoint(self.width() // 2, self.height() // 2)
                for i, node in enumerate(self.graph.nodes()):
                    angle = 2 * np.pi * i / n
                    x = center.x() + radius * np.cos(angle)
                    y = center.y() + radius * np.sin(angle)
                    self.vertex_positions[node] = QPoint(int(x), int(y))
        self.update()

    def bfs(self, start_vertex):
        """Реализация алгоритма BFS"""
        self.visited_vertices.clear()
        queue = [start_vertex]
        visited = {start_vertex}
        result = []
        
        while queue:
            vertex = queue.pop(0)
            result.append(vertex)
            self.visited_vertices.add(vertex)
            
            # Получаем соседей вершины
            if isinstance(self.graph, nx.DiGraph):
                neighbors = self.graph.successors(vertex)
            else:
                neighbors = self.graph.neighbors(vertex)
                
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return result

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Рисуем рёбра
        for edge in self.graph.edges():
            start = self.vertex_positions[edge[0]]
            end = self.vertex_positions[edge[1]]
            painter.setPen(QPen(Qt.GlobalColor.black, 2))
            
            # Вычисляем точки начала и конца линии с учетом радиуса вершины
            angle = np.arctan2(end.y() - start.y(), end.x() - start.x())
            start_x = start.x() + 20 * np.cos(angle)
            start_y = start.y() + 20 * np.sin(angle)
            end_x = end.x() - 20 * np.cos(angle)
            end_y = end.y() - 20 * np.sin(angle)
            
            # Рисуем линию
            painter.drawLine(QPoint(int(start_x), int(start_y)), 
                           QPoint(int(end_x), int(end_y)))
            
            # Рисуем стрелки для ориентированного графа
            if self.main_window.directed_checkbox.isChecked():
                # Рисуем стрелку
                arrow_size = 20
                arrow_point1 = QPoint(
                    int(end_x - arrow_size * np.cos(angle - np.pi/6)),
                    int(end_y - arrow_size * np.sin(angle - np.pi/6))
                )
                arrow_point2 = QPoint(
                    int(end_x - arrow_size * np.cos(angle + np.pi/6)),
                    int(end_y - arrow_size * np.sin(angle + np.pi/6))
                )
                painter.drawLine(QPoint(int(end_x), int(end_y)), arrow_point1)
                painter.drawLine(QPoint(int(end_x), int(end_y)), arrow_point2)
            
            # Рисуем веса рёбер
            if is_weighted(self.graph):
                weight = self.graph[edge[0]][edge[1]].get('weight', '')
                if weight:
                    mid_point = QPoint((start.x() + end.x()) // 2, (start.y() + end.y()) // 2)
                    painter.drawText(mid_point, str(weight))

        # Рисуем вершины
        for vertex in self.graph.nodes():
            pos = self.vertex_positions[vertex]
            # Устанавливаем красную обводку для выбранных вершин при добавлении ребра
            if self.adding_edge and (vertex == self.edge_start or vertex == self.selected_vertex):
                painter.setPen(QPen(Qt.GlobalColor.red, 2))
            else:
                painter.setPen(QPen(Qt.GlobalColor.black, 2))
            
            # Устанавливаем цвет вершины в зависимости от того, посещена она или нет
            if vertex in self.visited_vertices:
                painter.setBrush(QColor(144, 238, 144))  # светло-зеленый для посещенных
            else:
                painter.setBrush(QColor(200, 200, 200))  # серый для непосещенных
                
            painter.drawEllipse(pos, 20, 20)
            painter.drawText(pos.x() - 5, pos.y() + 5, str(vertex))

        # Рисуем временную линию при добавлении ребра
        if self.adding_edge and self.edge_start is not None:
            painter.setPen(QPen(Qt.GlobalColor.blue, 2, Qt.PenStyle.DashLine))
            start_pos = self.vertex_positions[self.edge_start]
            painter.drawLine(start_pos, self.last_mouse_pos)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            self.last_mouse_pos = pos
            
            if self.adding_vertex and self.main_window.add_vertex_btn.isChecked():
                # Добавляем новую вершину
                new_vertex = max(self.graph.nodes()) + 1 if self.graph.nodes() else 0
                self.graph.add_node(new_vertex)
                self.vertex_positions[new_vertex] = pos
                self.update()
            else:
                vertex = self.find_vertex_at(pos)
                if self.adding_edge and self.main_window.add_edge_btn.isChecked():
                    if vertex is not None:
                        if self.edge_start is None:
                            self.edge_start = vertex
                            self.selected_vertex = None
                        else:
                            self.selected_vertex = vertex
                    self.update()
                else:
                    self.selected_vertex = vertex
                    if self.selected_vertex is not None:
                        self.dragging = True

    def mouseMoveEvent(self, event):
        if self.dragging and self.selected_vertex is not None:
            self.vertex_positions[self.selected_vertex] = event.position().toPoint()
            self.update()
        elif self.adding_edge and self.edge_start is not None:
            self.last_mouse_pos = event.position().toPoint()
            self.selected_vertex = self.find_vertex_at(event.position().toPoint())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.dragging:
                self.dragging = False
                self.selected_vertex = None
            elif self.adding_edge and self.edge_start is not None and self.selected_vertex is not None:
                if self.selected_vertex != self.edge_start:
                    # Сохраняем все текущие вершины
                    current_vertices = list(self.graph.nodes())
                    
                    # Проверяем, нужно ли преобразовать граф в ориентированный
                    if isinstance(self.graph, nx.Graph) and self.main_window.directed_checkbox.isChecked():
                        # Создаем новый ориентированный граф и копируем все рёбра с их весами
                        new_graph = nx.DiGraph()
                        # Добавляем все вершины
                        for vertex in current_vertices:
                            new_graph.add_node(vertex)
                        # Копируем все рёбра
                        for edge in self.graph.edges(data=True):
                            new_graph.add_edge(edge[0], edge[1], **edge[2])
                        self.graph = new_graph
                    
                    # Проверяем, существует ли уже ребро между этими вершинами
                    if self.graph.has_edge(self.edge_start, self.selected_vertex):
                        # Если ребро существует и граф ориентированный, предлагаем изменить направление
                        if self.main_window.directed_checkbox.isChecked():
                            reply = QMessageBox.question(
                                self, 'Изменить направление ребра',
                                'Ребро уже существует. Хотите изменить его направление?',
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                            )
                            if reply == QMessageBox.StandardButton.Yes:
                                # Сохраняем вес ребра, если он есть
                                weight = self.graph[self.edge_start][self.selected_vertex].get('weight', None)
                                # Удаляем старое ребро
                                self.graph.remove_edge(self.edge_start, self.selected_vertex)
                                # Добавляем новое ребро в обратном направлении
                                if weight is not None:
                                    self.graph.add_edge(self.selected_vertex, self.edge_start, weight=weight)
                                else:
                                    self.graph.add_edge(self.selected_vertex, self.edge_start)
                        # Если граф взвешенный, предлагаем изменить вес
                        if self.main_window.weighted_checkbox.isChecked():
                            current_weight = self.graph[self.edge_start][self.selected_vertex].get('weight', 1.0)
                            weight, ok = QInputDialog.getDouble(
                                self, 'Редактировать вес ребра', 'Введите новый вес ребра:',
                                value=current_weight, min=-1000.0, max=1000.0, decimals=2
                            )
                            if ok:
                                self.graph[self.edge_start][self.selected_vertex]['weight'] = weight
                    else:
                        # Если ребра нет, добавляем новое
                        if self.main_window.weighted_checkbox.isChecked():
                            weight, ok = QInputDialog.getDouble(
                                self, 'Вес ребра', 'Введите вес ребра:',
                                value=1.0, min=-1000.0, max=1000.0, decimals=2
                            )
                            if ok:
                                self.graph.add_edge(self.edge_start, self.selected_vertex, weight=weight)
                        else:
                            self.graph.add_edge(self.edge_start, self.selected_vertex)
                self.edge_start = None
                self.selected_vertex = None
                self.update()

    def find_vertex_at(self, pos):
        for vertex, vertex_pos in self.vertex_positions.items():
            if (pos - vertex_pos).manhattanLength() < 20:
                return vertex
        return None

    def start_adding_vertex(self):
        """Включает/выключает режим добавления вершины"""
        if self.adding_vertex:
            self.adding_vertex = False
            self.selected_vertex = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.adding_vertex = True
            self.adding_edge = False  # Выключаем режим добавления рёбер
            self.selected_vertex = None
            self.setCursor(Qt.CursorShape.CrossCursor)

    def start_adding_edge(self):
        """Включает/выключает режим добавления ребра"""
        if self.adding_edge:
            self.adding_edge = False
            self.selected_vertex = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.adding_edge = True
            self.adding_vertex = False  # Выключаем режим добавления вершин
            self.selected_vertex = None
            self.setCursor(Qt.CursorShape.CrossCursor)

    def stop_adding(self):
        """Выключает режимы добавления"""
        self.adding_vertex = False
        self.adding_edge = False
        self.selected_vertex = None
        self.edge_start = None
        self.setCursor(Qt.CursorShape.ArrowCursor)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Визуализатор графов")
        self.setMinimumSize(1000, 600)  # Увеличиваем минимальную ширину
        
        # Создаем центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Создаем главный layout
        main_layout = QHBoxLayout(central_widget)
        
        # Создаем панель инструментов слева
        tools_panel = QWidget()
        tools_layout = QVBoxLayout(tools_panel)
        
        # Добавляем кнопки
        self.load_file_btn = QPushButton("Загрузить из файла")
        self.save_file_btn = QPushButton("Сохранить в файл")
        self.incidence_matrix_btn = QPushButton("Матрица инцидентности")
        self.adjacency_matrix_btn = QPushButton("Матрица смежности")
        self.add_vertex_btn = QPushButton("Добавить вершину")
        self.add_edge_btn = QPushButton("Добавить ребро")
        
        # Создаем кнопку с выпадающим меню для алгоритмов
        self.algorithms_btn = QPushButton("Алгоритмы обхода")
        self.algorithms_menu = QMenu()
        self.bfs_action = self.algorithms_menu.addAction("BFS")
        self.bfs_action.triggered.connect(self.start_bfs)
        self.algorithms_btn.setMenu(self.algorithms_menu)
        
        # Делаем кнопки переключаемыми
        self.add_vertex_btn.setCheckable(True)
        self.add_edge_btn.setCheckable(True)
        
        # Добавляем чекбоксы для типа графа
        self.directed_checkbox = QCheckBox("Ориентированный")
        self.weighted_checkbox = QCheckBox("Взвешенный")
        
        # Подключаем обработчик изменения состояния чекбокса
        self.directed_checkbox.stateChanged.connect(self.on_directed_changed)
        self.weighted_checkbox.stateChanged.connect(self.on_weighted_changed)
        
        # Добавляем все элементы на панель инструментов
        tools_layout.addWidget(self.load_file_btn)
        tools_layout.addWidget(self.save_file_btn)
        tools_layout.addWidget(self.incidence_matrix_btn)
        tools_layout.addWidget(self.adjacency_matrix_btn)
        tools_layout.addWidget(self.add_vertex_btn)
        tools_layout.addWidget(self.add_edge_btn)
        tools_layout.addWidget(self.algorithms_btn)
        tools_layout.addWidget(self.directed_checkbox)
        tools_layout.addWidget(self.weighted_checkbox)
        tools_layout.addStretch()
        
        # Создаем виджет для ввода матрицы
        self.matrix_input = QTextEdit()
        self.matrix_input.setVisible(False)
        self.matrix_input.setPlaceholderText(
            "Введите матрицу (каждый элемент через пробел, строки через перенос строки)\n"
            "Пример:\n1 0 1\n0 1 0\n1 0 1"
        )
        
        # Создаем кнопку для применения матрицы
        self.apply_matrix_btn = QPushButton("Применить матрицу")
        self.apply_matrix_btn.setVisible(False)
        self.apply_matrix_btn.clicked.connect(self.apply_matrix)
        
        # Создаем виджет для отображения псевдокода
        self.pseudocode_widget = QTextEdit()
        self.pseudocode_widget.setReadOnly(True)
        self.pseudocode_widget.setVisible(False)
        self.pseudocode_widget.setMinimumWidth(300)
        self.pseudocode_widget.setStyleSheet("""
            QTextEdit {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 10px;
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }
        """)
        
        # Создаем виджет графа
        self.graph_widget = GraphWidget(self)
        
        # Добавляем все элементы в главный layout
        main_layout.addWidget(tools_panel)
        matrix_layout = QVBoxLayout()
        matrix_layout.addWidget(self.matrix_input)
        matrix_layout.addWidget(self.apply_matrix_btn)
        main_layout.addLayout(matrix_layout)
        main_layout.addWidget(self.pseudocode_widget)
        main_layout.addWidget(self.graph_widget)
        
        # Подключаем сигналы
        self.load_file_btn.clicked.connect(self.load_graph_from_file)
        self.save_file_btn.clicked.connect(self.save_graph_to_file)
        self.incidence_matrix_btn.clicked.connect(lambda: self.toggle_matrix_input('incidence'))
        self.adjacency_matrix_btn.clicked.connect(lambda: self.toggle_matrix_input('adjacency'))
        self.add_vertex_btn.clicked.connect(self.start_adding_vertex)
        self.add_edge_btn.clicked.connect(self.start_adding_edge)
        
        # Устанавливаем пропорции layout
        main_layout.setStretch(0, 1)  # tools_panel
        main_layout.setStretch(1, 1)  # matrix_layout
        main_layout.setStretch(2, 1)  # pseudocode_widget
        main_layout.setStretch(3, 4)  # graph_widget

    def load_graph_from_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите файл с графом", "", "Text Files (*.txt)")
        if file_name:
            try:
                graph = load_graph_from_file(file_name)
                # Сохраняем текущие позиции вершин
                old_positions = self.graph_widget.vertex_positions.copy()
                
                # Устанавливаем новый граф
                self.graph_widget.set_graph(graph)
                
                # Если есть старые позиции, используем их для существующих вершин
                if old_positions:
                    for vertex in graph.nodes():
                        if vertex in old_positions:
                            self.graph_widget.vertex_positions[vertex] = old_positions[vertex]
                
                self.directed_checkbox.setChecked(isinstance(graph, nx.DiGraph))
                self.weighted_checkbox.setChecked(is_weighted(graph))
                self.graph_widget.stop_adding()
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить граф: {str(e)}")

    def save_graph_to_file(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Сохранить граф", "", "Text Files (*.txt)")
        if file_name:
            try:
                save_graph_to_file(self.graph_widget.graph, file_name)
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", f"Не удалось сохранить граф: {str(e)}")

    def toggle_matrix_input(self, matrix_type):
        self.matrix_input.setVisible(not self.matrix_input.isVisible())
        self.apply_matrix_btn.setVisible(not self.apply_matrix_btn.isVisible())
        if self.matrix_input.isVisible():
            self.matrix_input.setPlaceholderText(
                f"Введите матрицу {matrix_type} (каждый элемент через пробел, строки через перенос строки)\n"
                "Пример:\n1 0 1\n0 1 0\n1 0 1"
            )
        self.graph_widget.stop_adding()

    def apply_matrix(self):
        try:
            matrix = parse_matrix(self.matrix_input.toPlainText())
            if self.incidence_matrix_btn.isChecked():
                graph = create_graph_from_incidence_matrix(
                    matrix,
                    self.directed_checkbox.isChecked(),
                    self.weighted_checkbox.isChecked()
                )
            else:
                # Проверяем, что матрица квадратная
                if len(matrix) != len(matrix[0]):
                    raise ValueError("Матрица смежности должна быть квадратной")
                
                # Проверяем, что на главной диагонали только нули
                for i in range(len(matrix)):
                    if matrix[i][i] != 0:
                        raise ValueError("На главной диагонали матрицы смежности должны быть только нули")
                
                # Проверяем, что матрица симметрична для неориентированного графа
                if not self.directed_checkbox.isChecked():
                    for i in range(len(matrix)):
                        for j in range(i + 1, len(matrix)):
                            if matrix[i][j] != matrix[j][i]:
                                raise ValueError("Матрица смежности неориентированного графа должна быть симметричной")
                
                # Проверяем, что все элементы матрицы неотрицательные
                for row in matrix:
                    for elem in row:
                        if elem < 0:
                            raise ValueError("Все элементы матрицы смежности должны быть неотрицательными")
                
                graph = create_graph_from_adjacency_matrix(
                    matrix,
                    self.directed_checkbox.isChecked(),
                    self.weighted_checkbox.isChecked()
                )
            self.graph_widget.set_graph(graph)
            self.matrix_input.setVisible(False)
            self.apply_matrix_btn.setVisible(False)
            self.graph_widget.stop_adding()
        except ValueError as e:
            QMessageBox.warning(self, "Ошибка в матрице", str(e))
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось создать граф из матрицы: {str(e)}")

    def start_adding_vertex(self):
        """Включает/выключает режим добавления вершины"""
        if self.add_vertex_btn.isChecked():
            self.add_edge_btn.setChecked(False)
            self.graph_widget.start_adding_vertex()
        else:
            self.graph_widget.stop_adding()

    def start_adding_edge(self):
        """Включает/выключает режим добавления ребра"""
        if self.add_edge_btn.isChecked():
            self.add_vertex_btn.setChecked(False)
            self.graph_widget.start_adding_edge()
        else:
            self.graph_widget.stop_adding()

    def on_directed_changed(self):
        """Обработчик изменения состояния чекбокса 'Ориентированный'"""
        if self.graph_widget.graph.number_of_edges() > 0:
            # Создаем новый граф нужного типа
            if self.directed_checkbox.isChecked():
                new_graph = nx.DiGraph()
            else:
                new_graph = nx.Graph()
            
            # Копируем все рёбра с их весами
            for edge in self.graph_widget.graph.edges(data=True):
                new_graph.add_edge(edge[0], edge[1], **edge[2])
            
            # Обновляем граф
            self.graph_widget.set_graph(new_graph)

    def on_weighted_changed(self):
        """Обработчик изменения состояния чекбокса 'Взвешенный'"""
        if self.graph_widget.graph.number_of_edges() > 0:
            # Создаем новый граф того же типа
            if isinstance(self.graph_widget.graph, nx.DiGraph):
                new_graph = nx.DiGraph()
            else:
                new_graph = nx.Graph()
            
            # Копируем все рёбра
            for edge in self.graph_widget.graph.edges(data=True):
                if self.weighted_checkbox.isChecked():
                    # Если включаем веса, добавляем вес 1.0 к рёбрам без веса
                    if 'weight' not in edge[2]:
                        new_graph.add_edge(edge[0], edge[1], weight=1.0)
                    else:
                        new_graph.add_edge(edge[0], edge[1], **edge[2])
                else:
                    # Если выключаем веса, копируем рёбра без весов
                    new_graph.add_edge(edge[0], edge[1])
            
            # Обновляем граф
            self.graph_widget.set_graph(new_graph)

    def show_pseudocode(self, algorithm):
        """Показывает псевдокод выбранного алгоритма"""
        pseudocodes = {
            'BFS': '''BFS(G, start):
    visited = ∅
    queue = [start]
    result = []
    
    while queue is not empty:
        vertex = queue.pop(0)
        result.append(vertex)
        visited.add(vertex)
        
        for neighbor in G.neighbors(vertex):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result'''
        }
        
        if algorithm in pseudocodes:
            self.pseudocode_widget.setPlainText(pseudocodes[algorithm])
            self.pseudocode_widget.setVisible(True)
        else:
            self.pseudocode_widget.setVisible(False)

    def start_bfs(self):
        """Запускает алгоритм BFS"""
        if not self.graph_widget.graph.nodes():
            QMessageBox.warning(self, "Ошибка", "Граф пуст")
            return
            
        # Показываем псевдокод
        self.show_pseudocode('BFS')
            
        # Запрашиваем начальную вершину
        vertices = sorted(list(self.graph_widget.graph.nodes()))
        if not vertices:
            return
            
        vertex, ok = QInputDialog.getInt(
            self, 'Выбор начальной вершины',
            'Введите номер начальной вершины:',
            vertices[0],  # начальное значение
            vertices[0],  # минимальное значение
            vertices[-1],  # максимальное значение
            1  # шаг
        )
        
        if ok and vertex in vertices:
            # Запускаем BFS
            result = self.graph_widget.bfs(vertex)
            QMessageBox.information(
                self, "Результат BFS",
                f"Порядок обхода вершин: {' -> '.join(map(str, result))}"
            )

if __name__ == '__main__':
    # Устанавливаем переменную окружения для использования X11 вместо Wayland
    import os
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 