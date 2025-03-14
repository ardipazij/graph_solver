"""
Модуль, содержащий главное окно приложения.
Управляет всеми компонентами интерфейса и их взаимодействием.
"""

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QFileDialog, QTextEdit, QMessageBox,
                           QCheckBox, QInputDialog, QMenu, QSlider, QGraphicsOpacityEffect)
from PyQt6.QtCore import Qt, QTimer, QDateTime, QPropertyAnimation
import networkx as nx

from widgets.graph_widget import GraphWidget
from algorithms.graph_algorithms import BFSAlgorithm, DFSAlgorithm
from utils.graph_utils import (load_graph_from_file, save_graph_to_file,
                           parse_matrix, create_graph_from_adjacency_matrix,
                           create_graph_from_incidence_matrix, is_weighted)

class MainWindow(QMainWindow):
    """
    Главное окно приложения.
    Содержит все элементы управления и отображения графа.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Визуализатор графов")
        self.setMinimumSize(1000, 600)
        
        # Инициализация компонентов UI
        self.setup_ui()
        
        # Инициализация алгоритмов
        self.bfs_algorithm = BFSAlgorithm(self)
        self.dfs_algorithm = DFSAlgorithm(self)
        
        # Инициализация состояния
        self.is_paused = False
        self.current_delay = 1000  # 1 секунда
        
    def setup_ui(self):
        """Инициализирует все компоненты пользовательского интерфейса"""
        # Создаем центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Создаем главный layout
        main_layout = QHBoxLayout(central_widget)
        
        # Создаем панель инструментов слева
        tools_panel = QWidget()
        tools_layout = QVBoxLayout(tools_panel)
        
        # Создаем кнопки
        self.create_buttons()
        
        # Создаем чекбоксы
        self.create_checkboxes()
        
        # Добавляем все элементы на панель инструментов
        self.setup_tools_panel(tools_layout)
        
        # Создаем виджеты для матрицы и псевдокода
        self.create_matrix_widgets()
        self.create_pseudocode_widget()
        
        # Создаем виджет для пояснений
        self.create_explanation_widget()
        
        # Создаем контроллер скорости
        speed_layout = self.create_speed_controller()
        
        # Создаем виджет графа
        self.graph_widget = GraphWidget(self)
        
        # Добавляем все элементы в главный layout
        self.setup_main_layout(main_layout, tools_panel, speed_layout)
        
        # Подключаем сигналы
        self.connect_signals()

    def create_buttons(self):
        """Создает все кнопки интерфейса"""
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
        self.dfs_action = self.algorithms_menu.addAction("DFS")
        self.algorithms_btn.setMenu(self.algorithms_menu)
        
        # Создаем кнопку справки
        self.help_btn = QPushButton("Справка")
        self.help_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
        """)
        
        # Делаем кнопки переключаемыми
        self.add_vertex_btn.setCheckable(True)
        self.add_edge_btn.setCheckable(True)

    def create_checkboxes(self):
        """Создает чекбоксы для типа графа"""
        self.directed_checkbox = QCheckBox("Ориентированный")
        self.weighted_checkbox = QCheckBox("Взвешенный")

    def setup_tools_panel(self, tools_layout):
        """Настраивает панель инструментов"""
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
        tools_layout.addWidget(self.help_btn)

    def create_matrix_widgets(self):
        """Создает виджеты для работы с матрицами"""
        self.matrix_input = QTextEdit()
        self.matrix_input.setVisible(False)
        self.matrix_input.setPlaceholderText(
            "Введите матрицу (каждый элемент через пробел, строки через перенос строки)\n"
            "Пример:\n1 0 1\n0 1 0\n1 0 1"
        )
        
        self.apply_matrix_btn = QPushButton("Применить матрицу")
        self.apply_matrix_btn.setVisible(False)

    def create_pseudocode_widget(self):
        """Создает виджет для отображения псевдокода"""
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

    def create_explanation_widget(self):
        """Создает виджет для отображения пояснений"""
        self.explanation_widget = QTextEdit()
        self.explanation_widget.setReadOnly(True)
        self.explanation_widget.setVisible(True)
        self.explanation_widget.setMinimumHeight(100)
        self.explanation_widget.setMaximumHeight(150)  # Ограничиваем максимальную высоту
        self.explanation_widget.setStyleSheet("""
            QTextEdit {
                background-color: #f0f0f0;
                border: 2px solid #ccc;
                border-radius: 5px;
                padding: 10px;
                font-family: 'Arial', sans-serif;
                font-size: 14px;
            }
        """)

    def create_speed_controller(self):
        """Создает контроллер скорости"""
        speed_layout = QHBoxLayout()
        speed_label = QLabel("Скорость:")
        
        # Создаем слайдер скорости
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(0)  # 0.25x
        self.speed_slider.setMaximum(4)  # 4x
        self.speed_slider.setValue(2)     # 1x по умолчанию
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.setTickInterval(1)
        self.speed_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #f0f0f0;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4a90e2;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #4a90e2;
                border-radius: 4px;
            }
        """)
        
        # Создаем метку для отображения текущей скорости
        self.speed_value_label = QLabel("1x")
        self.speed_value_label.setMinimumWidth(40)
        self.speed_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Кнопка паузы
        self.pause_btn = QPushButton("⏸")
        self.pause_btn.setCheckable(True)
        self.pause_btn.setFixedSize(30, 30)
        self.pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 15px;
                font-size: 16px;
            }
            QPushButton:checked {
                background-color: #4a90e2;
                color: white;
            }
        """)
        
        # Добавляем все элементы управления скоростью в layout
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.pause_btn)
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_value_label)
        speed_layout.addStretch()
        
        return speed_layout

    def setup_main_layout(self, main_layout, tools_panel, speed_layout):
        """Настраивает главный layout окна"""
        # Создаем контейнер для графа с рамкой
        graph_container = QWidget()
        graph_container.setStyleSheet("""
            QWidget {
                border: 2px solid #4a90e2;
                border-radius: 5px;
                background-color: white;
            }
        """)
        graph_layout = QVBoxLayout(graph_container)
        graph_layout.setContentsMargins(10, 10, 10, 10)
        graph_layout.setSpacing(0)  # Убираем пространство между виджетами
        
        # Создаем контейнер для сообщений алгоритма с фиксированной высотой
        message_container = QWidget()
        message_container.setFixedHeight(80)  # Фиксируем высоту контейнера
        message_container.setStyleSheet("background: transparent;")  # Делаем фон прозрачным
        message_layout = QVBoxLayout(message_container)
        message_layout.setContentsMargins(0, 0, 0, 0)
        
        # Создаем виджет для отображения текущего шага алгоритма
        self.algorithm_step_label = QLabel()
        self.algorithm_step_label.setStyleSheet("""
            QLabel {
                background-color: rgba(74, 144, 226, 0.95);
                color: white;
                padding: 15px;
                border-radius: 8px;
                font-size: 16px;
                font-family: 'Arial', sans-serif;
                font-weight: bold;
                margin: 10px;
            }
        """)
        self.algorithm_step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.algorithm_step_label.setWordWrap(True)
        self.algorithm_step_label.setVisible(False)
        
        # Создаем эффект прозрачности и анимацию
        self.opacity_effect = QGraphicsOpacityEffect()
        self.opacity_effect.setOpacity(1.0)
        self.algorithm_step_label.setGraphicsEffect(self.opacity_effect)
        self.fade_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_animation.setDuration(300)
        
        # Добавляем метку в контейнер сообщений
        message_layout.addWidget(self.algorithm_step_label)
        
        # Добавляем виджеты в layout графа
        graph_layout.addWidget(message_container)
        graph_layout.addWidget(self.graph_widget)
        
        # Добавляем все элементы в главный layout
        main_layout.addWidget(tools_panel)
        
        matrix_layout = QVBoxLayout()
        matrix_layout.addWidget(self.matrix_input)
        matrix_layout.addWidget(self.apply_matrix_btn)
        main_layout.addLayout(matrix_layout)
        
        main_layout.addWidget(self.pseudocode_widget)
        main_layout.addWidget(self.explanation_widget)
        main_layout.addLayout(speed_layout)
        main_layout.addWidget(graph_container)
        
        # Устанавливаем пропорции layout
        main_layout.setStretch(0, 1)  # tools_panel
        main_layout.setStretch(1, 1)  # matrix_layout
        main_layout.setStretch(2, 1)  # pseudocode_widget
        main_layout.setStretch(3, 1)  # explanation_widget
        main_layout.setStretch(4, 0)  # speed_layout
        main_layout.setStretch(5, 4)  # graph_container

    def connect_signals(self):
        """Подключает все сигналы к соответствующим слотам"""
        self.load_file_btn.clicked.connect(self.load_graph_from_file)
        self.save_file_btn.clicked.connect(self.save_graph_to_file)
        self.incidence_matrix_btn.clicked.connect(lambda: self.toggle_matrix_input('incidence'))
        self.adjacency_matrix_btn.clicked.connect(lambda: self.toggle_matrix_input('adjacency'))
        self.add_vertex_btn.clicked.connect(self.start_adding_vertex)
        self.add_edge_btn.clicked.connect(self.start_adding_edge)
        self.apply_matrix_btn.clicked.connect(self.apply_matrix)
        self.bfs_action.triggered.connect(self.start_bfs)
        self.dfs_action.triggered.connect(self.start_dfs)
        self.directed_checkbox.stateChanged.connect(self.on_directed_changed)
        self.weighted_checkbox.stateChanged.connect(self.on_weighted_changed)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.help_btn.clicked.connect(self.show_help)

    def load_graph_from_file(self):
        """Загружает граф из файла"""
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите файл с графом", "", "Text Files (*.txt)")
        if file_name:
            try:
                graph = load_graph_from_file(file_name)
                self.graph_widget.set_graph(graph)
                self.directed_checkbox.setChecked(isinstance(graph, nx.DiGraph))
                self.weighted_checkbox.setChecked(is_weighted(graph))
                self.graph_widget.stop_adding()
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить граф: {str(e)}")

    def save_graph_to_file(self):
        """Сохраняет граф в файл"""
        file_name, _ = QFileDialog.getSaveFileName(self, "Сохранить граф", "", "Text Files (*.txt)")
        if file_name:
            try:
                save_graph_to_file(self.graph_widget.graph, file_name)
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", f"Не удалось сохранить граф: {str(e)}")

    def toggle_matrix_input(self, matrix_type):
        """Переключает отображение поля ввода матрицы"""
        self.matrix_input.setVisible(not self.matrix_input.isVisible())
        self.apply_matrix_btn.setVisible(not self.apply_matrix_btn.isVisible())
        if self.matrix_input.isVisible():
            if matrix_type == 'adjacency':
                if self.directed_checkbox.isChecked():
                    self.matrix_input.setPlaceholderText(
                        "Введите матрицу смежности (каждый элемент через пробел, строки через перенос строки)\n"
                        "Для ориентированного графа матрица может быть несимметричной\n"
                        "Пример для ориентированного графа с 3 вершинами:\n"
                        "0 1 0\n"
                        "0 0 1\n"
                        "1 0 0"
                    )
                else:
                    self.matrix_input.setPlaceholderText(
                        "Введите матрицу смежности (каждый элемент через пробел, строки через перенос строки)\n"
                        "Для неориентированного графа матрица должна быть симметричной\n"
                        "Пример для неориентированного графа с 3 вершинами:\n"
                        "0 1 1\n"
                        "1 0 0\n"
                        "1 0 0"
                    )
            else:  # incidence
                self.matrix_input.setPlaceholderText(
                    "Введите матрицу инцидентности (каждый элемент через пробел, строки через перенос строки)\n"
                    "Для неориентированного графа используйте 1 для связанных вершин\n"
                    "Для ориентированного графа используйте -1 для начала ребра и 1 для конца\n"
                    "Пример для неориентированного графа с 3 вершинами и 2 рёбрами:\n"
                    "1 0\n"
                    "1 1\n"
                    "0 1\n"
                    "\n"
                    "Пример для ориентированного графа с 3 вершинами и 2 рёбрами:\n"
                    "-1 0\n"
                    "1 -1\n"
                    "0 1"
                )
        self.graph_widget.stop_adding()

    def apply_matrix(self):
        """Применяет введенную матрицу для создания графа"""
        try:
            matrix = parse_matrix(self.matrix_input.toPlainText())
            if self.incidence_matrix_btn.isChecked():
                graph = create_graph_from_incidence_matrix(
                    matrix,
                    self.directed_checkbox.isChecked(),
                    self.weighted_checkbox.isChecked()
                )
            else:
                self._validate_adjacency_matrix(matrix)
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

    def _validate_adjacency_matrix(self, matrix):
        """Проверяет корректность матрицы смежности"""
        if len(matrix) != len(matrix[0]):
            raise ValueError("Матрица смежности должна быть квадратной")
        
        for i in range(len(matrix)):
            if matrix[i][i] != 0:
                raise ValueError("На главной диагонали матрицы смежности должны быть только нули")
        
        # Проверяем симметричность только для неориентированного графа
        if not self.directed_checkbox.isChecked():
            for i in range(len(matrix)):
                for j in range(i + 1, len(matrix)):
                    if matrix[i][j] != matrix[j][i]:
                        raise ValueError("Матрица смежности неориентированного графа должна быть симметричной")
        
        # Проверяем неотрицательность элементов
        for row in matrix:
            for elem in row:
                if elem < 0:
                    raise ValueError("Все элементы матрицы смежности должны быть неотрицательными")
                if self.weighted_checkbox.isChecked():
                    if not isinstance(elem, (int, float)):
                        raise ValueError("Для взвешенного графа все элементы должны быть числами")
                else:
                    if elem != 0 and elem != 1:
                        raise ValueError("Для невзвешенного графа допустимы только значения 0 и 1")

    def start_adding_vertex(self):
        """Включает режим добавления вершины"""
        if self.add_vertex_btn.isChecked():
            self.add_edge_btn.setChecked(False)
            self.graph_widget.start_adding_vertex()
        else:
            self.graph_widget.stop_adding()

    def start_adding_edge(self):
        """Включает режим добавления ребра"""
        if self.add_edge_btn.isChecked():
            self.add_vertex_btn.setChecked(False)
            self.graph_widget.start_adding_edge()
        else:
            self.graph_widget.stop_adding()

    def on_directed_changed(self):
        """Обработчик изменения типа графа (ориентированный/неориентированный)"""
        if self.graph_widget.graph.number_of_edges() > 0:
            if self.directed_checkbox.isChecked():
                new_graph = nx.DiGraph()
            else:
                new_graph = nx.Graph()
            
            for edge in self.graph_widget.graph.edges(data=True):
                new_graph.add_edge(edge[0], edge[1], **edge[2])
            
            self.graph_widget.set_graph(new_graph)

    def on_weighted_changed(self):
        """Обработчик изменения типа графа (взвешенный/невзвешенный)"""
        if self.graph_widget.graph.number_of_edges() > 0:
            if isinstance(self.graph_widget.graph, nx.DiGraph):
                new_graph = nx.DiGraph()
            else:
                new_graph = nx.Graph()
            
            for edge in self.graph_widget.graph.edges(data=True):
                if self.weighted_checkbox.isChecked():
                    if 'weight' not in edge[2]:
                        new_graph.add_edge(edge[0], edge[1], weight=1.0)
                    else:
                        new_graph.add_edge(edge[0], edge[1], **edge[2])
                else:
                    new_graph.add_edge(edge[0], edge[1])
            
            self.graph_widget.set_graph(new_graph)

    def show_pseudocode(self, algorithm):
        """Показывает псевдокод выбранного алгоритма"""
        pseudocodes = {
            'BFS': '''Алгоритм BFS (поиск в ширину):
1. Инициализация:
   visited = ∅        // множество посещенных вершин
   queue = [start]    // очередь вершин для обработки
   result = []        // результат обхода

2. Основной цикл:
   Пока queue не пуста:
       vertex = queue.pop(0)    // берем первую вершину из очереди
       result.append(vertex)    // добавляем в результат
       visited.add(vertex)      // помечаем как посещенную

       // Обработка соседей:
       Для каждого соседа neighbor вершины vertex:
           Если neighbor не посещен:
               visited.add(neighbor)    // помечаем как посещенного
               queue.append(neighbor)    // добавляем в очередь

3. Завершение:
   Возвращаем result''',
            
            'DFS': '''Алгоритм DFS (поиск в глубину):
1. Инициализация:
   visited = ∅        // множество посещенных вершин
   stack = [start]    // стек вершин для обработки
   result = []        // результат обхода

2. Основной цикл:
   Пока stack не пуст:
       vertex = stack.pop()     // берем последнюю вершину из стека
       Если vertex не посещена:
           result.append(vertex)    // добавляем в результат
           visited.add(vertex)      // помечаем как посещенную

           // Обработка соседей:
           Для каждого соседа neighbor вершины vertex:
               Если neighbor не посещен:
                   stack.append(neighbor)    // добавляем в стек

3. Завершение:
   Возвращаем result'''
        }
        
        if algorithm in pseudocodes:
            self.pseudocode_widget.setPlainText(pseudocodes[algorithm])
            self.pseudocode_widget.setVisible(True)
            self.current_pseudocode = pseudocodes[algorithm]
            self.current_algorithm = algorithm
        else:
            self.pseudocode_widget.setVisible(False)

    def highlight_pseudocode_line(self, line_number):
        """Подсвечивает указанную строку в псевдокоде"""
        if not hasattr(self, 'current_pseudocode'):
            return
            
        lines = self.current_pseudocode.split('\n')
        highlighted_lines = []
        
        for i, line in enumerate(lines):
            if i == line_number:
                highlighted_lines.append(f'<span style="background-color: #fff3cd;">{line}</span>')
            else:
                highlighted_lines.append(line)
        
        highlighted_text = '<br>'.join(highlighted_lines)
        self.pseudocode_widget.setHtml(highlighted_text)

    def toggle_pause(self):
        """Переключает состояние паузы"""
        self.is_paused = self.pause_btn.isChecked()
        self.pause_btn.setText("▶" if self.is_paused else "⏸")
        
        if self.is_paused:
            if hasattr(self, 'animation_timer') and self.animation_timer.isActive():
                self.animation_timer.stop()
        else:
            if hasattr(self, 'animation_timer'):
                if ((hasattr(self.bfs_algorithm, 'queue') and self.bfs_algorithm.queue) or 
                    (hasattr(self.dfs_algorithm, 'stack') and self.dfs_algorithm.stack)):
                    self.animation_timer.start()

    def on_speed_changed(self, value):
        """Обработчик изменения значения слайдера скорости"""
        speed_multipliers = {
            0: 0.25,  # Очень медленно
            1: 0.5,   # Медленно
            2: 1.0,   # Нормально
            3: 2.0,   # Быстро
            4: 4.0    # Очень быстро
        }
        
        multiplier = speed_multipliers[value]
        self.current_delay = int(1000 / multiplier)
        self.speed_value_label.setText(f"{multiplier}x")
        
        if hasattr(self, 'animation_timer') and self.animation_timer.isActive():
            self.animation_timer.setInterval(self.current_delay)

    def start_bfs(self):
        """Запускает алгоритм BFS"""
        if not self.graph_widget.graph.nodes():
            QMessageBox.warning(self, "Ошибка", "Граф пуст")
            return
            
        self.show_pseudocode('BFS')
        vertices = sorted(list(self.graph_widget.graph.nodes()))
        
        if not vertices:
            return
            
        vertex, ok = QInputDialog.getInt(
            self, 'Выбор начальной вершины',
            'Введите номер начальной вершины:',
            vertices[0], vertices[0], vertices[-1], 1
        )
        
        if ok and vertex in vertices:
            # Создаем и настраиваем таймер анимации
            self.animation_timer = QTimer()
            self.animation_timer.timeout.connect(lambda: self._algorithm_step(self.bfs_algorithm))
            
            # Запускаем алгоритм
            self.bfs_algorithm.start(vertex)
            
            # Устанавливаем начальную скорость
            self.speed_slider.setValue(2)  # 1x
            self.current_delay = 1000
            
            # Сбрасываем состояние паузы
            self.pause_btn.setChecked(False)
            self.pause_btn.setText("⏸")
            self.is_paused = False
            
            # Запускаем анимацию
            self.animation_timer.setInterval(self.current_delay)
            self.animation_timer.start()

    def start_dfs(self):
        """Запускает алгоритм DFS"""
        if not self.graph_widget.graph.nodes():
            QMessageBox.warning(self, "Ошибка", "Граф пуст")
            return
            
        self.show_pseudocode('DFS')
        vertices = sorted(list(self.graph_widget.graph.nodes()))
        
        if not vertices:
            return
            
        vertex, ok = QInputDialog.getInt(
            self, 'Выбор начальной вершины',
            'Введите номер начальной вершины:',
            vertices[0], vertices[0], vertices[-1], 1
        )
        
        if ok and vertex in vertices:
            # Создаем и настраиваем таймер анимации
            self.animation_timer = QTimer()
            self.animation_timer.timeout.connect(lambda: self._algorithm_step(self.dfs_algorithm))
            
            # Запускаем алгоритм
            self.dfs_algorithm.start(vertex)
            
            # Устанавливаем начальную скорость
            self.speed_slider.setValue(2)  # 1x
            self.current_delay = 1000
            
            # Сбрасываем состояние паузы
            self.pause_btn.setChecked(False)
            self.pause_btn.setText("⏸")
            self.is_paused = False
            
            # Запускаем анимацию
            self.animation_timer.setInterval(self.current_delay)
            self.animation_timer.start()

    def _algorithm_step(self, algorithm):
        """Выполняет шаг алгоритма"""
        if not self.is_paused:
            step_info = algorithm.next_step()
            if isinstance(step_info, tuple):
                is_finished, message = step_info
            else:
                is_finished, message = step_info, None
                
            if message:
                # Форматируем сообщение для лучшего отображения
                formatted_message = f"Шаг алгоритма {self.current_algorithm}:\n{message}"
                
                # Показываем сообщение о текущем шаге
                self.algorithm_step_label.setText(formatted_message)
                self.algorithm_step_label.setStyleSheet("""
                    QLabel {
                        background-color: rgba(74, 144, 226, 0.95);
                        color: white;
                        padding: 15px;
                        border-radius: 8px;
                        font-size: 16px;
                        font-family: 'Arial', sans-serif;
                        font-weight: bold;
                        margin: 10px;
                    }
                """)
                
                # Сначала делаем метку видимой и устанавливаем прозрачность
                self.algorithm_step_label.setVisible(True)
                self.opacity_effect.setOpacity(1.0)
                
                # Записываем информацию в лог
                self._log_algorithm_step(message)
                
                # Ждем, пока пользователь прочитает сообщение
                QTimer.singleShot(int(self.current_delay * 1.5), self._fade_out_message)
            
            if is_finished:
                self.animation_timer.stop()
                self._show_completion_message()

    def _fade_out_message(self):
        """Плавно скрывает сообщение о шаге алгоритма"""
        # Запускаем анимацию исчезновения
        self.fade_animation.setStartValue(1.0)
        self.fade_animation.setEndValue(0.0)
        self.fade_animation.start()
        
        # После завершения анимации скрываем метку
        QTimer.singleShot(300, lambda: self.algorithm_step_label.setVisible(False))

    def _show_completion_message(self):
        """Показывает сообщение о завершении алгоритма"""
        completion_message = f"Алгоритм {self.current_algorithm} завершён!"
        self.algorithm_step_label.setText(completion_message)
        self.algorithm_step_label.setStyleSheet("""
            QLabel {
                background-color: rgba(40, 167, 69, 0.95);
                color: white;
                padding: 15px;
                border-radius: 8px;
                font-size: 16px;
                font-family: 'Arial', sans-serif;
                font-weight: bold;
                margin: 10px;
            }
        """)
        self.algorithm_step_label.setVisible(True)
        self.opacity_effect.setOpacity(1.0)
        QTimer.singleShot(2000, self._fade_out_message)

    def _log_algorithm_step(self, message):
        """Записывает информацию о шаге алгоритма в лог-файл"""
        try:
            with open('algorithm_log.txt', 'a', encoding='utf-8') as f:
                timestamp = QDateTime.currentDateTime().toString('yyyy-MM-dd HH:mm:ss')
                algorithm_name = getattr(self, 'current_algorithm', 'Unknown')
                f.write(f'[{timestamp}] {algorithm_name}: {message}\n')
        except Exception as e:
            print(f"Ошибка при записи в лог: {e}")

    def show_help(self):
        """Показывает справочную информацию"""
        help_text = """
<h2>Справка по работе с визуализатором графов</h2>

<h3>Основные функции:</h3>
<ul>
    <li><b>Загрузка и сохранение графа:</b>
        <ul>
            <li>Загрузить граф из файла</li>
            <li>Сохранить текущий граф в файл</li>
        </ul>
    </li>
    <li><b>Создание графа:</b>
        <ul>
            <li>Добавление вершин кликом по свободной области</li>
            <li>Добавление рёбер перетаскиванием от одной вершины к другой</li>
            <li>Ввод матрицы смежности</li>
            <li>Ввод матрицы инцидентности</li>
        </ul>
    </li>
    <li><b>Настройки графа:</b>
        <ul>
            <li>Переключение между ориентированным и неориентированным графом</li>
            <li>Переключение между взвешенным и невзвешенным графом</li>
        </ul>
    </li>
    <li><b>Алгоритмы обхода:</b>
        <ul>
            <li>BFS (поиск в ширину)</li>
            <li>DFS (поиск в глубину)</li>
        </ul>
    </li>
</ul>

<h3>Управление визуализацией:</h3>
<ul>
    <li><b>Скорость анимации:</b> регулируется слайдером от 0.25x до 4x</li>
    <li><b>Пауза/Продолжить:</b> кнопка ⏸/▶</li>
</ul>

<h3>Работа с матрицами:</h3>
<ul>
    <li><b>Матрица смежности:</b>
        <ul>
            <li>Для неориентированного графа: симметричная матрица</li>
            <li>Для ориентированного графа: может быть несимметричной</li>
            <li>Элементы: 0 (нет связи) и 1 (есть связь)</li>
            <li>Для взвешенного графа: вместо 1 указывается вес ребра</li>
        </ul>
    </li>
    <li><b>Матрица инцидентности:</b>
        <ul>
            <li>Для неориентированного графа: 1 - вершина инцидентна ребру</li>
            <li>Для ориентированного графа: -1 (начало ребра) и 1 (конец ребра)</li>
        </ul>
    </li>
</ul>
"""
        msg = QMessageBox(self)
        msg.setWindowTitle("Справка")
        msg.setText(help_text)
        msg.setStyleSheet("""
            QMessageBox {
                background-color: white;
            }
            QMessageBox QLabel {
                min-width: 700px;
            }
        """)
        msg.exec() 