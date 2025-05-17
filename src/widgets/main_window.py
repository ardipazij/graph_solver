"""
Модуль, содержащий главное окно приложения.
Управляет всеми компонентами интерфейса и их взаимодействием.
"""

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QFileDialog, QTextEdit, QMessageBox,
                           QCheckBox, QInputDialog, QMenu, QSlider, QGraphicsOpacityEffect)
from PySide6.QtCore import Qt, QTimer, QDateTime, QPropertyAnimation
from PySide6.QtGui import QFontMetrics, QFont
import networkx as nx

from widgets.graph_widget import GraphWidget
from algorithms.graph_algorithms import BFSAlgorithm, DFSAlgorithm, DijkstraAlgorithm
from utils.graph_utils import (load_graph_from_file, save_graph_to_file,
                           parse_matrix, create_graph_from_adjacency_matrix,
                           create_graph_from_incidence_matrix, is_weighted,
                           generate_random_graph)

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
        self.dijkstra_algorithm = DijkstraAlgorithm(self)
        
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
        
        # Создаем контроллер скорости
        self.speed_control_widget = self.create_speed_controller()
        
        # Добавляем все элементы на панель инструментов
        self.setup_tools_panel(tools_layout)
        
        # Создаем виджеты для матрицы и псевдокода
        self.create_matrix_widgets()
        self.create_pseudocode_widget()
        
        # Создаем виджет для пояснений
        self.create_explanation_widget()
        
        # Создаем виджет графа
        self.graph_widget = GraphWidget(self)
        
        # Создаем метку для отображения текущего шага алгоритма
        self.algorithm_step_label = QLabel()
        self.algorithm_step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.algorithm_step_label.setWordWrap(True)
        self.algorithm_step_label.setMinimumHeight(40)
        self.algorithm_step_label.setMaximumHeight(120)
        self.algorithm_step_label.setVisible(False)
        
        # Добавляем все элементы в главный layout
        self.setup_main_layout(main_layout, tools_panel)
        
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
        
        # Кнопка очистки графа
        self.clear_graph_btn = QPushButton("Очистить граф")
        self.clear_graph_btn.clicked.connect(self.clear_graph)
        
        # Кнопка "Пояснения" для сворачивания/разворачивания explanation_panel
        self.explanation_toggle_btn = QPushButton("Скрыть пояснения")
        self.explanation_toggle_btn.setCheckable(True)
        self.explanation_toggle_btn.setChecked(False)
        self.explanation_toggle_btn.clicked.connect(self.toggle_explanation_panel)
        
        # Кнопка "Псевдокод" для сворачивания/разворачивания панели работы алгоритма
        self.pseudocode_toggle_btn = QPushButton("Скрыть псевдокод")
        self.pseudocode_toggle_btn.setCheckable(True)
        self.pseudocode_toggle_btn.setChecked(False)
        self.pseudocode_toggle_btn.clicked.connect(self.toggle_pseudocode_panel)
        
        # Создаем кнопку с выпадающим меню для алгоритмов
        self.algorithms_btn = QPushButton("Алгоритмы")
        self.algorithms_menu = QMenu()
        self.bfs_action = self.algorithms_menu.addAction("BFS (поиск в ширину)")
        self.dfs_action = self.algorithms_menu.addAction("DFS (поиск в глубину)")
        self.dijkstra_action = self.algorithms_menu.addAction("Поиск кратчайшего пути")
        self.algorithms_btn.setMenu(self.algorithms_menu)
        
        # Создаем кнопку с выпадающим меню для выбора размещения
        self.layout_btn = QPushButton("Размещение вершин")
        self.layout_menu = QMenu()
        self.circular_action = self.layout_menu.addAction("Круговое")
        self.spring_action = self.layout_menu.addAction("Силовое")
        self.spectral_action = self.layout_menu.addAction("Спектральное")
        self.shell_action = self.layout_menu.addAction("Оболочка")
        self.kamada_kawai_action = self.layout_menu.addAction("Kamada-Kawai")
        self.layout_btn.setMenu(self.layout_menu)
        
        # Создаем кнопку справки
        self.help_btn = QPushButton("Справка")
        
        # Кнопка для генерации случайного графа
        self.generate_btn = QPushButton("Сгенерировать граф")
        
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
        tools_layout.addWidget(self.clear_graph_btn)
        tools_layout.addWidget(self.algorithms_btn)
        tools_layout.addWidget(self.layout_btn)
        tools_layout.addWidget(self.pseudocode_toggle_btn)
        tools_layout.addWidget(self.explanation_toggle_btn)
        tools_layout.addWidget(self.directed_checkbox)
        tools_layout.addWidget(self.weighted_checkbox)
        tools_layout.addWidget(self.generate_btn)
        # --- Контроллер скорости ---
        tools_layout.addWidget(self.speed_control_widget)
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
        """Создает виджеты для отображения псевдокода и состояния переменных"""
        # --- Сворачиваемая панель ---
        self.right_collapsible_panel = QWidget()
        right_panel_layout = QVBoxLayout(self.right_collapsible_panel)
        right_panel_layout.setContentsMargins(0, 0, 0, 0)
        right_panel_layout.setSpacing(2)
        # Кнопку сворачивания убираем отсюда
        # --- Содержимое панели ---
        self.right_panel_content = QWidget()
        right_content_layout = QHBoxLayout(self.right_panel_content)
        right_content_layout.setContentsMargins(0, 0, 0, 0)
        right_content_layout.setSpacing(4)
        # Виджет для псевдокода
        self.pseudocode_widget = QTextEdit()
        self.pseudocode_widget.setReadOnly(True)
        self.pseudocode_widget.setMinimumWidth(400)
        # Виджет для отображения состояния переменных
        self.variables_widget = QTextEdit()
        self.variables_widget.setReadOnly(True)
        self.variables_widget.setMinimumWidth(300)
        right_content_layout.addWidget(self.pseudocode_widget)
        right_content_layout.addWidget(self.variables_widget)
        self.right_panel_content.setLayout(right_content_layout)
        right_panel_layout.addWidget(self.right_panel_content)
        self.pseudocode_container = self.right_collapsible_panel
        self.pseudocode_container.setVisible(True)

    def create_explanation_widget(self):
        """Создает виджет для отображения пояснений и справки"""
        self.explanation_panel = QWidget()
        explanation_layout = QVBoxLayout(self.explanation_panel)
        explanation_layout.setContentsMargins(0, 0, 0, 0)
        explanation_layout.setSpacing(2)
        # Кнопку сворачивания убираем отсюда
        self.explanation_widget = QTextEdit()
        self.explanation_widget.setReadOnly(True)
        self.explanation_widget.setVisible(True)
        self.explanation_widget.setMinimumHeight(100)
        self.explanation_widget.setMaximumHeight(150)  # Ограничиваем максимальную высоту
        explanation_layout.addWidget(self.explanation_widget)
        # --- Плашка справки ---
        self.help_panel = QWidget()
        help_layout = QVBoxLayout(self.help_panel)
        self.help_text = QTextEdit()
        self.help_text.setReadOnly(True)
        self.help_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.help_panel.setVisible(False)
        self.help_toggle_btn = QPushButton("Свернуть")
        self.help_toggle_btn.clicked.connect(self.toggle_help_panel)
        help_layout.addWidget(self.help_text)
        help_layout.addWidget(self.help_toggle_btn, alignment=Qt.AlignmentFlag.AlignRight)

    def create_speed_controller(self):
        """Создает контроллер скорости (вертикально для левой панели)"""
        speed_control_widget = QWidget()
        speed_vlayout = QVBoxLayout(speed_control_widget)
        speed_vlayout.setContentsMargins(0, 10, 0, 10)
        speed_vlayout.setSpacing(6)
        speed_label = QLabel("Скорость:")
        speed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Слайдер скорости
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(0)  # 0.25x
        self.speed_slider.setMaximum(4)  # 4x
        self.speed_slider.setValue(2)     # 1x по умолчанию
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.setTickInterval(1)
        # Метка скорости
        self.speed_value_label = QLabel("1x")
        self.speed_value_label.setMinimumWidth(40)
        self.speed_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Кнопка паузы
        self.pause_btn = QPushButton("⏸")
        self.pause_btn.setCheckable(True)
        self.pause_btn.setFixedSize(30, 30)
        # Добавляем элементы вертикально
        speed_vlayout.addWidget(speed_label)
        speed_vlayout.addWidget(self.speed_slider)
        speed_vlayout.addWidget(self.speed_value_label)
        speed_vlayout.addWidget(self.pause_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        return speed_control_widget

    def setup_main_layout(self, main_layout, tools_panel):
        """Настраивает главный layout окна"""
        # Создаем контейнер для графа с рамкой
        graph_container = QWidget()
        graph_container.setStyleSheet("""
            QWidget {
                border: 2px solid #4a90e2;
                border-radius: 5px;
            }
        """)
        graph_layout = QVBoxLayout(graph_container)
        graph_layout.setContentsMargins(10, 10, 10, 10)
        graph_layout.setSpacing(0)  # Убираем пространство между виджетами
        
        # Создаем контейнер для сообщений алгоритма
        message_container = QWidget()
        message_container.setStyleSheet("background: transparent;")  # Делаем фон прозрачным
        message_container.setMaximumHeight(140)
        message_layout = QVBoxLayout(message_container)
        
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
        
        # Создаем правую панель для псевдокода и состояния
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(self.pseudocode_container)
        right_layout.addWidget(self.explanation_panel)
        right_layout.addWidget(self.help_panel)
        
        # Добавляем все элементы в главный layout
        main_layout.addWidget(tools_panel)
        
        matrix_layout = QVBoxLayout()
        matrix_layout.addWidget(self.matrix_input)
        matrix_layout.addWidget(self.apply_matrix_btn)
        main_layout.addLayout(matrix_layout)
        
        main_layout.addWidget(right_panel)
        main_layout.addWidget(graph_container)
        
        # Устанавливаем пропорции layout
        main_layout.setStretch(0, 1)  # tools_panel
        main_layout.setStretch(1, 1)  # matrix_layout
        main_layout.setStretch(2, 2)  # right_panel (псевдокод + состояние)
        main_layout.setStretch(3, 4)  # graph_container

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
        self.dijkstra_action.triggered.connect(self.start_dijkstra)
        self.directed_checkbox.stateChanged.connect(self.on_directed_changed)
        self.weighted_checkbox.stateChanged.connect(self.on_weighted_changed)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.help_btn.clicked.connect(self.show_help)
        self.generate_btn.clicked.connect(self.generate_random_graph)
        self.clear_graph_btn.clicked.connect(self.clear_graph)
        
        # Подключаем действия для выбора размещения
        self.circular_action.triggered.connect(lambda: self.graph_widget.set_layout('circular'))
        self.spring_action.triggered.connect(lambda: self.graph_widget.set_layout('spring'))
        self.spectral_action.triggered.connect(lambda: self.graph_widget.set_layout('spectral'))
        self.shell_action.triggered.connect(lambda: self.graph_widget.set_layout('shell'))
        self.kamada_kawai_action.triggered.connect(lambda: self.graph_widget.set_layout('kamada_kawai'))

    def load_graph_from_file(self):
        """Загружает граф из файла"""
        try:
            self.graph_widget.reset_visual_state()
            self.variables_widget.clear()
            self.pseudocode_widget.clear()
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Выберите файл с графом",
                "",
                "Текстовые файлы (*.txt);;Все файлы (*.*)"
            )
            
            if not file_name:
                return
                
            # Проверяем, что файл существует и не пустой
            with open(file_name, 'r') as f:
                content = f.read().strip()
                if not content:
                    QMessageBox.warning(self, "Ошибка", "Файл пуст")
                    return
            
            # Загружаем граф
            graph = load_graph_from_file(file_name)
            
            # Обновляем интерфейс
            self.graph_widget.set_graph(graph)
            self.directed_checkbox.setChecked(isinstance(graph, nx.DiGraph))
            self.weighted_checkbox.setChecked(is_weighted(graph))
            self.graph_widget.stop_adding()
            
            # Информируем пользователя об успешной загрузке
            vertices = len(graph.nodes())
            edges = len(graph.edges())
            QMessageBox.information(
                self,
                "Успешно",
                f"Граф загружен:\n- Вершин: {vertices}\n- Рёбер: {edges}"
            )
            
        except FileNotFoundError:
            QMessageBox.warning(self, "Ошибка", "Файл не найден")
        except ValueError as e:
            QMessageBox.warning(self, "Ошибка формата", str(e))
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить граф:\n{str(e)}")

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
        self.pseudocode_container.setVisible(True)
        pseudocodes = {
            'BFS': '''Алгоритм BFS (поиск в ширину):
1. Инициализация:
   visited = ∅        // множество посещенных вершин
   queue = [start]    // очередь вершин для обработки
   result = []        // результат обхода
   parent = {}        // словарь для хранения родительских вершин

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
               parent[neighbor] = vertex // запоминаем родителя

3. Завершение:
   Возвращаем result, parent''',
            
            'DFS': '''Алгоритм DFS (поиск в глубину):
1. Инициализация:
   visited = ∅        // множество посещенных вершин
   stack = [start]    // стек вершин для обработки
   result = []        // результат обхода
   parent = {}        // словарь для хранения родительских вершин
   discovery_time = {} // время обнаружения вершин
   finish_time = {}   // время завершения обработки вершин
   time = 0           // текущее время

2. Основной цикл:
   Пока stack не пуст:
       vertex = stack.pop()     // берем последнюю вершину из стека
       Если vertex не посещена:
           time += 1
           discovery_time[vertex] = time  // запоминаем время обнаружения
           result.append(vertex)    // добавляем в результат
           visited.add(vertex)      // помечаем как посещенную

           // Обработка соседей:
           Для каждого соседа neighbor вершины vertex:
               Если neighbor не посещен:
                   stack.append(neighbor)    // добавляем в стек
                   parent[neighbor] = vertex // запоминаем родителя
           time += 1
           finish_time[vertex] = time  // запоминаем время завершения

3. Завершение:
   Возвращаем result, parent, discovery_time, finish_time''',
            
            'Dijkstra': '''Алгоритм поиска кратчайшего пути (Дейкстра):
1. Инициализация:
   distances = {v: ∞ для всех вершин v}  // расстояния до вершин
   previous = {v: null для всех вершин v} // предыдущие вершины
   unvisited = все вершины графа         // непосещенные вершины
   distances[start] = 0                   // расстояние до начальной вершины
   path = []                             // путь до конечной вершины

2. Основной цикл:
   Пока есть непосещенные вершины:
       v = вершина с min расстоянием среди непосещенных
       Если v не найдена, выход         // нет пути до оставшихся вершин
       Помечаем v как посещенную
       
       // Обновляем расстояния до соседей:
       Для каждого соседа u вершины v:
           d = distances[v] + вес ребра (v,u)
           Если d < distances[u]:
               distances[u] = d          // найден более короткий путь
               previous[u] = v           // запоминаем предыдущую вершину
           Иначе:
               # оставляем текущее расстояние

3. Восстановление пути:
   Если previous[end] не null:
       current = end
       Пока current не null:
           path.append(current)
           current = previous[current]
       path.reverse()

4. Завершение:
   Возвращаем distances[end], path'''
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
                # Подсвечиваем только одну строку
                highlighted_lines.append(f'<div style="background-color: #fff3cd; color: #000; font-weight: bold;">{line}</div>')
            else:
                # Остальные строки — обычные
                highlighted_lines.append(f'<div>{line}</div>')
        highlighted_text = ''.join(highlighted_lines)
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
                if (hasattr(self.bfs_algorithm, 'queue') and self.bfs_algorithm.queue) or \
                   (hasattr(self.dfs_algorithm, 'stack') and self.dfs_algorithm.stack) or \
                   (hasattr(self.dijkstra_algorithm, 'distances') and self.dijkstra_algorithm.distances):
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
        self.graph_widget.reset_visual_state()
        self.variables_widget.clear()
        self.pseudocode_widget.clear()
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
            
            # Используем текущее значение скорости
            speed_multipliers = {0: 0.25, 1: 0.5, 2: 1.0, 3: 2.0, 4: 4.0}
            current_multiplier = speed_multipliers[self.speed_slider.value()]
            self.current_delay = int(1000 / current_multiplier)
            
            # Сбрасываем состояние паузы
            self.pause_btn.setChecked(False)
            self.pause_btn.setText("⏸")
            self.is_paused = False
            
            # Запускаем анимацию
            self.animation_timer.setInterval(self.current_delay)
            self.animation_timer.start()

    def start_dfs(self):
        """Запускает алгоритм DFS"""
        self.graph_widget.reset_visual_state()
        self.variables_widget.clear()
        self.pseudocode_widget.clear()
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
            
            # Используем текущее значение скорости
            speed_multipliers = {0: 0.25, 1: 0.5, 2: 1.0, 3: 2.0, 4: 4.0}
            current_multiplier = speed_multipliers[self.speed_slider.value()]
            self.current_delay = int(1000 / current_multiplier)
            
            # Сбрасываем состояние паузы
            self.pause_btn.setChecked(False)
            self.pause_btn.setText("⏸")
            self.is_paused = False
            
            # Запускаем анимацию
            self.animation_timer.setInterval(self.current_delay)
            self.animation_timer.start()

    def start_dijkstra(self):
        """Запускает алгоритм поиска кратчайшего пути"""
        self.graph_widget.reset_visual_state()
        self.variables_widget.clear()
        self.pseudocode_widget.clear()
        if not self.graph_widget.graph.nodes():
            QMessageBox.warning(self, "Ошибка", "Граф пуст")
            return
            
        if not self.weighted_checkbox.isChecked():
            response = QMessageBox.question(
                self,
                "Предупреждение",
                "Граф не взвешенный. Все рёбра будут иметь вес 1. Продолжить?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if response == QMessageBox.StandardButton.No:
                return
            
        self.show_pseudocode('Dijkstra')
        
        # Показываем сообщение о необходимости выбрать начальную вершину
        self.algorithm_step_label.setText("Выберите начальную вершину")
        self.algorithm_step_label.setVisible(True)
        self.opacity_effect.setOpacity(1.0)
        
        # Включаем режим выбора вершины
        self.graph_widget.waiting_for_vertex_selection = True
        self.graph_widget.vertex_selection_callback = self._on_dijkstra_vertex_selected
        self.graph_widget.update()  # Обновляем отображение

    def _on_dijkstra_vertex_selected(self, vertex):
        """Обработчик выбора вершины для алгоритма Дейкстры"""
        if not hasattr(self.dijkstra_algorithm, 'waiting_for_end') or not self.dijkstra_algorithm.waiting_for_end:
            # Первый клик - выбор начальной вершины
            self.graph_widget.dijkstra_start_vertex = vertex
            self.graph_widget.dijkstra_end_vertex = None
            self.graph_widget.update()
            # Создаем и настраиваем таймер анимации
            self.animation_timer = QTimer()
            self.animation_timer.timeout.connect(lambda: self._algorithm_step(self.dijkstra_algorithm))
            # Запускаем алгоритм
            message = self.dijkstra_algorithm.start(vertex)
            self.algorithm_step_label.setText(message)
            # Используем текущее значение скорости
            speed_multipliers = {0: 0.25, 1: 0.5, 2: 1.0, 3: 2.0, 4: 4.0}
            current_multiplier = speed_multipliers[self.speed_slider.value()]
            self.current_delay = int(1000 / current_multiplier)
            # Сбрасываем состояние паузы
            self.pause_btn.setChecked(False)
            self.pause_btn.setText("⏸")
            self.is_paused = False
        else:
            # Второй клик - выбор конечной вершины
            self.graph_widget.dijkstra_end_vertex = vertex
            self.graph_widget.update()
            message = self.dijkstra_algorithm.set_end_vertex(vertex)
            self.algorithm_step_label.setText(message)
            # Запускаем анимацию
            self.animation_timer.setInterval(self.current_delay)
            self.animation_timer.start()

    def _algorithm_step(self, algorithm):
        """Выполняет шаг алгоритма"""
        if not self.is_paused:
            step_info = algorithm.next_step()
            if isinstance(step_info, tuple):
                if len(step_info) == 3:
                    is_finished, message, state = step_info
                else:
                    is_finished, message = step_info
                    state = None
            else:
                is_finished, message, state = step_info, None, None
                
            # Отменяем предыдущий таймер исчезновения, если он был
            if hasattr(self, '_fade_timer'):
                self._fade_timer.stop()
            
            # Формируем и показываем сообщение для каждого шага
            if message:
                # Показываем сообщение
                self.set_algorithm_step_text(message)
                self.algorithm_step_label.setVisible(True)
                self.opacity_effect.setOpacity(1.0)
                
                # Обновляем состояние переменных
                if state:
                    self.update_variables_state(self.current_algorithm, state)
                
                # Записываем информацию в лог
                self._log_algorithm_step(message)
                
                # Создаем новый таймер для исчезновения
                self._fade_timer = QTimer()
                self._fade_timer.setSingleShot(True)
                self._fade_timer.timeout.connect(self._fade_out_message)
                self._fade_timer.start(int(self.current_delay * 1.5))
            
            # Если алгоритм завершен, показываем сообщение о завершении
            if is_finished:
                self.animation_timer.stop()
                completion_message = "Обход завершен!"
                
                # Показываем сообщение о завершении
                self.set_algorithm_step_text(completion_message)
                self.algorithm_step_label.setVisible(True)
                self.opacity_effect.setOpacity(1.0)
                
                # Записываем информацию о завершении в лог
                self._log_algorithm_step("Алгоритм завершён")
                
                # Создаем новый таймер для исчезновения
                if hasattr(self, '_fade_timer'):
                    self._fade_timer.stop()
                self._fade_timer = QTimer()
                self._fade_timer.setSingleShot(True)
                self._fade_timer.timeout.connect(self._fade_out_message)
                self._fade_timer.start(int(self.current_delay * 2))

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
        # Этот метод больше не используется, так как сообщение о завершении
        # теперь отображается в _algorithm_step
        pass

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
        """Показывает справочную информацию в help_panel"""
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
            <li>Dijkstra (поиск кратчайшего пути)</li>
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
        self.help_text.setHtml(help_text)
        self.help_panel.setVisible(True)

    def toggle_help_panel(self):
        """Скрывает или показывает плашку справки"""
        is_visible = self.help_panel.isVisible()
        self.help_panel.setVisible(not is_visible)

    def generate_random_graph(self):
        """Генерирует случайный граф"""
        try:
            self.graph_widget.reset_visual_state()
            self.variables_widget.clear()
            self.pseudocode_widget.clear()
            # Диалог для ввода параметров
            num_vertices, ok = QInputDialog.getInt(
                self,
                "Генерация графа",
                "Количество вершин:",
                5,  # value
                2,  # min
                100,  # max
                1   # step
            )
            if not ok:
                return
                
            num_edges, ok = QInputDialog.getInt(
                self,
                "Генерация графа",
                "Количество рёбер:",
                5,  # value
                1,  # min
                num_vertices * (num_vertices - 1) // 2,  # max
                1   # step
            )
            if not ok:
                return
                
            is_directed = QMessageBox.question(
                self,
                "Генерация графа",
                "Ориентированный граф?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) == QMessageBox.StandardButton.Yes
            
            is_weighted = QMessageBox.question(
                self,
                "Генерация графа",
                "Взвешенный граф?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) == QMessageBox.StandardButton.Yes
            
            # Генерируем граф
            graph = generate_random_graph(
                num_vertices=num_vertices,
                num_edges=num_edges,
                is_directed=is_directed,
                is_weighted=is_weighted
            )
            
            # Обновляем интерфейс
            self.graph_widget.set_graph(graph)
            self.directed_checkbox.setChecked(is_directed)
            self.weighted_checkbox.setChecked(is_weighted)
            self.graph_widget.stop_adding()
            
            # Информируем пользователя
            QMessageBox.information(
                self,
                "Успешно",
                f"Граф сгенерирован:\n- Вершин: {num_vertices}\n- Рёбер: {num_edges}"
            )
            
            # Сбрасываем состояние кнопки генерации
            self.generate_btn.setChecked(False)
            
        except ValueError as e:
            QMessageBox.warning(self, "Ошибка", str(e))

    def update_variables_state(self, algorithm, state):
        """Обновляет отображение состояния переменных"""
        if algorithm == 'BFS':
            visited = state.get('visited', set())
            queue = state.get('queue', [])
            result = state.get('result', [])
            parent = state.get('parent', {})
            
            text = f"""Состояние переменных:
visited = {sorted(list(visited))}
queue = {queue}
result = {result}
parent = {parent}"""
            
        elif algorithm == 'DFS':
            visited = state.get('visited', set())
            stack = state.get('stack', [])
            result = state.get('result', [])
            parent = state.get('parent', {})
            discovery_time = state.get('discovery_time', {})
            finish_time = state.get('finish_time', {})
            
            text = f"""Состояние переменных:
visited = {sorted(list(visited))}
stack = {stack}
result = {result}
parent = {parent}
discovery_time = {discovery_time}
finish_time = {finish_time}"""
            
        elif algorithm == 'Dijkstra':
            distances = state.get('distances', {})
            previous = state.get('previous', {})
            unvisited = state.get('unvisited', set())
            path = state.get('path', [])
            
            text = f"""Состояние переменных:
distances = {distances}
previous = {previous}
unvisited = {sorted(list(unvisited))}
path = {path}"""
            
        self.variables_widget.setPlainText(text) 

    def set_algorithm_step_text(self, text):
        label = self.algorithm_step_label
        max_width = label.width() - 20  # небольшой отступ
        max_height = label.height() - 10  # небольшой отступ по высоте
        font = label.font()
        font_size = 16  # начальный размер
        font.setPointSize(font_size)
        label.setFont(font)
        metrics = QFontMetrics(font)
        # Учитываем и ширину, и высоту
        while (metrics.horizontalAdvance(text) > max_width or
               metrics.boundingRect(0, 0, max_width, 1000, Qt.TextWordWrap, text).height() > max_height) and font_size > 8:
            font_size -= 1
            font.setPointSize(font_size)
            label.setFont(font)
            metrics = QFontMetrics(font)
        label.setText(text) 

    def clear_graph(self):
        """Очищает граф и сбрасывает все визуальные состояния"""
        self.graph_widget.graph.clear()
        self.graph_widget.vertex_positions.clear()
        self.graph_widget.reset_visual_state()
        self.variables_widget.clear()
        self.pseudocode_widget.clear()
        self.graph_widget.update() 

    def toggle_pseudocode_panel(self):
        """Скрывает или показывает панель работы алгоритма (псевдокод и переменные)"""
        is_collapsed = self.pseudocode_toggle_btn.isChecked()
        if is_collapsed:
            self.pseudocode_container.setVisible(False)
            self.pseudocode_toggle_btn.setText("Псевдокод (скрыт)")
        else:
            self.pseudocode_container.setVisible(True)
            self.pseudocode_toggle_btn.setText("Скрыть псевдокод")

    def toggle_explanation_panel(self):
        """Скрывает или показывает explanation_panel (пояснения)"""
        is_collapsed = self.explanation_toggle_btn.isChecked()
        if is_collapsed:
            self.explanation_panel.setVisible(False)
            self.explanation_toggle_btn.setText("Пояснения (скрыты)")
        else:
            self.explanation_panel.setVisible(True)
            self.explanation_toggle_btn.setText("Скрыть пояснения") 