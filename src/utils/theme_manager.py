"""
Модуль для управления темами приложения.
Обеспечивает поддержку светлой и темной темы, а также автоматическое определение системной темы.
"""

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication

class ThemeManager(QObject):
    """Класс для управления темами приложения"""
    
    theme_changed = Signal()  # Сигнал об изменении темы
    
    def __init__(self):
        super().__init__()
        self._is_dark = self._is_system_dark()
        self._colors = self._get_theme_colors()
        
    def _is_system_dark(self) -> bool:
        """Определяет, использует ли система темную тему"""
        app = QApplication.instance()
        if app is None:
            return False
            
        palette = app.palette()
        background = palette.color(QPalette.Window)
        # Если фон темный, считаем что используется темная тема
        return background.lightness() < 128
        
    def _get_theme_colors(self) -> dict:
        """Возвращает цвета для текущей темы"""
        if self._is_dark:
            return {
                'background': '#2b2b2b',
                'foreground': '#ffffff',
                'primary': '#4a90e2',
                'secondary': '#666666',
                'accent': '#2196F3',
                'error': '#f44336',
                'success': '#4CAF50',
                'warning': '#FFC107',
                'vertex': '#4a90e2',
                'vertex_text': '#ffffff',
                'edge': '#666666',
                'edge_text': '#ffffff',
                'visited_vertex': '#4CAF50',
                'current_vertex': '#FFC107',
                'button': '#3d3d3d',
                'button_text': '#ffffff',
                'button_hover': '#4d4d4d',
                'input_background': '#3d3d3d',
                'input_text': '#ffffff',
                'border': '#404040'
            }
        else:
            return {
                'background': '#ffffff',
                'foreground': '#000000',
                'primary': '#4a90e2',
                'secondary': '#666666',
                'accent': '#2196F3',
                'error': '#f44336',
                'success': '#4CAF50',
                'warning': '#FFC107',
                'vertex': '#4a90e2',
                'vertex_text': '#ffffff',
                'edge': '#666666',
                'edge_text': '#000000',
                'visited_vertex': '#4CAF50',
                'current_vertex': '#FFC107',
                'button': '#f0f0f0',
                'button_text': '#000000',
                'button_hover': '#e0e0e0',
                'input_background': '#ffffff',
                'input_text': '#000000',
                'border': '#cccccc'
            }
    
    def get_color(self, color_name: str) -> str:
        """Возвращает цвет по имени"""
        return self._colors.get(color_name, '#000000')
    
    def get_button_style(self) -> str:
        """Возвращает стиль для кнопок"""
        return f"""
            QPushButton {{
                background-color: {self.get_color('button')};
                color: {self.get_color('button_text')};
                border: 1px solid {self.get_color('border')};
                border-radius: 5px;
                padding: 5px;
            }}
            QPushButton:hover {{
                background-color: {self.get_color('button_hover')};
            }}
            QPushButton:checked {{
                background-color: {self.get_color('primary')};
                color: {self.get_color('vertex_text')};
            }}
        """
    
    def get_text_edit_style(self) -> str:
        """Возвращает стиль для текстовых полей"""
        return f"""
            QTextEdit {{
                background-color: {self.get_color('input_background')};
                color: {self.get_color('input_text')};
                border: 1px solid {self.get_color('border')};
                border-radius: 5px;
                padding: 5px;
            }}
        """
    
    def get_checkbox_style(self) -> str:
        """Возвращает стиль для чекбоксов"""
        return f"""
            QCheckBox {{
                color: {self.get_color('foreground')};
            }}
            QCheckBox::indicator {{
                width: 15px;
                height: 15px;
                border: 1px solid {self.get_color('border')};
                border-radius: 3px;
            }}
            QCheckBox::indicator:checked {{
                background-color: {self.get_color('primary')};
            }}
        """
    
    def get_slider_style(self) -> str:
        """Возвращает стиль для слайдеров"""
        return f"""
            QSlider::groove:horizontal {{
                border: 1px solid {self.get_color('border')};
                height: 8px;
                background: {self.get_color('button')};
                margin: 2px 0;
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {self.get_color('primary')};
                border: 1px solid {self.get_color('border')};
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }}
        """
    
    def update_theme(self):
        """Обновляет тему на основе системных настроек"""
        new_is_dark = self._is_system_dark()
        if new_is_dark != self._is_dark:
            self._is_dark = new_is_dark
            self._colors = self._get_theme_colors()
            self.theme_changed.emit()
    
    @property
    def is_dark(self) -> bool:
        """Возвращает True, если используется темная тема"""
        return self._is_dark 