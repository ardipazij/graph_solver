"""
Главный файл приложения.
Содержит точку входа и инициализацию приложения.
"""

import sys
import platform
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from widgets.main_window import MainWindow

def main():
    # Настройка платформо-зависимых параметров
    if platform.system() == 'Windows':
        # Включаем поддержку высокого DPI на Windows
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    elif platform.system() == 'Linux':
        # Явно указываем использование XCB на Linux
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
    
    # Создаем приложение
    app = QApplication(sys.argv)
    
    # Устанавливаем имя приложения
    app.setApplicationName("Graph Solver")
    
    # Создаем главное окно
    window = MainWindow()
    window.show()
    
    # Запускаем главный цикл приложения
    sys.exit(app.exec())

if __name__ == '__main__':
    main() 