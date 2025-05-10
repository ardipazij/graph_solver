import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QCoreApplication

def setup_platform():
    """Настройка платформо-зависимых параметров Qt"""
    if sys.platform == 'win32':
        # Windows
        os.environ['QT_QPA_PLATFORM'] = 'windows'
    elif sys.platform == 'darwin':
        # macOS
        os.environ['QT_QPA_PLATFORM'] = 'cocoa'
    else:
        # Linux и другие Unix-подобные системы
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
        # Проверка наличия X11
        if not os.environ.get('DISPLAY'):
            os.environ['QT_QPA_PLATFORM'] = 'offscreen'

def main():
    # Настройка платформы перед созданием приложения
    setup_platform()
    
    # Создание приложения
    app = QApplication(sys.argv)
    
    # Установка атрибутов приложения
    QCoreApplication.setOrganizationName("GraphSolver")
    QCoreApplication.setApplicationName("Graph Solver")
    
    # Импорт и создание главного окна
    from widgets.main_window import MainWindow
    window = MainWindow()
    window.show()
    
    # Запуск приложения
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 