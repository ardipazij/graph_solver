import sys
import os
from PyQt6.QtWidgets import QApplication
from widgets.main_window import MainWindow

if __name__ == '__main__':
    # Устанавливаем переменную окружения для использования X11 вместо Wayland
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 