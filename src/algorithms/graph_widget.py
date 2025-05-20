from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen, QColor

class GraphWidget(QWidget):
    def __init__(self, graph, vertex_positions, vertex_radius, visited_vertices):
        super().__init__()
        self.graph = graph
        self.vertex_positions = vertex_positions
        self.vertex_radius = vertex_radius
        self.visited_vertices = visited_vertices

    def paintEvent(self, event):
        painter = QPainter(self)
        # Отрисовка рёбер
        for edge in self.graph.edges():
            self._draw_edge(painter, edge)
        # Отрисовка вершин
        for v in self.graph.nodes():
            pos = self.vertex_positions[v]
            if v in self.visited_vertices:
                pen = QPen(QColor(0, 180, 0), 3)
            else:
                pen = QPen(self.palette().windowText().color(), 2)
            painter.setPen(pen)
            painter.setBrush(self.palette().base())
            painter.drawEllipse(pos, self.vertex_radius, self.vertex_radius)

    def _draw_edge(self, painter, edge):
        # Implementation of _draw_edge method
        pass

    def _draw_vertex(self, painter, v):
        # Implementation of _draw_vertex method
        pass 