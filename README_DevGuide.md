# Руководство разработчика: добавление своего алгоритма в визуализатор графов

## Общая структура

1. **Наследуйте класс `GraphAlgorithm`** из `src/algorithms/graph_algorithms.py`.
2. Реализуйте основные методы:
   - `__init__(self, main_window)` — инициализация.
   - `reset(self)` — сброс состояния.
   - `start(self, ...)` — запуск алгоритма (может принимать параметры).
   - `next_step(self)` — выполнение одного шага (для пошаговой анимации).
   - `get_pseudocode(self)` — возвращает список строк псевдокода.
   - `get_highlight_map(self)` — отображение соответствия этапов и строк псевдокода.
   - `get_name(self)` и `get_description(self)` — для UI.

## Пример шаблона

```python
class MyAlgorithm(GraphAlgorithm):
    def __init__(self, main_window):
        super().__init__(main_window)
        # инициализация переменных

    def reset(self):
        super().reset()
        # сбросить свои переменные

    def start(self, start_vertex):
        self.reset()
        # инициализация алгоритма
        return False, "Описание первого шага", self._get_state(), 'init'

    def next_step(self):
        # логика одного шага
        return False, "Описание шага", self._get_state(), 'main_loop'

    def get_pseudocode(self):
        return [
            "1. ...",
            "2. ..."
        ]

    def get_highlight_map(self):
        return {
            'init': 0,
            'main_loop': 1,
            'finish': 2
        }

    def get_name(self):
        return "MyAlgorithm (мой алгоритм)"

    def get_description(self):
        return "Описание вашего алгоритма."
```

## Рекомендации

- Используйте переменные и состояния для пошаговой анимации.
- Для визуализации используйте методы и переменные `graph_widget` (подсветка вершин, рёбер, отображение путей и т.д.).
- Для пояснений используйте `explanation_widget.append(...)`.
- Для псевдокода и подсветки используйте методы `show_pseudocode` и `highlight_pseudocode_line` в `MainWindow`.
- Для новых алгоритмов добавьте пункт в меню и обработчик запуска в `src/widgets/main_window.py`.

## Как добавить алгоритм в интерфейс

1. Импортируйте класс в `src/widgets/main_window.py`.
2. Создайте экземпляр в конструкторе `MainWindow`.
3. Добавьте пункт в меню "Алгоритмы".
4. Реализуйте обработчик запуска (аналогично другим алгоритмам).

---

**Пример:**
- См. реализации `BFSAlgorithm`, `DFSAlgorithm`, `DijkstraAlgorithm`, `BellmanFordAlgorithm`, `MaxPathAlgorithm`, `KruskalAlgorithm`, `PrimAlgorithm` в `src/algorithms/graph_algorithms.py`. 