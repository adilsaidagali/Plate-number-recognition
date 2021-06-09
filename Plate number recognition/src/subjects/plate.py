from src.subjects import Subject


class Plate(Subject):
    def __init__(self, x1, y1, x2, y2, score):
        super().__init__(x1, y1, x2, y2, score)
        self._classname = 'PLATE'
