from typing import Tuple


class Subject:
    class Color:
        REGULAR = (255, 0, 0)
        HIGHLIGHTED = (0, 255, 0)

    _x1: int
    _y1: int
    _x2: int
    _y2: int
    _square: int
    _rect: Tuple[Tuple[int, int], Tuple[int, int]]
    _label_org: Tuple[int, int]
    _classname: str
    _score: int

    def __init__(self, x1: int, y1: int, x2: int, y2: int, score):
        self._x1 = x1
        self._y1 = y1
        self._x2 = x2
        self._y2 = y2
        self._square = abs(x2 - x1) * abs(y2 - y1)
        self._rect = ((x1, y1), (x2, y2))
        self._label_org = (min(x1, x2), min(y1, y2))
        self._score = score

    @property
    def x1(self):
        return self._x1

    @property
    def x2(self):
        return self._x2

    @property
    def y1(self):
        return self._y1

    @property
    def y2(self):
        return self._y2

    @property
    def square(self):
        return self._square

    @property
    def rect(self):
        return self._rect

    @property
    def label_origin(self):
        return self._label_org

    @property
    def classname(self):
        return self._classname

    @property
    def score(self):
        return self._score
