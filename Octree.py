# created by chuchong 2020/06/06
# a python version of https://github.com/hjwdzh/Manifold
# Octree

# annotation
# TODO: *
# means it needs to clarify its usage
# the more *, the more important
import numpy as np
from geometry import triBoxOverlap

class GridIndex:

    def __init__(self):
        self.index = np.zeros((3))

    def __init__(self, x, y, z):
        self.index = np.array([x,y,z])

    def __init__(self, index: np.array):
        self.index = index

    def __lt__(self, other):
        return self.index < other.index

    def __le__(self, other):
        return self.index <= other.index

    def __str__(self):
        return self.index.__str__()

    def __add__(self, other):
        return GridIndex(self.index + other.index)

    def __truediv__(self, other):
        return GridIndex(self.index / other)

class Octree:

    def __init__(self):
        self.min_corner = np.zeros(3)
        self.length = np.zeros(3)

        self.level = 0
        self.number = 1
        self.occupied = 1
        self.exterior = 0

        self.children = [None] * 8
        self.connection = [None ] * 6
        self.empty_conn = [None ] * 6
        self.empty_neighbors = []

        self.face_indices = []
        self.face_ind = []

    def __init__(self, min_corner:np.array, max_corner:np.array, faces:np.array, thickness):
        self.__init__()
        self.min_corner = min_corner
        self.length = max_corner - min_corner
        max_ind = np.argmax(self.length)

        # TODO: *
        max_length = self.length[max_ind]
        self.min_corner -= ((max_length - self.length) * 0.5 + thickness * 0.5)
        self.length = np.ones(3) * (max_length + thickness)

        self.face_indeces = faces
        self.face_ind = np.arange(len(self.face_indeces.shape))

        # self.level = 0
        # self.number = 1
        # self.occupied = 1
        # self.exterior = 0


    def __init__(self, min_c, length):
        self.__init__()
        self.min_corner = min_c
        self.length = length

    def isExterior(self, p:np.array):
        # not in bounding box,
        if not np.less(p, self.min_corner + self.length) and np.less(self.min_corner, p):
            return True

        # TODO:*
        if not self.occupied:
            return self.exterior

        if self.level == 0:
            return False

        # repressively use children to find it
        middle_box = self.min_corner + self.length / 2
        inbox_mask = middle_box < p
        index = (int)(np.sum(inbox_mask * np.array([1, 2, 4])))
        return self.children[index].isExterior(p)

    def intersection(self, face_index: int, min_corner: np.array, length: np.array, vertices: np.array):

        halfsize_box = length * 0.5
        box_center = min_corner + halfsize_box

        # TODO *
        # TODO check validity
        # triverts: three
        triverts = vertices[self.face_indices[face_index]]

        return triBoxOverlap(box_center, halfsize_box, triverts)

    def


x = np.zeros(3)
xx = GridIndex(x)
y = np.array([1,2,3])
yy = GridIndex(y)
zz = yy / 1.5

max_index = np.argmax(yy.index)
print(max_index)
print()
print(xx)
print(yy)
print(zz)

x -= (3 - yy.index) * 0.5 + 2
print(x)

p = np.array([1, 1, 1])
middle_box = np.array([1,2,4])
inbox_mask = middle_box <= p
index = np.sum(inbox_mask * np.array([1, 2, 4]))
print(index)

face_indices = np.array([[1,2,3],[4,5,6],[7,8,9]])
vertices = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
face_index = 0
triverts = np.zeros((3, 3))
triverts = vertices[face_indices[face_index]]
print(triverts)