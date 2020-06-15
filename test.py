import  numpy as np

mid = np.array([[0],[1],[2]])
tri = np.array([[0,1,2],[2,3,4],[4,5,6]])
print(np.sum(tri > mid, axis=1))

from geometry import triBoxOverlap
box_center = np.array([0, 0, 0])
halfbox_size = np.array([4, 4, 4])
triangle = np.array([[1, 0, 0],
                     [0, 4, 0],
                     [0, 0, 9]])
print(triBoxOverlap(box_center, halfbox_size, triangle))
triangle = np.array([[13, 0, 0],
                     [0, 13, 0],
                     [0, 0, 13]])
print(triBoxOverlap(box_center, halfbox_size, triangle))
triangle = np.array([[11, 0, 0],
                     [0, 11, 0],
                     [0, 0, 11]])
print(triBoxOverlap(box_center, halfbox_size, triangle))
triangle = np.array([[-12, 0, 0],
                     [0, -12, 0],
                     [0, 0, -12]])
print(triBoxOverlap(box_center, halfbox_size, triangle))

triangle = np.array([[5, 5, 5],
                     [4, 5, 6],
                     [6, 5, 4]])
print(triBoxOverlap(box_center, halfbox_size, triangle))

import time
x_set = set()
class X:
    def __init__(self):
        time.sleep(0.1)
        self.pointer = time.time()
        x_set.add(self)

    def __str__(self):
        return str(self.pointer)

x = X()
y = X()
z = X()
print(x in x_set)
for k in x_set:
    print(k)

from Octree import Octree
offset = Octree.get_offset()
print(offset[0 ,0])

v_color = {1: 0, 2: 1}
for i in v_color.items():
    print(i)

# for i in range(8):
#     x, y, z = (i & 4) >> 2, (i & 2) >> 1, (i & 1)
#     print(x, y, z)
#
# for i in range(8):
#     x = i // 4
#     y = (i - x * 4) // 2
#     z = i - x * 4 - y * 2
#     print(x, y, z)

from Octree import GridIndex
x = np.zeros(3)
xx = GridIndex(x)
y = np.array([1,2,3])
yy = GridIndex(y)
zz = yy / 1.5

max_index = np.argmax(yy.id)
print(max_index)
print()
print(xx)
print(yy)
print(zz)

x -= (3 - yy.id) * 0.5 + 2
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