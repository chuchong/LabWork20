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