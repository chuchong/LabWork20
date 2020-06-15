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

    def __init__(self, *args):
        if len(args) == 0:
            self.id = np.zeros((3))
        elif len(args) == 3:
            self.id = np.array(args)
        elif len(args) == 1:
            self.id = args[0]

    def __lt__(self, other):
        return self.id < other.id

    def __le__(self, other):
        return self.id <= other.id

    def __str__(self):
        return self.id.__str__()

    def __add__(self, other):
        return GridIndex(self.id + other.id)

    def __truediv__(self, other):
        return GridIndex(self.id / other)

class Octree:

    def __init__(self, *args):
        self.min_corner = np.zeros(3)
        self.length = np.zeros(3)

        self.level = 0
        self.number = 1
        self.occupied = 1
        self.exterior = 0

        self.children = [] * 8
        self.connection = [ ] * 6
        self.empty_connection = [ ] * 6
        self.empty_neighbors = []

        self.face_indices = []
        self.face_ind = []

        if len(args) == 4:
            self.__init4__(args[0], args[1], args[2], args[3])
        elif len(args) == 2:
            self.__init2__(args[0], args[1])
        elif len(args) == 3:
            self.__init3__(args[0], args[1], args[2])
        else:
            raise Exception("unsupported initialization")

    def __init4__(self, min_corner:np.array, max_corner:np.array, faces:np.array, thickness):

        self.min_corner = min_corner
        self.length = max_corner - min_corner
        max_ind = np.argmax(self.length)

        # TODO: *
        max_length = self.length[max_ind]
        self.min_corner -= ((max_length - self.length) * 0.5 + thickness * 0.5)
        self.length = np.ones(3) * (max_length + thickness)

        self.face_indices = faces
        self.face_ind = np.arange(len(self.face_indices.shape))

        # self.level = 0
        # self.number = 1
        # self.occupied = 1
        # self.exterior = 0


    def __init2__(self, min_c, length):
        self.min_corner = min_c
        self.length = length

    def __init3__(self, min_c, length, unoccupied):
        self.__init2__(min_c, length)
        self.occupied = 0
        self.number = 0



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
        '''

        :param face_index: face_id 和 face_indices 公用数据变量
        :param min_corner:
        :param length:
        :param vertices:  应该为点集?
        :return:
        '''
        halfsize_box = length * 0.5
        box_center = min_corner + halfsize_box

        # TODO *
        # TODO check validity
        # triverts: three
        triverts = vertices[self.face_indices[face_index]]

        return triBoxOverlap(box_center, halfsize_box, triverts)

    def split(self, vertices):
        self.level += 1
        self.number = 0
        if self.level > 1:# index 大于1, 递归地split
            for i in range(8):
                if self.children[i] and (Octree)(self.children[i]).occupied:
                    (Octree)(self.children[i]).split(vertices)
                    self.number += (Octree)(self.children[i]).number
            self.face_indices = []
            self.face_ind = []
            return
        halfsize = self.length * 0.5
        for ind in range(8): # level 小于等于1的时候,遍历子节点,寻找和mesh面是否相交,据此分配给子点
            k , j , i = ind & 1, (ind & 2) >> 1, (ind & 4) >> 2
            startpoint = self.min_corner + halfsize * np.array([i,j,k])
            self.children[ind] = Octree(startpoint, halfsize, True)

            for face in range(len(self.face_indices)):
                if self.intersection(face, startpoint, halfsize, vertices):
                    self.children[ind].face_indice.push_back(self.face_indices[face])
                    self.children[ind].face_ind.push_back(self.face_ind[face])
                    if  not self.children[ind].occupied:
                        self.children[ind].occupied = 1
                        self.number += 1
                        self.children[ind].number = 1
        self.face_indices = []
        self.face_ind = []

        pass

    def buildConnection(self):
        # 根据子树,递归地建立connection
        if self.level == 0:
            return

        for i in range(8):
            if self.children[i]:
                (Octree)(self.children[i]).buildConnection()

        y_index = [[0, 1, 4, 5], [2, 3, 6, 7]]
        x_index = [[0, 1, 2, 3], [1, 3, 5, 7]]
        z_index = [[0, 1, 2, 3], [4, 5, 6, 7]]
        for i in y_index:
            if self.children[i[0]] and self.children[i[1]]:
                self.connectTree(self.children[i[0]], self.children[i[1]], 2)

        for i in x_index:
            if self.children[i[0]] and self.children[i[1]]:
                self.connectTree(self.children[i[0]], self.children[i[1]], 1)

        for i in z_index:
            if self.children[i[0]] and self.children[i[1]]:
                self.connectTree(self.children[i[0]], self.children[i[1]], 0)

    def _checkAndConnectTree(self, index, l_children, r_children, dim):
        for i in range(len(index[0])):
            if l_children[index[0][i]] and r_children[index[1][i]]:
                self.connectTree(l_children[index[0][i]], r_children[index[1][i]], dim)


    def connectTree(self, l, r, dim):
        # 两个子节点之间建立connection, 步骤都是先对自己建立,再对子节点建立
        if dim == 2:
            l.connection[2] = r
            r.connection[5] = l
            self._checkAndConnectTree([[1, 3, 5, 7], [0, 1, 2 ,3]], l.children, r.children, dim)
        elif dim == 1:
            l.connection[1] = r
            r.connection[4] = l
            self._checkAndConnectTree([[2, 3, 6, 7],[0, 1, 4, 5]], l.children, r.children, dim)
        elif dim == 0:
            l.connection[0] = r
            r.connection[3] = l
            self._checkAndConnectTree([[4,5,6,7], [0,1,2,3]], l.children, r.children, dim)
        pass

    def _connectEmptyTree(self, index, l, r, dim):
        for i in range(len(index[0])):
            self.connectEmptyTree(l[index[0][i]], r[index[0][i]], dim)

    def _connectEmptyTreeWithROccupied(self, index, r_edge, l, r, dim):
        r.empty_connection[r_edge] = l
        if r.level > 0:
            for i in range(len(index)):
                self.connectEmptyTree(l, r.children[index[i]], dim)

    def _connectEmptyTreeWithLOccupied(self, index, r_edge, l, r, dim):
        l.empty_connection[r_edge] = l
        if l.level > 0:
            for i in range(len(index)):
                self.connectEmptyTree(l.children[index[i]], r, dim)


    def connectEmptyTree(self, l, r, dim):
        # 当有节点是空(应该指的是没有三角形, 不需要迭代到其对应子树
        if l.occupird and r.occupied:
            dim2 = [[1, 3, 5, 7], [0, 1, 2, 3]]
            dim1 = [[2, 3, 6, 7], [0, 1, 4, 5]]
            dim0 = [[4, 5, 6, 7], [0, 1, 2, 3]]
            selectTable2Occupied = {0: dim0, 1: dim1, 2: dim2}
            if l.level == 0:
                return
            self._connectEmptyTree(selectTable2Occupied[dim], l.children, r.children, dim)
            return
        elif not (l.occupird or r.occupied):
            l.empty_neighbors.push_back(r)
            r.empty_neighbors.push_back(l)
            return
        elif not l.occupird:
            dim2 = [0, 2, 4, 6]
            dim1 = [0, 1, 4, 5]
            dim0 = [0, 1, 2, 3]
            selectTable1Occupied = {0: (dim0, 3), 1: (dim1, 4), 2: (dim2, 5)}
            self._connectEmptyTreeWithROccupied(selectTable1Occupied[dim][0],
                                                selectTable1Occupied[dim][1],
                                                l,
                                                r,
                                                dim)
        elif not r.occupied:
            dim2 = [1, 3, 5, 7]
            dim1 = [2, 3, 6, 7]
            dim0 = [4, 5, 6, 7]
            selectTable1Occupied = {0: (dim0, 0), 1: (dim1, 1), 2: (dim2, 2)}
            self._connectEmptyTreeWithLOccupied(selectTable1Occupied[dim][0],
                                                selectTable1Occupied[dim][1],
                                                l,
                                                r,
                                                dim)

    def _expandEmpty(self, index, chlidren, empty_list : list, empty_set : set, dim):
        for i in range(len(index)):
            chlidren[index[i]].expandEmpty(empty_list, empty_set, dim)

    def expandEmpty(self, empty_list : list, empty_set : set, dim):
        # 从给定方向搜索子块是否为空的
        if not self.occupied:
            if self not in empty_set:
                empty_set.add(self)
                empty_list.append(self)
            return
        elif self.level == 0:
            return
        else:
            emptyTable = {0: [0, 1, 2, 3],
                          1: [0, 1, 4, 5],
                          2: [0, 2, 4, 6],
                          3: [4, 5, 6, 7],
                          4: [2, 3, 6, 7],
                          5: [1, 3, 5, 7]}
            self._expandEmpty(emptyTable[dim], self.children, empty_list, empty_set, dim)

    def buildEmptyConnection(self):
        if self.level == 0:
            return
        for  i in range(8):
            if self.children[i].occupied:
                self.children[i].builldEmptyConnection()
        pair_x = [0,2,4,6,0,1,4,5,0,1,2,3]
        pair_y = [1,3,5,7,2,3,6,7,4,5,6,7]
        dim= [2,2,2,2,1,1,1,1,0,0,0,0]
        for i in range(12):
             self.connectEmptyTree(self.children[pair_x[i]], self.children[pair_y[i]], dim[i])

    @classmethod
    def _offset(cls):
        '''
        the rectangle vertex of 6 faces
        1 0 0   0 1 0   0 0 1   0 0 0   0 0 0   0 0 0
        1 0 1   1 1 0   0 1 1   0 1 0   0 0 1   1 0 0
        1 1 1   1 1 1   1 1 1   0 1 1   1 0 1   1 1 0
        1 1 0   0 1 1   1 0 1   0 0 1   1 0 0   0 1 0
        if we see from column perspective, it's a mixture of  1111 0011 0110 0000
        :return:
        '''
        p1 = np.array([1, 1, 1, 1]).transpose()
        p2 = np.array([0, 0, 1, 1]).transpose()
        p3 = np.array([0, 1, 1, 0]).transpose()
        p4 = np.array([0, 0, 0, 0]).transpose()
        # then the faces can be implied by cyclic matrix
        f1 = np.column_stack([p1, p2, p3])
        f2 = np.column_stack([p3, p1, p2])
        f3 = np.column_stack([p2, p3, p1])

        f4 = np.column_stack([p4, p3, p2])
        f5 = np.column_stack([p2, p4, p3])
        f6 = np.column_stack([p3, p2, p4])
        return np.stack([f1, f2, f3, f4, f5, f6])

    face_offset = None

    @classmethod
    def get_offset(cls):
        if Octree.face_offset is None:
            Octree.face_offset = Octree._offset()
        return Octree.face_offset

    def constructFace(self, v_color: dict, start : np.array, vertices, faces, v_faces):
        offset = Octree.get_offset()
        if self.level == 0:
            if not self.occupied:
                return
            for i in range(6):
                if self.empty_connection[i] and self.empty_connection[i].exterior:
                    if self.connection[i] and self.connection[i].occupied:
                        raise Exception("connection {} is occupied in construct face phase".format(i))
                    id = np.array([0, 0, 0, 0])
                    for j in range(4):
                        vind = start + offset[i, j]
                        v_id = GridIndex()
                        v_id.id = vind * 2
                        for it in v_color.items():
                            id[j] = it[1]
                            for it1 in self.face_ind:
                                # TODO check validity
                                v_faces[it[1]].add(it1)
                        d = self.min_corner + offset[i, j] * self.length
                        v_color[v_id] = len(vertices)
                        id[j] = len(vertices)
                        vertices.append(d)
                        v_faces.append(set())
                        for ind in self.face_ind:
                            v_faces[id[j]].add(ind)
                    faces.append(id)
            else:
                for i in range(8):
                    if self.children[i] and self.children[i].occupied:
                        x, y, z = i & 4 >> 2, (i & 2) >> 1, (i & 1)
                        nstart = start * 2 + np.array([x, y, z])
                        self.children[i].constructFace(v_color, nstart, vertices, faces, v_faces)
