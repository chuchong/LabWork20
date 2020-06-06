# AABB-triangle overlap
# a translation of by Tomas Akenine-Möller's c++ ver
# chuchong

import numpy as np

'''
to calculate whether the vector that equals v0 v1 edge (x, y, z) and X/Y/Z axis dot
can separate the triangle and box

the dot matrix of (x, y, z) is 
    0   -z  y
    z   0   -x
    -y  x   0

to dot (1, 0, 0) (0, 1, 0) (0, 0, 1) generates normals n respectively
    (0   z   -y)
    (-z 0   x)
    (y  -x  0)

consider the other v2 = (vx', vy', vz'), the projection equals e2 * normals
    zvy'-yvz'
    xvz'-zvx'
    yvx'-xvy'
    
because the midpoint is (0,0,0), its projection is of course always (0,0)
and its radius is (by w: half width, l: half length, h: half height)
    zl - yh
    -zw + xh
    yw - xl

then we consider whether the v in radius, if not , False

e_X represents the dot matrix
radius = e_X [w l h]^T
projection = e_X^T
project_V = projection V^T
project_V_in = -|radius| < project_V < |radius|

if False in (project_V_in 1 or project_V_in 2 or project_V_in 3)
return false
'''

def dotMatrix(e):
    return np.array([[0, -e[2], e[1]],
                     [e[2], 0, -e[0]],
                     [-e[1], e[0], 0]])

def AXISTEST(eX, halfsize, relative_triverts):
    pass

def triBoxOverlap(box_center, halfsize_box, triverts):
    '''

    :param box_center:   np.array((3))
    :param halfsize_box:  np.array(3)
    :param triverts:      np.array(3,3) three vertices
    :return:
    '''
    relateive_triverts = triverts - box_center

    for i in range(3):
        j = (i + 1) % 3

        e = triverts[j] - triverts[i]
        dot_matrix = dotMatrix(e)
        radius = np.fabs(np.dot(dot_matrix, np.transpose(halfsize_box)).reshape(-1,1))
        projection = np.transpose(dot_matrix)
        project_v = np.dot(projection, np.transpose(relateive_triverts))
        in_bool = np.bitwise_or(np.all( - radius > project_v ,axis=1), np.all( project_v > radius, axis=1))
        if in_bool.any():
            return False



    return True

