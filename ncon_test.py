import numpy as np
from ncon import ncon
import time


r"""

Test case:

           i   j    k   l
            \ /      \ /
            (T1)    (T2)
            | \ b c /  |
          a |   (T3)   | d
            | e /  \ f |
             \  |  |  /
               (vec)

T1_{a b j i}
T2_{c d l k}
T3_{e f c b}
vec_{a e f d}

"""


if __name__ == '__main__':
    xi = 24
    t1 = np.random.rand(xi, xi, xi, xi)
    t2 = np.random.rand(xi, xi, xi, xi)
    t3 = np.random.rand(xi, xi, xi, xi)
    vec = np.random.rand(xi, xi, xi, xi)

    # non-optimal ordering

    start_time_m1 = time.time()
    m1 = ncon(
        [t1, t2, t3, vec],
        [
            [1, 5, -2, -1],
            [6, 4, -4, -3],
            [2, 3, 6, 5],
            [1, 2, 3, 4],
        ]
    )
    end_time_m1 = time.time()

    # optimal ordering

    start_time_m2 = time.time()
    m2 = ncon(
        [t1, t2, t3, vec],
        [
            [3, 4, -2, -1],
            [5, 6, -4, -3],
            [1, 2, 5, 4],
            [3, 1, 2, 6],
        ]
    )
    end_time_m2 = time.time()

    print(np.allclose(m1, m2))

    print("Time taken for constructing m1:", end_time_m1 - start_time_m1)
    print("Time taken for constructing m2:", end_time_m2 - start_time_m2)
