import numpy as np
arr = [[12,1,2,3,45],[1,2,3,4,5,6,7,8,9]]
x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
print("x_shape", x.shape)
print("x_shape@-1", x.shape[-1])
print("x_shape@1", x.shape[1])
print("x_shape@0", x.shape[0])
