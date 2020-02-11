# Linear Regression (w=2)

import numpy as np
.............

x = [1., 2., 3., 4.]
y = [2., 4., 6., 8.]
m = n_samples = len(x)

for i in range(-20, 30):
    w = i*0.1
    hypo = ..........   # H(x) = wx
    mse = ..........
    # rmse = np.sqrt(mse)
    print("i:%d, w:%.1f, cost:%f" % (i, i*0.1, mse))

