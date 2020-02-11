import numpy as np

ab = [3, 76]
#ab = [2.5, 76]
#ab = [2.3, 79]

data = [ [2, 81], [4, 93], [6, 91], [8, 97] ]
x = [ ..... ]
y = [ ..... ]

# y = ax + b
def predict(x):
    return ab[0]*x + ab[1]

# RMSE, Root Mean Square Error
def rmse(p, y):
    return np.sqrt( ((p-y)**2).mean() )

# predict value
p_rst = []
for i in range(len(x)):
    p_rst.append(predict(x[i]))
    print("study_hour:%.f, score:%.1f, predict_score:%.1f" % (x[i], y[i], p_rst[i]))

.......



