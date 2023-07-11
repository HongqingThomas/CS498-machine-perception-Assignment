# import matplotlib.pyplot as plt
# import numpy as np
#
def piecewise_function(x):
    if x < 1:
        return x
    else:
        return 1 + np.log(x)
#
# x = np.linspace(0, 100, 1000)
# y = np.array([piecewise_function(xi) for xi in x])
#
# plt.figure(figsize=(8,4))
# plt.plot(x, y, color='red')
# plt.xlabel('weighted map with large range')
# plt.ylabel('weighted map')
# plt.title('Piecewise Function')
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x, a):
    y = 1/(1 + np.exp(-1 * a *(x-20)))
    return y

x = np.linspace(0, 65, 1000)
plt.figure(figsize=(8,4))
plt.plot(x, sigmoid(x, 0.1), color='blue', linestyle='-', label='a = 0.1')
plt.plot(x, sigmoid(x, 0.5), color='red', linestyle='--', label='a = 0.5')
plt.plot(x, sigmoid(x, 1), color='green', linestyle='-.', label='a = 1')

line_styles = [('blue', '-', 'solid'), ('red','--', 'dashed'), ('green','-.', 'dashed')]
legend_labels = ['a = 0.1', 'a = 0.5', 'a = 1']
handles = [plt.Line2D([], [], color=color, linestyle=ls, label=label) for color, ls, label in line_styles]
plt.legend(handles=handles, labels=legend_labels)

plt.title('depth fusion weight')
plt.show()
