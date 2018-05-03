import matplotlib.pyplot as plt
import pickle
import numpy as np

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

#from Handful_functions import loader


filenames = ["REINFORCE0.npy", "RELAX0.npy", "REINFORCE1.npy", "RELAX1.npy"]

S = [np.load(filename).item() for filename in filenames]
legend = []

for method_dict, filename in zip(S, filenames):
    plt.plot(range(len(method_dict['res'])), method_dict['res'], label="KF-RELAX")
    legend.append(filename.split('.')[0])

plt.legend(legend)
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('reward', fontsize=12)
#plt.show()
plt.savefig('out1.png')
