from Data_sim_Utils import *
import matplotlib.pyplot as plt
import numpy as np

#--------------

T,Y = DRW_sim(0.01, 5, 1)

plt.figure(figsize=(6,2))
plt.plot(T,Y,c='b')
plt.grid()

plt.xlabel("Time / Timescale")
plt.tight_layout()
plt.show()



