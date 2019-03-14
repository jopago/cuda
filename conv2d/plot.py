import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Computer Modern'

times = pd.read_csv('results/timing.csv',sep=',')

plt.plot(np.log2(times['N']), times['CPU_Time'],label='CPU Time (s)',color='navy')
plt.plot(np.log2(times['N']), times['GPU_Time'],label='GPU Time (s)',color='darkred')

plt.xlabel(r'$\log_{2}(N)$')
plt.legend()

plt.savefig('img/timing_conv2d.png',bbox_inches='tight', pad_inches=0,dpi=1000,figsize=(5,8))
plt.show()

print(times.head())