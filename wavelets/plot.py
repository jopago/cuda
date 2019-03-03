import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Computer Modern'

times = pd.read_csv('timing.csv',sep=',')

plt.plot(np.log2(times['N']), times['CPU_Time'],label='CPU Time (s)',color='navy')
plt.plot(np.log2(times['N']), times['GPU_Time'],label='GPU Time (s)',color='darkred')

plt.xlabel(r'$\log_2(N)$')
plt.legend()
plt.show()

plt.savefig('timing_wavelets.pdf',bbox_inches='tight', pad_inches=0,dpi=1000)
print(times.head())