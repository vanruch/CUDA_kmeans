import pandas as pd
import matplotlib.pyplot as mpl
from mpl_toolkits.mplot3d import Axes3D


a=pd.read_csv('kmeans.csv')

fig=mpl.figure()
ax = fig.add_subplot(111, projection='3d')
mpl.interactive(False)

ax.scatter(xs=a[['X']], ys=a[['Y']], zs=a[['Z']], c=a[['cluster']])
fig.savefig('plot.png')
