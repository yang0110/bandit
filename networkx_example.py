import numpy as np 
import networkx as nx 
from networkx import *
import matplotlib.pyplot as plt 
a=np.reshape(np.random.random_integers(0,1, size=100), (10,10))
g=nx.Graph(a)
nx.draw(g, pos=nx.spring_layout(g))
plt.draw()
plt.show()

a2=np.ones([10,10])
g2=nx.Graph(a2,)
nx.draw(g2,pos=nx.spring_layout(g2))
plt.draw()
plt.show()


