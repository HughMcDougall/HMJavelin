
'''
Practice for graph generation

This DOESNT DO MATHS

HM 15/12
'''

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import daft

#=========================
#INDEX
#=========================
pt1=False
pt2=False


#=========================
#PART ONE: DRAWING SIMPLE GRAPHS
#=========================

if pt1:
    #Make a probabalistic graph model (PGM)
    pgm = daft.PGM(dpi=120)

    #Add some example nodes

    #pgm.add_node(handle, label, (x,y), ratio)
    pgm.add_node("node 1", "Central",   0, 0, aspect=2.5)
    pgm.add_node("node 2", "Secondary", 0, 1, aspect=1)

    pgm.add_edge("node 1", "node 2",     directed=False)

    pgm.render()

    #=========================

    #Edges are directed by default. The following is a _cyclic graph_
    pgm2 = daft.PGM(dpi=120)

    pgm2.add_node("A","Scissors",0,0)
    pgm2.add_node("B","Paper",1,0)
    pgm2.add_node("C","Rock",0,-1)

    pgm2.add_edge("A","B")
    pgm2.add_edge("B","C")
    pgm2.add_edge("C","A")

    pgm2.render()

#=========================
#PART TWO: SIMPLE MODELS
#=========================
if pt2:
    pgm3 = daft.PGM(dpi=120)

    pgm3.add_node("plx", r"$\varpi$", 0, 0, observed=True)
    pgm3.add_node("r",  "$r$", 0, 1)

    pgm3.add_edge("r", "plx")

    pgm3.add_node("L",  "$L$", 0, 2, fixed =True) #Parameter of a prior
    pgm3.add_edge("L", "r")

    pgm3.render();
    #=========================

    #Adding Plates
    '''
    Plates are simple labeled boxes in daft. They have no effect on the nodes and edges
    '''
    pgm4 = daft.PGM(dpi=120)

    pgm4.add_node("plx", r"$\varpi$", 0, 0, observed=True)
    pgm4.add_node("r",  "$r$", 0, 1)

    pgm4.add_edge("r", "plx")
    pgm4.add_plate([-1.25, -0.5, 2.5, 2], label=r"$n = 1...N$", position="bottom left")

    pgm4.render();

#=========================
#PART THRE: AGN Draft
#=========================

sig_spread=1
vert_sep=-1.5
plate_sep = -0.8
group_sep=2
group_spread=0.6

pgm_ex = daft.PGM(dpi=120)

pgm_ex.add_node('sigc','$\sigma _c$',-group_spread/2,0)
pgm_ex.add_node('tau','$t_c$',group_spread/2,0)

pgm_ex.add_node('w1',   '$w_1$',        -(group_sep-group_spread),0)
pgm_ex.add_node('sig1', '$\sigma _1$',  -group_sep,0)
pgm_ex.add_node('lag1', '$\Delta t_1$', -(group_sep+group_spread),0)

pgm_ex.add_node('w2',   '$w_2$',        (group_sep-group_spread),0)
pgm_ex.add_node('sig2', '$\sigma _2$',  group_sep,0)
pgm_ex.add_node('lag2', '$\Delta t_2$', (group_sep+group_spread),0)

pgm_ex.add_node('signal_c', '$\psi_c$',  0,  vert_sep, observed=True)
pgm_ex.add_node('signal_1', '$\psi_1$', -sig_spread,  vert_sep, observed=True)
pgm_ex.add_node('signal_2', '$\psi_2$', sig_spread,  vert_sep, observed=True)

pgm_ex.add_node('signal_c_error', '$\Delta \psi_c$',  0,  vert_sep+plate_sep, fixed=True, offset=[15,0])
pgm_ex.add_node('signal_1_error', '$\Delta \psi_1$', -sig_spread,  vert_sep+plate_sep, fixed=True, offset=[15,0])
pgm_ex.add_node('signal_2_error', '$\Delta \psi_2$', sig_spread,  vert_sep+plate_sep, fixed=True , offset=[15,0])


pgm_ex.add_edge('sigc','signal_c')
pgm_ex.add_edge('tau','signal_c')

pgm_ex.add_edge('sigc','signal_1')
pgm_ex.add_edge('tau','signal_1')
pgm_ex.add_edge('w1','signal_1')
pgm_ex.add_edge('sig1','signal_1')
pgm_ex.add_edge('lag1','signal_1')

pgm_ex.add_edge('sigc','signal_2')
pgm_ex.add_edge('tau','signal_2')
pgm_ex.add_edge('w2','signal_2')
pgm_ex.add_edge('sig2','signal_2')
pgm_ex.add_edge('lag2','signal_2')

pgm_ex.add_edge('signal_c_error','signal_c')
pgm_ex.add_edge('signal_1_error','signal_1')
pgm_ex.add_edge('signal_2_error','signal_2')

pgm_ex.add_plate([-sig_spread*1.5, vert_sep+plate_sep*1.5, sig_spread*1.5*2, -plate_sep*2], label="Measured Signals", position="bottom left")

pgm_ex.render()
#=========================
plt.show()

