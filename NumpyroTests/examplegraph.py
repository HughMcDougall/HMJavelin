
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
twoline      = True
fixedwidths  = True
bayesian     = True
laglum_prior = True
lagsim_prior = True

#=========================
sig_spread=1
vert_sep=-1.5
plate_sep = -0.8
group_sep=2
group_spread=0.6
#=========================

pgm_ex = daft.PGM(dpi=120)

pgm_ex.add_node('sigc','$\sigma _c$',-group_spread/2,0)
pgm_ex.add_node('tau','$t_c$',group_spread/2,0)

pgm_ex.add_node('w1',   '$w_1$',        -(group_sep-group_spread),0, fixed=fixedwidths)
pgm_ex.add_node('sig1', '$\sigma _1$',  -group_sep,0)
pgm_ex.add_node('lag1', '$\Delta t_1$', -(group_sep+group_spread),0)

if twoline:
    pgm_ex.add_node('w2',   '$w_2$',        (group_sep-group_spread),0, fixed=fixedwidths)
    pgm_ex.add_node('sig2', '$\sigma _2$',  group_sep,0)
    pgm_ex.add_node('lag2', '$\Delta t_2$', (group_sep+group_spread),0)

pgm_ex.add_node('signal_c', '$\psi_c$',  0,  vert_sep, observed=True)
pgm_ex.add_node('signal_1', '$\psi_1$', -sig_spread,  vert_sep, observed=True)
if twoline:
    pgm_ex.add_node('signal_2', '$\psi_2$', sig_spread,  vert_sep, observed=True)

pgm_ex.add_node('signal_c_error', '$\Delta \psi_c$',  0,  vert_sep+plate_sep, fixed=True, offset=[15,0])
pgm_ex.add_node('signal_1_error', '$\Delta \psi_1$', -sig_spread,  vert_sep+plate_sep, fixed=True, offset=[15,0])
if twoline:
    pgm_ex.add_node('signal_2_error', '$\Delta \psi_2$', sig_spread,  vert_sep+plate_sep, fixed=True , offset=[15,0])


pgm_ex.add_edge('sigc','signal_c')
pgm_ex.add_edge('tau','signal_c')

#pgm_ex.add_edge('sigc','signal_1')
pgm_ex.add_edge('tau','signal_1')
pgm_ex.add_edge('w1','signal_1')
pgm_ex.add_edge('sig1','signal_1')
pgm_ex.add_edge('lag1','signal_1')

if twoline:
    #pgm_ex.add_edge('sigc','signal_2')
    pgm_ex.add_edge('tau','signal_2')
    pgm_ex.add_edge('w2','signal_2')
    pgm_ex.add_edge('sig2','signal_2')
    pgm_ex.add_edge('lag2','signal_2')

pgm_ex.add_edge('signal_c_error','signal_c')
pgm_ex.add_edge('signal_1_error','signal_1')
if twoline:
    pgm_ex.add_edge('signal_2_error','signal_2')


#=========================
if laglum_prior:
    pgm_ex.add_node('z', '$z$', 0,  group_spread*4/2, fixed=True)
    pgm_ex.add_node('L', '$L$', 0,  group_spread*2/2)

    pgm_ex.add_edge('signal_2_error','signal_2')

    pgm_ex.add_edge('L','signal_c')
    
    pgm_ex.add_edge('z','lag1')
    pgm_ex.add_edge('L','lag1')

    if twoline:
        pgm_ex.add_edge('z','lag2')
        pgm_ex.add_edge('L','lag2')


#=========================
if bayesian:
    if twoline:
        platewidth=group_sep*2+group_spread*4
    else:
        platewidth=(group_sep*2+group_spread*4)/2

    if laglum_prior or lagsim_prior:
        plateheight=-vert_sep*3
    else:
        plateheight=-vert_sep*2
        
    pgm_ex.add_plate([-group_sep-group_spread*2,   vert_sep*1.7, platewidth,plateheight])

#=========================
pgm_ex.render()

#=========================
        
plt.show()

