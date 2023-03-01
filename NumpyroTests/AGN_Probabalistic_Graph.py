
'''
Practice for graph generation

This DOESNT DO MATHS

HM 15/12
'''

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import daft
#PART THRE: AGN Draft

group_sep_h =3
spacing_h = 1
group_sep_y = 2

pgm_ex = daft.PGM(dpi=120)

pgm_ex.add_node('L','$L$',-group_sep_h/2 -spacing_h*3/2,0)
pgm_ex.add_node('sigc','$\sigma _\infty$',-group_sep_h/2 -spacing_h/2,0)
pgm_ex.add_node('tau','$ τ_d$',-group_sep_h/2 + spacing_h/2,0)

pgm_ex.add_node('sig_i', '$\sigma _j$',  group_sep_h/2 -spacing_h/2,0)
pgm_ex.add_node('lag_i', '$\Delta t_j$', group_sep_h/2 +spacing_h/2,0)
pgm_ex.add_plate([group_sep_h/2-spacing_h, -spacing_h/2, spacing_h*2, spacing_h])

pgm_ex.add_node('signal_c', '$s_{i,j}(t)$',  0,  -group_sep_y , observed=True)
pgm_ex.add_plate([-spacing_h, -group_sep_y-spacing_h/2, spacing_h*2, spacing_h])

pgm_ex.add_plate([-group_sep_h/2-spacing_h*2, -group_sep_y-spacing_h, group_sep_h+spacing_h*3.5, group_sep_y+spacing_h*2])

pgm_ex.add_node('RLslope', '$A$',  group_sep_h/2 - spacing_h/2, group_sep_y)
pgm_ex.add_node('RLmean', '$C$',  group_sep_h/2 + spacing_h/2, group_sep_y)


pgm_ex.render()
#=========================
plt.show()

