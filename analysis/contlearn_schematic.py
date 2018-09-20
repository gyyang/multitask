# -*- coding: utf-8 -*-
"""
Plot schematics for continual learning
@author: guangyuyang
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

f_loss = lambda x, u : (np.abs(x-u))**4

# colors = sns.xkcd_palette(['red', 'blue', 'green'])
colors = np.array([[27, 158, 119], [117, 112, 179]])/255.
d = 4
u_ = [-2, 2]
lines = []
fig = plt.figure(figsize=(2.5,1.5))
ax = fig.add_axes([0.15,0.25,0.8,0.7])
for i in range(3):
    if i < 2:
        color = colors[i]
        u = u_[i]
        x_plot = np.linspace(u-d, u+d, 1000)
        loss_plot = f_loss(x_plot, u)
        lw = 4
    else:
        color = (colors[0]+colors[1])/2
        x_plot = np.linspace(np.min(u_)-0.5, np.max(u_)+.5, 1000)
        loss_plot = 0
        for u in u_:
            loss_plot = loss_plot + f_loss(x_plot, u)
        lw = 2
        # color = (np.array(colors[0])+np.array(colors[1]))/2
    line = ax.plot(x_plot, loss_plot, lw=lw, color=color)
    lines.append(line[0])
    i_min = np.argmin(loss_plot)
    ax.plot(x_plot[i_min], loss_plot[i_min], markersize=lw*2, marker='o',
            color='white', markeredgecolor=color, markeredgewidth=lw*0.7)

lg = ax.legend(lines, ('Task 1', 'Task 2', 'Task 1 + 2'),
               fontsize=7, ncol=1, bbox_to_anchor=(1.1,1.1),
               labelspacing=0.2, loc=1, frameon=False)

ax.set_xticks([])
ax.set_yticks([])

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['left'].set_position(('outward', 10))  # outward by 10 points
ax.spines['bottom'].set_position(('outward', 10))  # outward by 10 points
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax.set_ylim(bottom=-50)
ax.set_xlim([np.min(u_)-d-0.3, np.max(u_)+d+1])

ax.set_xlabel(r'Parameter $\theta$', fontsize=7)
ax.set_ylabel(r'Loss $L$', fontsize=7)
plt.savefig('figure/schematic_contlearn.pdf', transparent=True)