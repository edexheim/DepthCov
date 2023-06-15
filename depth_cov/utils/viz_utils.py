import matplotlib.pyplot as plt

def remove_all_ticks(axs):  
  for i in range(len(axs)):
    for j in range(len(axs[i])):
      axs[i][j].tick_params(axis='both',which='both',
          labelbottom=False,labelleft=False,
          bottom=False, left=False)