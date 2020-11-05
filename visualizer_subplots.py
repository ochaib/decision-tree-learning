import matplotlib.pyplot as plt
plt.switch_backend('agg')

def plot_node(node, xmin, xmax, y, ax):
  x = (xmin + xmax) // 2
  a = ax[y][x]
  if not node.is_leaf:
    a.text(0.5, 0.5, f'Room {node.attr + 1} > {int(node.value)}', horizontalalignment='center', fontsize=9)
    plot_node(node.left, xmin, x, y+1, ax)
    plot_node(node.right, x, xmax, y+1, ax)
  else:
    a.text(0.5, 0.5, f'Room {int(node.value)}', horizontalalignment='center', fontsize=9)

def visualize(tree_root, depth, save='test.png'):
  plt.clf()
  plt.axis('off')

  print("depth", depth)
  width = 2**depth - 1
  print("width", width)
  fig, ax = plt.subplots(nrows=depth, ncols=width)
  fig.tight_layout()
  for row in ax:
    for subplot in row:
      subplot.axis('off')
  print("Made subplots")

  plot_node(tree_root, 0, width, 0, ax)

  if save:
    plt.savefig(save)
  else:
    plt.show()