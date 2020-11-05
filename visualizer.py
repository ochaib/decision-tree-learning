import matplotlib.pyplot as plt
plt.switch_backend('agg')

Y_INCREMENT=0.1
PADDING = 0.04

def plot_node(node, depth=1, xmin=0, xmax=1):
  x = (xmin + xmax) / 2
  y = depth*Y_INCREMENT
  if not node.is_leaf:
    depth += 1
    plt.text(x, y, f'Room {node.attr+1} > {int(node.value)}', horizontalalignment='center')
    plot_node(node.left, depth+1, xmin, x)
    plt.plot([x, (xmin + x) / 2], [y + PADDING, y + Y_INCREMENT*2 - PADDING])
    plot_node(node.right, depth+1, x, xmax)
    plt.plot([x, (xmax + x) / 2], [y + PADDING, y + Y_INCREMENT*2 - PADDING])
  else:
    plt.text(x, y, f'Room {int(node.value)}', horizontalalignment='center')

def visualize(tree_root, depth, save='test.png'):
  plt.clf()
  fig = plt.figure()
  plt.axis('off')
  plt.tight_layout()
  plt.ylim((0,Y_INCREMENT*depth*2))
  plot_node(tree_root)
  fig.set_size_inches(depth**2, depth)
  if save:
    plt.savefig(save)
  else:
    plt.show()