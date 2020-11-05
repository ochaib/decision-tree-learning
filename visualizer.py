import matplotlib.pyplot as plt
plt.switch_backend('agg')

Y_INCREMENT=0.1
X_INCREMENT=0.3

def plot_node(node, depth=1, xmin=0, xmax=1):
  x = (xmin + xmax) / 2
  y = depth*Y_INCREMENT
  if not node.is_leaf:
    depth += 1
    plt.text(x, y, f'Room {node.attr+1} > {int(node.value)}', horizontalalignment='center')
    plot_node(node.left, depth+1, xmin, x)
    plot_node(node.right, depth+1, x, xmax)
  else:
    plt.text(x, y, f'Room {int(node.value)}', horizontalalignment='center')

def visualize(tree_root, depth, save='test.png'):
  print(depth)
  plt.clf()
  plt.axis('off')
  plt.tight_layout()
  plt.ylim((0,Y_INCREMENT*depth*2))
  plt.xlim((0,X_INCREMENT*depth*2))
  plot_node(tree_root, xmax=X_INCREMENT*depth*2)
  if save:
    plt.savefig(save)
  else:
    plt.show()