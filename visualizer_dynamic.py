import matplotlib.pyplot as plt
plt.switch_backend('agg')

def get_children_coords(parent_x, parent_y, depth):
  y = parent_y * 2
  x_offset = parent_x / 2
  left_x = parent_x - x_offset



def plot_node(node, depth=1, x=0.5, y=0.1):
  if not node.is_leaf:
    plt.text(x, y, f'{node.attr} > {int(node.value)}', horizontalalignment='center')
    depth += 1
    plot_node(node.left, max_depth, depth, xmin, x)
    plot_node(node.right, max_depth, depth, x, xmax)
  else:
    plt.text(x, y, f'{int(node.value)}')

def visualize(tree_root, save='test.png'):
  node = tree_root
  tree_height = 1
  while not node.is_leaf:
    node = node.left
    tree_height += 1
  plt.clf()
  plt.axis('off')
  ax = plt.subplot(111)
  ax.set_xlim([0.1, 0.9])
  plot_node(tree_root, max_depth=tree_height)
  if save:
    plt.savefig(save)
  else:
    plt.show()