import matplotlib.pyplot as plt

plt.switch_backend('agg')

Y_INCREMENT = 0.1
PADDING = 0.04
ARROW_SIZE = 0.5


def plot_node(node, depth=1, xmin=0, xmax=1, arrow_heads=False):
    x = (xmin + xmax) / 2
    y = depth * Y_INCREMENT

    if not node.is_leaf:
        depth += 1
        plt.text(
            x, y, f'Signal {node.attr + 1} > {int(node.value)}', horizontalalignment='center')
        plot_node(node.left, depth + 1, xmin, x)
        plot_node(node.right, depth + 1, x, xmax)
        if arrow_heads:
            plt.arrow(x, y + PADDING, (xmin + x) / 2 - x,
                      Y_INCREMENT * 2 - PADDING * 2, color='red')
            plt.arrow(x, y + PADDING, (xmax + x) / 2 - x,
                      Y_INCREMENT * 2 - PADDING * 2, color='green')
        else:
            plt.plot([x, (xmin + x) / 2], [y + PADDING,
                                           y + Y_INCREMENT * 2 - PADDING], 'r-')
            plt.plot([x, (xmax + x) / 2], [y + PADDING,
                                           y + Y_INCREMENT * 2 - PADDING], 'g-')
    else:
        plt.text(x, y, f'Room {int(node.value)}',
                 horizontalalignment='center',
                 bbox=dict(facecolor='none', edgecolor='purple', pad=3.0))


def visualize(tree_root, depth, save=False, maxwidth=2 ** 16):
    plt.clf()
    fig = plt.figure()
    plt.axis('off')
    plt.tight_layout()
    plt.ylim((0, Y_INCREMENT * depth * 2))
    plot_node(tree_root)
    fig.set_size_inches(min(depth ** 2, maxwidth), depth)
    if save:
        plt.savefig(save)
    else:
        plt.show()
