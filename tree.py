class TreeNode:
    treeNode = {}

    def __init__(self, value, attr=None, left=None, right=None):
        self.value = value
        self.attr = attr
        self.left = left
        self.right = right

    def TreeNode(self):
        return {self.attr, self.value, self.left, self.right}

    def add_left_child(self, child):
        self.left = child

    def add_right_child(self, child):
        self.right = child

    @property
    def is_leaf(self):
        return (self.left is None) & (self.right is None)

    # Called on root tree node.
    def get_leaf_nodes(self):
        leaf_nodes = []
        self._collect_leaf_nodes(self, leaf_nodes)
        return leaf_nodes

    def _collect_leaf_nodes(self, node, leaf_nodes):
        if node is not None:
            if node.is_leaf:
                leaf_nodes.append(node)
            self._collect_leaf_nodes(node.left, leaf_nodes)
            self._collect_leaf_nodes(node.right, leaf_nodes)

    def __str__(self):
        return f"{self.attr} > {self.value}\n" + "L: " + str(self.left) + " R: " + str(self.right)