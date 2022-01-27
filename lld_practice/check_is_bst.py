class Node:
    def __init__(self, val=None):
        self.left, self.right, self.val = None, None, val
 
INFINITY = float("infinity")
NEG_INFINITY = float("-infinity")


# the idea is to use the min and max value for every node otehrwise some other tree will get
# categoriesed as bst.

def isBST(tree, minVal=NEG_INFINITY, maxVal=INFINITY):
    if tree is None:
        return True
 
    if not minVal <= tree.val <= maxVal:
        return False
 
    return isBST(tree.left, minVal, tree.val) and \
           isBST(tree.right, tree.val, maxVal)

# another alternative is to do strictly non-decreasing in-order traversal if you have DUPS (not done below)

def isBST2(tree, lastNode=[NEG_INFINITY]):
    if tree is None:
        return True
 
    if not isBST2(tree.left, lastNode):
        return False
 
    if tree.val < lastNode[0]:
        return False
 
    lastNode[0]=tree.val
 
    return isBST2(tree.right, lastNode)
