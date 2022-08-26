import random
from collections import deque

r"""
ref -: TBD, forgot where i borrowed pats from the src :(, but it's there on git
I have limited the snippet that one can come up with by removing the self balancing-tree sections
from the snippet.
lecture ref notes -: https://research.engineering.nyu.edu/~greg/algorithms/classnotes/interval-trees.pdf
lecture ref -: https://www.youtube.com/watch?v=q0QOYtSsTg4&ab_channel=OsirisSalazar
# NB implementation is based ont he pdf notes, not on the YT video but video is amazing!
"""

class IntervalNode:
    def __init__(self, val):
        assert len(val) == 2
        assert val[0] <= val[1]
        val = tuple(val) 
        self.val = val
        self.parent = None
        self.left = None
        self.right = None
        self.height = 0
        # Augmented DataStructure: Store the max right value of the subtree rooted at this node
        self.maxRight = val[1]

    def __str__(self):
        return "IntervalNode("+ str(self.val)+ ", maxRight: %d )" % self.maxRight
    

class IntervalTree:
    def __init__(self):
        self.root = None
        self.nodes_count = 0
    
    def setRoot(self, val):
        r"""Set the root value"""
        self.root = IntervalNode(val)
    
    def countNodes(self):
        return self.nodes_count

    def insert(self, val):
        r"""insert a val into IntervalTree"""
        assert len(val) == 2
        assert val[1] >= val[0]
        val = tuple(val)
        if self.root is None:
            self.setRoot(val)
        else:
            self._insertNode(self.root, val)
        self.nodes_count += 1
    
    def _insertNode(self, currentNode, val):
        r"""Helper function to insert a value into IntervalTree."""
        # first, update the currentNode
        currentNode.maxRight = max(currentNode.maxRight, val[1])

        if currentNode.val > val:
            if(currentNode.left):
                self._insertNode(currentNode.left, val)
            else:
                child_node = IntervalNode(val)
                currentNode.left = child_node
                child_node.parent = currentNode
        else:
            if(currentNode.right):
                self._insertNode(currentNode.right, val)
            else:
                child_node = IntervalNode(val)
                currentNode.right = child_node
                child_node.parent = currentNode


    def search(self, key):
        r"""Search a IntervalNode satisfies IntervalNode.val = key.
        if found return IntervalNode, else return None.
        """
        assert len(key) == 2
        assert key[1] >= key[0]
        key = tuple(key)
        return self._dfsSearch(self.root, key)
    
    def _dfsSearch(self, currentNode, key):
        r"""Helper function to search a key in IntervalTree."""
        if currentNode is None:
            return None
        elif currentNode.val == key:
            return currentNode
        elif currentNode.val > key:
            return self._dfsSearch(currentNode.left, key)
        else:
            return self._dfsSearch(currentNode.right, key)

    def queryOverlap(self, val):
        r"""val should be an input interval.
        return IntervalNode that overlaps with the input interval that we find first in the IntervalTree.
        if not found, return None
        """
        assert len(val) == 2
        assert val[1] >= val[0]
        val = tuple(val)
        return self._dfsQueryOverlap(self.root, val)

    def _dfsQueryOverlap(self, node, val):
        r"""Helper function for query.
        returns the first interval we find in the subtree rooted at node.
        """
        if not node: 
            return None
        
        # if interval intersects with the root node, return the same
        if self._isOverlap(node.val, val):
            return node.val
        else:
            L, R = val
            if R < node.val[0]:
                # Case1
                #  Right subtree can't overlap, search left
                # ----- val
                #        ------ node.val
                #        /    \
                #     ----    -----
                return self._dfsQueryOverlap(node.left, val)
            
            elif L > node.val[1]:
                z = node.left.maxRight if node.left else (float("-inf"))
                if z >= L:
                    # Case2
                    # Left subtree guranteed overlap
                    #               L-----R 
                    #        ------ node.val
                    #        /    
                    #     -----.......z   
                    return self._dfsQueryOverlap(node.left, val)
                else:
                    # Case3
                    # Left subtree no overlap, search right
                    #               L-----R 
                    #        ------ node.val
                    #        /    
                    #     -----..z  
                    return self._dfsQueryOverlap(node.right, val)
            return None
    
    def queryAllOverlaps(self, val):
        r"""find all the intervals in the interval tree.
        return a list of all intervals that overlap with val.
        """
        assert len(val) == 2
        assert val[1] >= val[0]
        val = tuple(val)
        res = []
        self._dfsFind(self.root, val, res)
        return res
    
    def _dfsFind(self, node, val, res):
        r"""it should be better then naive iterations.
        """
        if not node: return None 

        if self._isOverlap(node.val, val):
            res.append(node.val)
        
        L, R = val

        if R < node.val[0]:
            # Case1
            #  Right subtree can't overlap, search left
            # ----- val
            #        ------ node.val
            #        /    \
            #     ----    -----
            self._dfsFind(node.left, val, res)
        elif L > node.val[1]:
            z = node.left.maxRight if node.left else (float("-inf"))
            if z<L:
                # Case3
                # Left subtree no overlap, search right
                #               L-----R 
                #        ------ node.val
                #        /    
                #     -----..z  
                self._dfsFind(node.right, val, res)
        else:
            self._dfsFind(node.left, val, res)
            self._dfsFind(node.right, val, res)
        return None
                
    def _isOverlap(self, interval1, interval2):
        r"""check intervals"""
        l = sorted([interval1, interval2])
        if l[1][0] <= l[0][1] : 
            return True
        else:
            return False
    
    def getDepth(self):
        r"""Get the max depth of the BST"""
        if self.root:
            return self.root.height
        else:
            return -1
    
    @classmethod
    def buildFromList(cls, l, shuffle = True):
        r"""return a IntervalTree object from l"""
        if shuffle:
            random.seed()
            random.shuffle(l)
        
        IT = IntervalTree()
        for item in l:
            IT.insert(item)
        return IT

    def visulize(self):
        r"""Naive Visulization. 
        Warn: Only for simple test usage.
        """
        if self.root is None:
            print("EMPTY TREE.")
        else:
            print("-----------------Visualize Tree----------------------")
            layer = deque([self.root])
            layer_count = self.getDepth()
            while len( list(filter(lambda x:x is not None, layer) )):
                new_layer = deque([])
                val_list = []
                while len(layer):
                    node = layer.popleft()
                    if node is not None:
                        val_list.append((node.val,node.maxRight))
                    else:
                        val_list.append(" ")
                    if node is None:
                        new_layer.append(None)
                        new_layer.append(None)
                    else:
                        new_layer.append(node.left)
                        new_layer.append(node.right)
                val_list = [" "] * layer_count + val_list
                print(*val_list, sep="  ", end="\n")
                layer = new_layer
                layer_count -= 1
            print("-----------------End Visualization-------------------")


# TEST THE SAME
intervals = [
    [7,10],
    [5,11],
    [4,8],
    [17,19],
    [15,18],
    [21,23]
]

overlaps = IntervalTree.buildFromList(intervals)
overlaps.visulize()

print("Overlap with [20,22]",overlaps.queryOverlap([20,22]))
print("Overlap with [24,25]",overlaps.queryOverlap([24,25]))
print("After Insert [24,24]")

overlaps.insert([24,24])

print("Overlap with [24,25]",overlaps.queryOverlap([24,25]))
print("Overlap with [5,9]",overlaps.queryOverlap([5,9]))
print("Overlap with [16,17]",overlaps.queryOverlap([16,17]))

print("Overlap with [16,17]",overlaps.queryOverlap([16,17]))
print("Overlap with [0,3]",overlaps.queryOverlap([0,3]))

# Test findAllOverlaps
print("queryAllOverlaps with [10,20]",overlaps.queryAllOverlaps([10,20]))
print("[END] Implementation of IntervalTree.")
