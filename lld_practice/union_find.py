"""
Very Nice Explanation is There
Reference -: 
- https://yuminlee2.medium.com/union-find-algorithm-ffa9cd7d2dba (code comes from here)
- https://cp-algorithms.com/data_structures/disjoint_set_union.html
"""

class UnionFind:
    def __init__(self, numOfElements):
        self.parent = self.makeSet(numOfElements)
        self.size = [1]*numOfElements
        self.count = numOfElements
    
    def makeSet(self, numOfElements):
        return [x for x in range(numOfElements)]

    # Time: O(logn) | Space: O(1)
    def find(self, node):
        while node != self.parent[node]:
            # path compression
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node
    
    # Time: O(1) | Space: O(1)
    def union(self, node1, node2):
        root1 = self.find(node1)
        root2 = self.find(node2)

        # already in the same set
        if root1 == root2:
            return

        if self.size[root1] > self.size[root2]:
            self.parent[root2] = root1
            self.size[root1] += 1
        else:
            self.parent[root1] = root2
            self.size[root2] += 1
        
        self.count -= 1


if __name__ == "__main__":
    edges = [
        [0, 2],
        [1, 4],
        [1, 5],
        [2, 3],
        [2, 7],
        [4, 8],
        [5, 8],
    ]
    numberOfElements = 9

    uf = UnionFind(9)

    for node1, node2 in edges:
        uf.union(node1, node2)
    
    print("number of connected components", uf.count)
