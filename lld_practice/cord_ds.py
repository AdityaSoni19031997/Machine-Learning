# Cord DS aka Rope DS to hold very large string texts (Cool DS!)
# https://en.wikipedia.org/wiki/Rope_(data_structure)
# https://blog.csdn.net/ai_xiangjuan/article/details/79246289

class LeafNode:
    def __init__(self):
        self.val = ""
        self.length = 0
        
class InternalNode:
    def __init__(self):
        self.length = 0
        self.left = self.right = None
        
class Cord:
    def __init__(self, s):
        self.threshold = 5
        self.root = self.build(s)
        
    def build(self, s):
        if len(s) > self.threshold:
            mid = len(s) // 2
            node = InternalNode()
            node.left = self.build(s[:mid])
            node.right = self.build(s[mid:])
            node.length = node.left.length + node.right.length
            return node
        else:
            node = LeafNode()
            node.val = s
            node.length = len(s)
            return node
        
    def toString(self, node = None):
        if not node:
            node = self.root
        if isinstance(node, LeafNode):
            return node.val
        return self.toString(node.left) + self.toString(node.right)
    
    def charAt(self, i, node = None):
        if node == None:
            node = self.root
        if i >= node.length:
            raise Exception("Out of bound")
        if isinstance(node, LeafNode):
            return node.val[i]
        if i < node.left.length:
            return self.charAt(i, node.left)
        return self.charAt(i - node.left.length, node.right)
    
    def merge(self, other):
        newRoot = InternalNode()
        newRoot.left = self.root
        newRoot.right = other.root
        newRoot.length = self.root.length + other.root.length
        self.root = newRoot
        
    def substring(self, left, right, node = None):
        if node == None:
            node = self.root
        if not 0 <= left < node.length or not left <= right or not 0 <= right <= node.length:
            raise Exception("Substring size is incorrect")
        if isinstance(node, LeafNode):
            return node.val[left:right]
        ans = ""
        if left < node.left.length:
            ans += self.substring(left, min(node.left.length, right), node.left)
        if right > node.left.length:
            ans += self.substring(max(0, left - node.left.length), right - node.left.length, node.right)
        return ans

    
    
# Dry Run

s = "abcde"
s2 = "pdgsadfssagdasdgasda"

cord = Cord(s)
cord2 = Cord(s2)

for i in range(len(s)):
    for j in range(i, len(s) + 1):
        if cord.substring(i,j) != s[i:j]:
            print("Found Bug")

for i in range(len(s)):
    if s[i] != cord.charAt(i):
        print("Found Bug")

cord.merge(cord2)

s += s2

print(s == cord.toString())

for i in range(len(s)):
    for j in range(i + 1, len(s) + 1):
        if cord.substring(i,j) != s[i:j]:
            print("Found bug")

for i in range(len(s)):
    if s[i] != cord.charAt(i):
        print("Found Bug")
