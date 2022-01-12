class Node:
    def __init__(self, key=-1, val=-1, nxt=None, prv=None):
        self.key, self.val, self.next, self.prev = key, val, nxt, prv


class LinkedList:
    def __init__(self):
        # Head is the Most Recently Used -- Tail is the Least Recently Used
        self.head = self.tail = None

    def appendleft(self, node) -> None:
        # Makes node the new head (appends left)
        if self.head:
            node.next, self.head.prev, self.head = self.head, node, node
        else:
            self.head = self.tail = node

    def remove(self, node) -> None:
        # Removes node from linkedlist (if statements are necessary to take care of edge cases)
        if node == self.head == self.tail:
            self.head = self.tail = None
        elif node == self.head:
            self.popleft()
        elif node == self.tail:
            self.pop()
        else:
            node.prev.next, node.next.prev = node.next, node.prev

    def pop(self) -> Node :
        # Removing tail and reassigning tail to the previous node, then we return the old tail
        oldTail, self.tail, self.tail.next = self.tail, self.tail.prev, None
        return oldTail
    
    def popleft(self) -> None:
        # Removing head and reassigning head to the next node
        self.head, self.head.prev = self.head.next, None


class LRUCache:
    def __init__(self, capacity: int):
        # Linkedlist will hold nodes in the following order: Most recent -> ... -> Least recent
        self.cap, self.elems, self.l  = capacity, {}, LinkedList()

    def get(self, key: int) -> int:
        if key in self.elems:
            # Before returning the value, we update position of the element in the linkedlist
            self.l.remove(self.elems[key])
            self.l.appendleft(self.elems[key])
            return self.elems[key].val
        return -1        

    def put(self, key: int, value: int) -> None:
        # Remove from linked list if node exists, because we will later appendleft the NEW node
        if key in self.elems: self.l.remove(self.elems[key])
            
        # Create new node, then add and appenleft
        self.elems[key] = Node(key, value)
        self.l.appendleft(self.elems[key])
        
        # Check if we have more elements than capacity, delete LRU from linked list and dictionary
        if len(self.elems) > self.cap: del self.elems[self.l.pop().key]
