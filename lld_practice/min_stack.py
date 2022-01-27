# min_stack
# Ref -: Please refer GFG post as well for the prrof as to why this works?

"""
**How does this approach work? **

When element to be inserted is less than minEle, we insert “2x – minEle”. The important thing to notes is, 2x – minEle will always be less than x (proved below), i.e., new minEle and while popping out this element we will see that something unusual has happened as the popped element is less than the minEle. So we will be updating minEle.
 
### PUSH

```
How 2*x - minEle is less than x in push()? 
x < minEle which means x - minEle < 0

// Adding x on both sides
x - minEle + x < 0 + x 

2*x - minEle < x 

We can conclude 2*x - minEle < new minEle 
```

### POP
While popping out, if we find the element(y) less than the **current** **minEle**, we find the **new minEle = 2*minEle – y**.
 
```
How previous minimum element, prevMinEle is, 2*minEle - y
in pop() is y the popped element?

 // We pushed y as 2x - prevMinEle. Here 
 // prevMinEle is minEle before y was inserted
 y = 2*x - prevMinEle  

 // Value of minEle was made equal to x
 minEle = x .
    
 new minEle = 2 * minEle - y 
            = 2*x - (2*x - prevMinEle)
            = prevMinEle // This is what we wanted**
 ```
"""


# Class to make a Node
class Node:
    # Constructor which assign argument to nade's value 
    def __init__(self, value):
        self.value = value
        self.next = None

    # This method returns the string representation of the object.
    def __str__(self):
        return "Node({})".format(self.value)
    
    # __repr__ is same as __str__
    __repr__ = __str__


class MinStack:
    # Stack Constructor initialise top of stack and counter.
    def __init__(self):
        self.top_ = None
        self.count = 0
        self.minimum = None

    
    # This method is used to get minimum element of stack
    def getMin(self):
        if self.top_ is None:
            return "Stack is empty"
        else:
            print("Minimum Element in the stack is: {}" .format(self.minimum))
            return self.minimum

    # This method returns length of stack     
    def __len__(self):
        self.count = 0
        tempNode = self.top_
        while tempNode:
            tempNode = tempNode.next
            self.count += 1
        return self.count

    # This method returns top of stack     
    def top(self):
        if self.top_ is None:
            print ("Stack is empty")
        else: 
            if self.top_.value < self.minimum:
                print("Top Most Element is: {}" .format(self.minimum))
                return self.minimum
            else:
                print("Top Most Element is: {}" .format(self.top_.value))
                return self.top_.value

    # This method is used to add node to stack
    def push(self,value):
        if self.top_ is None:
            self.top_ = Node(value)
            self.minimum = value
        
        elif value < self.minimum:
            # incoming value is smaller than the minimum, we gotta update our minimum.
            temp = (2 * value) - self.minimum
            new_node = Node(temp)
            new_node.next = self.top_
            self.top_ = new_node
            self.minimum = value
        else:
            new_node = Node(value)
            new_node.next = self.top_
            self.top_ = new_node

        print("Number Inserted: {}" .format(value))

    # This method is used to pop top of stack
    def pop(self):
        if self.top_ is None:
            print( "Stack is empty")
        else:
            removedNode = self.top_.value
            self.top_ = self.top_.next
            if removedNode < self.minimum:
                print ("Top Most Element Removed :{} " .format(self.minimum))
                self.minimum = 2*self.minimum - removedNode
            else:
                print ("Top Most Element Removed : {}" .format(removedNode))



# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()


######################################################
##################Another Simple Idea ################
######################################################

class MinimumStack:
    def __init__(self):
        self.q = []
    
    def append(self, val):
        curMin = self.min()
        if curMin == None or val < curMin:
            curMin = val
        self.q.append((val, curMin));        

    def peek(self):
        if len(self.q) == 0:
            return None
        else:
            return self.q[len(self.q) - 1][0]        

    def min(self):
        if len(self.q) == 0:
            return None
        else:
            return self.q[len(self.q) - 1][1]        

    def pop(self):
        return self.q.pop()[0]
