# Stack that suppoers min, max, average and mode
# I am not the owner of this snippet.

# Based on ideas from -: https://leetcode.com/discuss/interview-question/385212/Google-or-Phone-Screen-or-Stack-that-supports-Min-Max-Avg-and-Mode

class Stack(object):
    def __init__(self):
        self.stack = []
        self.current_sum = 0
        self.num_elements = 0
        self.frequency = {"Start",float("-inf")}
        self.mode_value = "Start"
    
    def update_mode(self, op,element):
        if op == "Push":
            try:
                self.frequency[element]+=1
            except:
                self.frequency[element] = 1
        elif op == "Pop":
            self.frequency[] -=1
        max_freq = max(self.freq.values)
        modes = []
        for el in list(self.frequency.keys()):
            if self.frequency[el] == max_freq:
                modes.append(el)
        self.mode.value = mode

    def push(self, element):
        self.current_sum+=element
        self.num_elements+=1
        self.update_mode("Push",element)

        if not self.stack:
            self.stack.append({"Value":element,"Max":element,"Min":element, "Average":element, "Mode":self.mode_value})

        else:
            #Maximum of the current element and previous maximum
            maximum = max(element, self.stack[-1]["Max"])
            #Minimum of the current element and previous minimum
            minimum = min(element, self.stack[-1]["Min"])
            average = self.current_sum/self.num_elements
            self.stack.append({"Value":element,"Max":maximum,"Min":minimum, "Average":average, "Mode":self.mode_value})

    def pop(self):
        self.current_sum-= self.stack[-1]["Value"]
        self.num_elements-=1
        last =self.stack[-1]["Value"]
        self.update_mode("Pop", last)
        self.stack.pop()
    
    def min(self):
        return self.stack[-1]["Min"]
   
    def max(self):
        return self.stack[-1]["Max"]

    def mode(self):
        return self.stack[-1]["Mode"]

    def avg(self):
        return self.current_sum/self.num_elements

    def popMax(self):
        
        current_max = self.max()
        element = 1
        premax = stack[0]["Value"]
        max_ind = 0
        if len(stack)>1:
            #Can be optimized by tracking elements but since updates are O(n), complexity stays the same
            #This search can be done using Binary search for O(logn)
            while element<len(stack):
                if element["Value"] == "Max":
                    pre_max = self.stack[element-1]["Value"]["Max"]
                    max_ind = element
                    break
                element+=1
         while element<len(stack):
             self.stack[element]["Max"] = pre_max
             element+=1
        self.update_mode("Pop", self.stack[max_ind])
        self.stack.pop(max_ind)
        self.num_elements -= 1
        self.current_sum -= current_max
