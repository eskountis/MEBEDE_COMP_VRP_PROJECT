import random
import math


class Model:

# instance variables
    def __init__(self):
        self.allNodes = []
        self.customers = [] #Service Locations
        self.matrix = [] #Matrix is on h, so convert min into h.
        self.capacity = -1 #3000

    def BuildModel(self):
        random.seed(1)
        depot = Node(0, 0, 0, 50, 50)
        self.allNodes.append(depot)
        self.capacity = 3000 #Routes capacity
        totalCustomers = 200 # Total service locations (=200)
        for i in range(0, totalCustomers):
            tp = random.randint(1, 3)
            dem = random.randint(1, 5) * 100
            xx = random.randint(0, 100)
            yy = random.randint(0, 100)
            cust = Node(i + 1, tp, dem, xx, yy)
            self.allNodes.append(cust)
            self.customers.append(cust)

        rows = len(self.allNodes)
        self.matrix = [[0.0 for x in range(rows)] for y in range(rows)]

        for i in range(0, len(self.allNodes)):
            for j in range(1, len(self.allNodes)):
                a = self.allNodes[i]#from
                b = self.allNodes[j]#to
                dist = round(math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2))
                time = dist / 35 # Converting km/h into h
                if i != j:
                    if b.type == 1:
                        time += 5/60
                    elif b.type == 2:
                        time += 15/60
                    elif b.type == 3:
                        time += 25/60
                self.matrix[i][j] = time #operator 1 needs round , 0,2 needs round ,1(remove prints), operator 0 needs round ,0




class Node:
    def __init__(self, id, tp, dem, x, y):
        self.x = x
        self.y = y
        self.ID = id
        self.demand = dem
        self.type = tp
        self.isRouted = False

class Route:
    def __init__(self, dp, cap):
        self.sequenceOfNodes = []
        self.sequenceOfNodes.append(dp)
        self.sequenceOfNodes.append(dp)
        self.cost = 0
        self.capacity = cap
        self.load = 0