#from TSP_Model import Model
from Solver import *
import time

def print_nodes(m):
    for n in m.allNodes:
        print(n.ID, " ", n.x, " ", n.y, "  ", n.type, " ", n.demand)

def print_matrix(m):
    for i in range(len(m.matrix)):
        print(m.matrix[i])

def write_output_file(sol):
    f = open('sol_8180023.txt', 'w')
    f.write(str(calculate_obj(sol)) +"\n")
    for r in sol.routes:#Does not write the return to depot.
        #print(r.sequenceOfNodes)
        for n in range(0, len(r.sequenceOfNodes) - 2):
            f.write(str(r.sequenceOfNodes[n].ID) + ",")
        #write last node of route
        f.write(str(r.sequenceOfNodes[-2].ID) + "\n")
    f.close()

def calculate_obj(sol):
    rsortedlist = sorted(sol.routes, key= lambda r :r.cost, reverse= True)
    return rsortedlist[0].cost, rsortedlist[0].capacity, rsortedlist[0].load, rsortedlist[0].sequenceOfNodes[0].ID, rsortedlist[0].sequenceOfNodes[-2].ID

m = Model()
m.BuildModel()
#print_nodes(m)
# print_matrix(m)

s = Solver(m)
sol = s.solve()
# print(len(sol.routes))
for c in s.customers:
    if c.isRouted == False:
        print('KINDA SOMETHING WENT WRONG!')
write_output_file(sol)



