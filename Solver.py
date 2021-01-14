from VRP_Model import *
from SolutionDrawer import *
import random, copy


class Solution:
    def __init__(self):
        self.cost = 0.0
        self.routes = []


class RelocationMove(object):
    def __init__(self):
        self.originRoutePosition = None
        self.targetRoutePosition = None
        self.originNodePosition = None
        self.targetNodePosition = None
        self.costChangeOriginRt = None
        self.costChangeTargetRt = None
        self.moveCost = None

    def Initialize(self):
        self.originRoutePosition = None
        self.targetRoutePosition = None
        self.originNodePosition = None
        self.targetNodePosition = None
        self.costChangeOriginRt = None
        self.costChangeTargetRt = None
        self.moveCost = 0


class SwapMove(object):
    def __init__(self):
        self.positionOfFirstRoute = None
        self.positionOfSecondRoute = None
        self.positionOfFirstNode = None
        self.positionOfSecondNode = None
        self.costChangeFirstRt = None
        self.costChangeSecondRt = None
        self.moveCost = None

    def Initialize(self):
        self.positionOfFirstRoute = None
        self.positionOfSecondRoute = None
        self.positionOfFirstNode = None
        self.positionOfSecondNode = None
        self.costChangeFirstRt = None
        self.costChangeSecondRt = None
        self.moveCost = 0


class CustomerInsertion(object):
    def __init__(self):
        self.customer = None
        self.route = None
        self.cost = 10 ** 9


class CustomerInsertionAllPositions(object):
    def __init__(self):
        self.customer = None
        self.route = None
        self.insertionPosition = None
        self.cost = 10 ** 9


class TwoOptMove(object):
    def __init__(self):
        self.positionOfFirstRoute = None
        self.positionOfSecondRoute = None
        self.positionOfFirstNode = None
        self.positionOfSecondNode = None
        self.moveCost = None

    def Initialize(self):
        self.positionOfFirstRoute = None
        self.positionOfSecondRoute = None
        self.positionOfFirstNode = None
        self.positionOfSecondNode = None
        self.moveCost = 0


class Solver:
    def __init__(self, m):
        self.allNodes = m.allNodes
        self.customers = m.customers
        self.depot = m.allNodes[0]
        self.distanceMatrix = m.matrix
        self.capacity = m.capacity
        self.sol = None
        self.bestSolution = None
        self.overallBestSol = None
        self.rcl_size = 3
        # optimal
        self.best_seed = None

    def max_route_cost(self, sol):
        rsortedlist = sorted(sol.routes, key=lambda r: r.cost, reverse=True)
        return rsortedlist[0].cost
    def max_route(self, sol):
        rsortedlist = sorted(sol.routes, key=lambda r: r.cost, reverse=True)
        return rsortedlist[0]

    def solve(self):
        for i in range(5, 25):
            self.SetRoutedFlagToFalseForAllCustomers()
            self.NewApplyNearestNeighborMethod(i)
            #self.BestFitReversed(i)
            self.tsp()
            cc = self.sol.cost
            print(i, 'Constr:', self.sol.cost, self.max_route_cost(self.sol))
            # self.MinimumInsertions(i)
            # self.ReportSolution(self.sol)
            # self.LocalSearch(2)  # 0 relocations, 1 swap, 2 twoopt
            # print('TWO OPT DONE! ')
            # self.LocalSearch(1)  # 0 relocations, 1 swap, 2 twoopt
            # print('RELO DONE! ')
            # self.LocalSearch(0)

            while True:

                self.LocalSearch(2)
                print('-LS(two opt) on BestOverall: ', self.sol.cost, self.max_route_cost(self.sol))
                c = self.max_route_cost(self.sol)
                # if a != b:
                #     continue

                self.LocalSearch(1)
                print('-LS(swap) on BestOverall: ', self.sol.cost, self.max_route_cost(self.sol))
                b = self.max_route_cost(self.sol)
                if c != b:
                    continue

                self.LocalSearch(0)
                print('-LS(relo) on BestOverall: ', self.sol.cost, self.max_route_cost(self.sol))
                a= self.max_route_cost(self.sol)
                if a != b:
                    continue

                if a == b and c == a:
                    break

            if self.overallBestSol == None or self.max_route_cost(self.overallBestSol) > self.max_route_cost(self.sol):
                self.overallBestSol = self.cloneSolution(self.sol)
                # optimal
                self.best_seed = i
            print(i, 'Const: ', cc, ' LS:', self.sol.cost, self.max_route_cost(self.sol), 'BestOverall: ',
                  self.overallBestSol.cost, self.max_route_cost(self.overallBestSol))
            # SolDrawer.draw(i, self.sol, self.allNodes)
        self.sol = self.overallBestSol

        # while True:
        #     self.LocalSearch(2)
        #     print('LS(2-opt) on BestOverall: ', self.sol.cost, self.max_route_cost(self.sol))
        #     a = self.max_route_cost(self.sol)
        #     self.LocalSearch(1)
        #     print('LS(swap) on BestOverall: ', self.sol.cost, self.max_route_cost(self.sol))
        #     b = self.max_route_cost(self.sol)
        #     if a == b:
        #         break

        # self.ReportSolution(self.sol)
        SolDrawer.draw(10000, self.sol, self.allNodes)
        # optimal
        print("best_seed: ", self.best_seed)
        return self.sol

    def SetRoutedFlagToFalseForAllCustomers(self):
        for i in range(0, len(self.customers)):
            self.customers[i].isRouted = False

    def tsp(self):
        cloneSol = self.cloneSolution(self.sol)
        cloneSol.cost = 0
        # self.sol.cost = 0
        # self.SetRoutedFlagToFalseForAllCustomers()
        for rt in cloneSol.routes:
            custs = rt.sequenceOfNodes[1:-1]
            del rt.sequenceOfNodes[1:]

            for cust in custs:
                cust.isRouted = False

            for i in range(0, len(custs)):
                min_cost = 10 ** 10
                insert_cust = None
                for cust in custs:
                    if cust.isRouted == True:
                        continue
                    trialCost = self.distanceMatrix[rt.sequenceOfNodes[-1].ID][cust.ID]
                    if trialCost < min_cost:
                        insert_cust = cust
                        min_cost = trialCost

                cloneSol.cost += self.distanceMatrix[rt.sequenceOfNodes[-1].ID][insert_cust.ID]
                rt.sequenceOfNodes.append(insert_cust)
                insert_cust.isRouted = True
            rt.sequenceOfNodes.append(rt.sequenceOfNodes[0])
            cloneSol.cost += self.distanceMatrix[rt.sequenceOfNodes[-2].ID][rt.sequenceOfNodes[-1].ID]
            self.UpdateRouteCostAndLoad(rt)
        #Mhpws if maxroutecost clonesol < maxroutecost self.sol then: (?)
        #if self.max_route_cost(cloneSol) < self.max_route_cost(self.sol):
        self.sol = cloneSol



    def BestFitReversed(self, itr=0):
        self.sol = Solution()
        random.seed(itr)
        sortedcust = sorted(self.customers, key = lambda x: x.demand, reverse = True)
        for cust in sortedcust:  # self.customers is a list of nodes(customers)
            rcl = []
            if cust.isRouted is False:
                route_builder = True
                if len(self.sol.routes) >= 25:  # I got 25 vehicles.
                    route_builder = False
                    for r in self.sol.routes:
                        # if r.cost == self.max_route_cost(self.sol):
                        #     continue
                        if r.load + cust.demand <= r.capacity:
                            trialcost = self.distanceMatrix[r.sequenceOfNodes[-2].ID][cust.ID]
                            if len(rcl) < self.rcl_size:
                                tup = (r.load, r, trialcost)
                                rcl.append(tup)
                                rcl.sort(key=lambda x: x[0])
                            elif r.load < rcl[-1][0]:  # check the last rcl element's load.
                                rcl.pop(len(rcl) - 1)
                                tup = (r.load, r, trialcost)
                                rcl.append(tup)
                                rcl.sort(key=lambda x: x[0])
                    if len(rcl) > 0:
                        tup_index = random.randint(0, len(rcl) - 1)
                        tpl = rcl[tup_index]  # which element to choose from rcl
                        bestInsertion = CustomerInsertion()
                        bestInsertion.customer = cust
                        bestInsertion.route = tpl[1]
                        bestInsertion.cost = tpl[2]
                        # print('Customer: ', str(cust.ID), ", added at route: ", str(tpl[1]))
                        self.ApplyCustomerInsertion(bestInsertion)
                if route_builder == True:
                    rt = Route(self.depot, self.capacity)
                    self.sol.routes.append(rt)
                    # print("New route added")
                    bestInsertion = CustomerInsertion()
                    bestInsertion.customer = cust
                    bestInsertion.route = rt
                    # rt.sequenceOfNodes[-2] should be the node 0, depo
                    bestInsertion.cost = self.distanceMatrix[rt.sequenceOfNodes[-2].ID][cust.ID]
                    self.ApplyCustomerInsertion(bestInsertion)

    def NewApplyNearestNeighborMethod(self, itr=0):
        self.sol = Solution()
        random.seed(itr)
        for cust in self.customers:  # self.customers is a list of nodes(customers)
            rcl = []
            if cust.isRouted is False:
                route_builder = True
                if len(self.sol.routes) >= 25:  # I got 25 vehicles.
                    route_builder = False
                    for r in self.sol.routes:
                        if r.cost == self.max_route_cost(self.sol):
                            continue
                        if r.load + cust.demand <= r.capacity:
                            trialcost = self.distanceMatrix[r.sequenceOfNodes[-2].ID][cust.ID]
                            if len(rcl) < self.rcl_size:
                                tup = (r.load, r, trialcost)
                                rcl.append(tup)
                                rcl.sort(key=lambda x: (x[2], x[0]))
                            elif trialcost < rcl[-1][2]:  # check the last rcl element's load.
                                rcl.pop(len(rcl) - 1)
                                tup = (r.load, r, trialcost)
                                rcl.append(tup)
                                rcl.sort(key=lambda x: (x[2], x[0]))
                    if len(rcl) > 0:
                        tup_index = random.randint(0, len(rcl) - 1)
                        tpl = rcl[tup_index]  # which element to choose from rcl
                        bestInsertion = CustomerInsertion()
                        bestInsertion.customer = cust
                        bestInsertion.route = tpl[1]
                        bestInsertion.cost = tpl[2]
                        # print('Customer: ', str(cust.ID), ", added at route: ", str(tpl[1]))
                        self.ApplyCustomerInsertion(bestInsertion)
                if route_builder == True:
                    rt = Route(self.depot, self.capacity)
                    self.sol.routes.append(rt)
                    # print("New route added")
                    bestInsertion = CustomerInsertion()
                    bestInsertion.customer = cust
                    bestInsertion.route = rt
                    # rt.sequenceOfNodes[-2] should be the node 0, depo
                    bestInsertion.cost = self.distanceMatrix[rt.sequenceOfNodes[-2].ID][cust.ID]
                    self.ApplyCustomerInsertion(bestInsertion)

    def ApplyNearestNeighborMethod(self, itr=0):
        modelIsFeasible = True
        self.sol = Solution()
        insertions = 0
        while (insertions < len(self.customers)):
            bestInsertion = CustomerInsertion()
            lastOpenRoute: Route = self.GetLastOpenRoute()

            if lastOpenRoute is not None:
                self.IdentifyBest_NN_ofLastVisited(bestInsertion, lastOpenRoute, itr)

            if (bestInsertion.customer is not None):
                self.ApplyCustomerInsertion(bestInsertion)
                insertions += 1
            else:
                # If there is an empty available route
                if lastOpenRoute is not None and len(lastOpenRoute.sequenceOfNodes) == 2:
                    modelIsFeasible = False
                    break
                else:
                    rt = Route(self.depot, self.capacity)
                    self.sol.routes.append(rt)

        if (modelIsFeasible == False):
            print('FeasibilityIssue')
            # reportSolution

    def MinimumInsertions(self, itr=0):
        modelIsFeasible = True
        self.sol = Solution()
        insertions = 0

        while (insertions < len(self.customers)):
            bestInsertion = CustomerInsertionAllPositions()
            lastOpenRoute: Route = self.GetLastOpenRoute()

            if lastOpenRoute is not None:
                self.IdentifyBestInsertionAllPositions(bestInsertion, lastOpenRoute, itr)

            if (bestInsertion.customer is not None):
                self.ApplyCustomerInsertionAllPositions(bestInsertion)
                insertions += 1
            else:
                # If there is an empty available route
                if lastOpenRoute is not None and len(lastOpenRoute.sequenceOfNodes) == 2:
                    modelIsFeasible = False
                    break
                # If there is no empty available route and no feasible insertion was identified
                else:
                    rt = Route(self.depot, self.capacity)
                    self.sol.routes.append(rt)

        if (modelIsFeasible == False):
            print('FeasibilityIssue')
            # reportSolution

        self.TestSolution()

    def LocalSearch(self, operator):
        self.bestSolution = self.cloneSolution(self.sol)
        terminationCondition = False
        localSearchIterator = 0

        rm = RelocationMove()
        sm = SwapMove()
        top = TwoOptMove()

        while terminationCondition is False:
            self.InitializeOperators(rm, sm, top)
            # if operator == 0:
            #     SolDrawer.draw(localSearchIterator, self.sol, self.allNodes)

            # Relocations
            if operator == 0:
                self.FindBestRelocationMove(rm)
                if rm.originRoutePosition is not None:
                    # if rm.moveCost < 0:
                    self.ApplyRelocationMove(rm)
                    # print('total cost: ', self.sol.cost)
                    # print("obj: ", self.max_route_cost(self.sol))
                else:
                    terminationCondition = True
            # Swaps
            elif operator == 1:
                self.FindBestSwapMove(sm)
                if sm.positionOfFirstRoute is not None:
                    #if sm.moveCost < 0:
                    self.ApplySwapMove(sm)
                    # print('total cost: ', self.sol.cost)
                    # print("obj: ", self.max_route_cost(self.sol))
                else:
                    terminationCondition = True
            elif operator == 2:
                self.FindBestTwoOptMove(top)
                if top.positionOfFirstRoute is not None:
                    #if top.moveCost < 0:
                    self.ApplyTwoOptMove(top)
                    # print('total cost: ', self.sol.cost)
                    # print("obj: ", self.max_route_cost(self.sol))
                else:
                    terminationCondition = True

            self.TestSolution()


            if (self.max_route_cost(self.sol) < self.max_route_cost(self.bestSolution)):
            # if (self.sol.cost < self.bestSolution.cost):
                self.bestSolution = self.cloneSolution(self.sol)
            localSearchIterator = localSearchIterator + 1

            # Print every 10 moves the tc, obj
            if divmod(localSearchIterator, 10)[1] == 0:
                print('total cost: ', self.sol.cost)
                print("obj: ", self.max_route_cost(self.sol))
            # if localSearchIterator >= 60:
            #     print('Loop Observed')
            #     break

        self.sol = self.bestSolution

    def cloneRoute(self, rt: Route):
        cloned = Route(self.depot, self.capacity)
        cloned.cost = rt.cost
        cloned.load = rt.load
        cloned.sequenceOfNodes = rt.sequenceOfNodes.copy()
        return cloned

    def cloneSolution(self, sol: Solution):
        cloned = Solution()
        for i in range(0, len(sol.routes)):
            rt = sol.routes[i]
            clonedRoute = self.cloneRoute(rt)
            cloned.routes.append(clonedRoute)
        cloned.cost = self.sol.cost
        return cloned

    def FindBestRelocationMove(self, rm):
        maxobj_dif = 0
        for originRouteIndex in range(0, len(self.sol.routes)):
            rt1: Route = self.sol.routes[originRouteIndex]
            for targetRouteIndex in range(0, len(self.sol.routes)):
                rt2: Route = self.sol.routes[targetRouteIndex]
                for originNodeIndex in range(1, len(rt1.sequenceOfNodes) - 1):
                    for targetNodeIndex in range(0, len(rt2.sequenceOfNodes) - 1):

                        if originRouteIndex == targetRouteIndex and (
                                targetNodeIndex == originNodeIndex or targetNodeIndex == originNodeIndex - 1):
                            continue

                        A = rt1.sequenceOfNodes[originNodeIndex - 1]
                        B = rt1.sequenceOfNodes[originNodeIndex]
                        C = rt1.sequenceOfNodes[originNodeIndex + 1]

                        F = rt2.sequenceOfNodes[targetNodeIndex]
                        G = rt2.sequenceOfNodes[targetNodeIndex + 1]

                        if rt1 != rt2:
                            if rt2.load + B.demand > rt2.capacity:
                                continue

                        costAdded = self.distanceMatrix[A.ID][C.ID] + self.distanceMatrix[F.ID][B.ID] + \
                                    self.distanceMatrix[B.ID][G.ID]
                        costRemoved = self.distanceMatrix[A.ID][B.ID] + self.distanceMatrix[B.ID][C.ID] + \
                                      self.distanceMatrix[F.ID][G.ID]

                        originRtCostChange = self.distanceMatrix[A.ID][C.ID] - self.distanceMatrix[A.ID][B.ID] - \
                                             self.distanceMatrix[B.ID][C.ID]
                        targetRtCostChange = self.distanceMatrix[F.ID][B.ID] + self.distanceMatrix[B.ID][G.ID] - \
                                             self.distanceMatrix[F.ID][G.ID]

                        moveCost = costAdded - costRemoved



                        rmtesting = RelocationMove()
                        rmtesting.Initialize()
                        self.StoreBestRelocationMove(originRouteIndex, targetRouteIndex, originNodeIndex,
                                                     targetNodeIndex, moveCost, originRtCostChange,
                                                     targetRtCostChange, rmtesting)

                        obj_dif = round(self.max_route_cost(self.sol) - self.max_route_cost(self.clonedSol_appliedmoveRel(rmtesting)), 2)

                        if obj_dif > maxobj_dif or (abs(obj_dif - maxobj_dif) <= 0.00000000000001 and moveCost < rm.moveCost and moveCost < -0.01):

                            self.StoreBestRelocationMove(originRouteIndex, targetRouteIndex, originNodeIndex,
                                                         targetNodeIndex, moveCost, originRtCostChange,
                                                         targetRtCostChange, rm)
                            maxobj_dif = obj_dif




    def FindBestSwapMove(self, sm):
        maxobj_dif = 0
        for firstRouteIndex in range(0, len(self.sol.routes)):
            rt1: Route = self.sol.routes[firstRouteIndex]
            for secondRouteIndex in range(firstRouteIndex, len(self.sol.routes)):
                rt2: Route = self.sol.routes[secondRouteIndex]
                for firstNodeIndex in range(1, len(rt1.sequenceOfNodes) - 1):
                    startOfSecondNodeIndex = 1
                    if rt1 == rt2:
                        startOfSecondNodeIndex = firstNodeIndex + 1
                    for secondNodeIndex in range(startOfSecondNodeIndex, len(rt2.sequenceOfNodes) - 1):

                        a1 = rt1.sequenceOfNodes[firstNodeIndex - 1]
                        b1 = rt1.sequenceOfNodes[firstNodeIndex]
                        c1 = rt1.sequenceOfNodes[firstNodeIndex + 1]

                        a2 = rt2.sequenceOfNodes[secondNodeIndex - 1]
                        b2 = rt2.sequenceOfNodes[secondNodeIndex]
                        c2 = rt2.sequenceOfNodes[secondNodeIndex + 1]

                        moveCost = None
                        costChangeFirstRoute = None
                        costChangeSecondRoute = None

                        if rt1 == rt2:
                            if firstNodeIndex == secondNodeIndex - 1:
                                costRemoved = self.distanceMatrix[a1.ID][b1.ID] + self.distanceMatrix[b1.ID][b2.ID] + \
                                              self.distanceMatrix[b2.ID][c2.ID]
                                costAdded = self.distanceMatrix[a1.ID][b2.ID] + self.distanceMatrix[b2.ID][b1.ID] + \
                                            self.distanceMatrix[b1.ID][c2.ID]
                                moveCost = costAdded - costRemoved
                            else:

                                costRemoved1 = self.distanceMatrix[a1.ID][b1.ID] + self.distanceMatrix[b1.ID][c1.ID]
                                costAdded1 = self.distanceMatrix[a1.ID][b2.ID] + self.distanceMatrix[b2.ID][c1.ID]
                                costRemoved2 = self.distanceMatrix[a2.ID][b2.ID] + self.distanceMatrix[b2.ID][c2.ID]
                                costAdded2 = self.distanceMatrix[a2.ID][b1.ID] + self.distanceMatrix[b1.ID][c2.ID]
                                moveCost = costAdded1 + costAdded2 - (costRemoved1 + costRemoved2)
                        else:
                            if rt1.load - b1.demand + b2.demand > self.capacity:
                                continue
                            if rt2.load - b2.demand + b1.demand > self.capacity:
                                continue

                            costRemoved1 = self.distanceMatrix[a1.ID][b1.ID] + self.distanceMatrix[b1.ID][c1.ID]
                            costAdded1 = self.distanceMatrix[a1.ID][b2.ID] + self.distanceMatrix[b2.ID][c1.ID]
                            costRemoved2 = self.distanceMatrix[a2.ID][b2.ID] + self.distanceMatrix[b2.ID][c2.ID]
                            costAdded2 = self.distanceMatrix[a2.ID][b1.ID] + self.distanceMatrix[b1.ID][c2.ID]

                            costChangeFirstRoute = costAdded1 - costRemoved1
                            costChangeSecondRoute = costAdded2 - costRemoved2

                            moveCost = costAdded1 + costAdded2 - (costRemoved1 + costRemoved2)

                        smtesting = SwapMove()
                        self.StoreBestSwapMove(firstRouteIndex, secondRouteIndex, firstNodeIndex, secondNodeIndex,
                                               moveCost, costChangeFirstRoute, costChangeSecondRoute, smtesting)
                        obj_dif = round(self.max_route_cost(self.sol) - self.max_route_cost(self.clonedSol_appliedsm(smtesting)), 2)
                        #if moveCost < sm.moveCost and abs(moveCost) > 0.0001:
                        if obj_dif > maxobj_dif or (abs(obj_dif - maxobj_dif) <= 0.00000000000001 and moveCost < sm.moveCost and moveCost < -0.01):
                            self.StoreBestSwapMove(firstRouteIndex, secondRouteIndex, firstNodeIndex, secondNodeIndex, moveCost, costChangeFirstRoute, costChangeSecondRoute, sm)
                            maxobj_dif = obj_dif

    def clonedSol_appliedmoveRel(self, rm: RelocationMove):
        cloneSol = self.cloneSolution(self.sol)

        oldCost = self.CalculateTotalCost(cloneSol)

        originRt = cloneSol.routes[rm.originRoutePosition]
        targetRt = cloneSol.routes[rm.targetRoutePosition]

        B = originRt.sequenceOfNodes[rm.originNodePosition]

        if originRt == targetRt:
            del originRt.sequenceOfNodes[rm.originNodePosition]
            if (rm.originNodePosition < rm.targetNodePosition):
                targetRt.sequenceOfNodes.insert(rm.targetNodePosition, B)
            else:
                targetRt.sequenceOfNodes.insert(rm.targetNodePosition + 1, B)

            originRt.cost += rm.moveCost
        else:
            del originRt.sequenceOfNodes[rm.originNodePosition]
            targetRt.sequenceOfNodes.insert(rm.targetNodePosition + 1, B)
            originRt.cost += rm.costChangeOriginRt
            targetRt.cost += rm.costChangeTargetRt
            originRt.load -= B.demand
            targetRt.load += B.demand

        cloneSol.cost += rm.moveCost

        newCost = self.CalculateTotalCost(cloneSol)
        # debuggingOnly
        if abs((newCost - oldCost) - rm.moveCost) > 0.0001:
            print('Cost Issue')
        return cloneSol

    def ApplyRelocationMove(self, rm: RelocationMove):

        oldCost = self.CalculateTotalCost(self.sol)

        originRt = self.sol.routes[rm.originRoutePosition]
        targetRt = self.sol.routes[rm.targetRoutePosition]

        B = originRt.sequenceOfNodes[rm.originNodePosition]

        if originRt == targetRt:
            del originRt.sequenceOfNodes[rm.originNodePosition]
            if (rm.originNodePosition < rm.targetNodePosition):
                targetRt.sequenceOfNodes.insert(rm.targetNodePosition, B)
            else:
                targetRt.sequenceOfNodes.insert(rm.targetNodePosition + 1, B)

            originRt.cost += rm.moveCost
        else:
            del originRt.sequenceOfNodes[rm.originNodePosition]
            targetRt.sequenceOfNodes.insert(rm.targetNodePosition + 1, B)
            originRt.cost += rm.costChangeOriginRt
            targetRt.cost += rm.costChangeTargetRt
            originRt.load -= B.demand
            targetRt.load += B.demand

        self.sol.cost += rm.moveCost

        newCost = self.CalculateTotalCost(self.sol)
        # debuggingOnly
        if abs((newCost - oldCost) - rm.moveCost) > 0.0001:
            print('Cost Issue')


    def clonedSol_appliedsm(self, sm):
        cloneSol = self.cloneSolution(self.sol)
        oldCost = self.CalculateTotalCost(self.sol)
        rt1 = cloneSol.routes[sm.positionOfFirstRoute]
        rt2 = cloneSol.routes[sm.positionOfSecondRoute]
        b1 = rt1.sequenceOfNodes[sm.positionOfFirstNode]
        b2 = rt2.sequenceOfNodes[sm.positionOfSecondNode]
        rt1.sequenceOfNodes[sm.positionOfFirstNode] = b2
        rt2.sequenceOfNodes[sm.positionOfSecondNode] = b1

        if (rt1 == rt2):
            rt1.cost += sm.moveCost
        else:
            rt1.cost += sm.costChangeFirstRt
            rt2.cost += sm.costChangeSecondRt
            rt1.load = rt1.load - b1.demand + b2.demand
            rt2.load = rt2.load + b1.demand - b2.demand

        cloneSol.cost += sm.moveCost

        newCost = self.CalculateTotalCost(cloneSol)
        # debuggingOnly
        if abs((newCost - oldCost) - sm.moveCost) > 0.0001:
            print('Cost Issue')
        return cloneSol

    def ApplySwapMove(self, sm):
        oldCost = self.CalculateTotalCost(self.sol)
        rt1 = self.sol.routes[sm.positionOfFirstRoute]
        rt2 = self.sol.routes[sm.positionOfSecondRoute]
        b1 = rt1.sequenceOfNodes[sm.positionOfFirstNode]
        b2 = rt2.sequenceOfNodes[sm.positionOfSecondNode]
        rt1.sequenceOfNodes[sm.positionOfFirstNode] = b2
        rt2.sequenceOfNodes[sm.positionOfSecondNode] = b1

        if (rt1 == rt2):
            rt1.cost += sm.moveCost
        else:
            rt1.cost += sm.costChangeFirstRt
            rt2.cost += sm.costChangeSecondRt
            rt1.load = rt1.load - b1.demand + b2.demand
            rt2.load = rt2.load + b1.demand - b2.demand

        self.sol.cost += sm.moveCost

        newCost = self.CalculateTotalCost(self.sol)
        # debuggingOnly
        if abs((newCost - oldCost) - sm.moveCost) > 0.0001:
            print('Cost Issue')

    def ReportSolution(self, sol):
        for i in range(0, len(sol.routes)):
            rt = sol.routes[i]
            for j in range(0, len(rt.sequenceOfNodes)):
                print(rt.sequenceOfNodes[j].ID, end=' ')
            print(rt.cost)
        print(self.sol.cost)

    def GetLastOpenRoute(self):
        if len(self.sol.routes) == 0:
            return None
        else:
            return self.sol.routes[-1]

    def IdentifyBest_NN_ofLastVisited(self, bestInsertion, rt, itr=0):
        random.seed(itr)
        rcl = []
        for i in range(0, len(self.customers)):
            candidateCust: Node = self.customers[i]
            if candidateCust.isRouted is False:
                if rt.load + candidateCust.demand <= rt.capacity:
                    lastNodePresentInTheRoute = rt.sequenceOfNodes[-2]
                    trialCost = self.distanceMatrix[lastNodePresentInTheRoute.ID][candidateCust.ID]
                    # Update rcl list
                    if len(rcl) < self.rcl_size:
                        new_tup = (trialCost, candidateCust, rt)
                        rcl.append(new_tup)
                        rcl.sort(key=lambda x: x[0])
                    elif trialCost < rcl[-1][0]:
                        rcl.pop(len(rcl) - 1)
                        new_tup = (trialCost, candidateCust, rt)
                        rcl.append(new_tup)
                        rcl.sort(key=lambda x: x[0])
            if len(rcl) > 0:
                tup_index = random.randint(0, len(rcl) - 1)
                tpl = rcl[tup_index]
                bestInsertion.cost = tpl[0]
                bestInsertion.customer = tpl[1]
                bestInsertion.route = tpl[2]

    def ApplyCustomerInsertion(self, insertion):
        insCustomer = insertion.customer
        rt = insertion.route
        # before the second depot occurrence
        insIndex = len(rt.sequenceOfNodes) - 1
        rt.sequenceOfNodes.insert(insIndex, insCustomer)

        beforeInserted = rt.sequenceOfNodes[-3]

        # Cust to depot in the matrix are 0.0
        costAdded = self.distanceMatrix[beforeInserted.ID][insCustomer.ID] + self.distanceMatrix[insCustomer.ID][
            self.depot.ID]
        costRemoved = self.distanceMatrix[beforeInserted.ID][self.depot.ID]

        rt.cost += costAdded - costRemoved
        self.sol.cost += costAdded - costRemoved

        rt.load += insCustomer.demand

        insCustomer.isRouted = True

    def StoreBestRelocationMove(self, originRouteIndex, targetRouteIndex, originNodeIndex, targetNodeIndex, moveCost,
                                originRtCostChange, targetRtCostChange, rm: RelocationMove):
        rm.originRoutePosition = originRouteIndex
        rm.originNodePosition = originNodeIndex
        rm.targetRoutePosition = targetRouteIndex
        rm.targetNodePosition = targetNodeIndex
        rm.costChangeOriginRt = originRtCostChange
        rm.costChangeTargetRt = targetRtCostChange
        rm.moveCost = moveCost

    def StoreBestSwapMove(self, firstRouteIndex, secondRouteIndex, firstNodeIndex, secondNodeIndex, moveCost,
                          costChangeFirstRoute, costChangeSecondRoute, sm):
        sm.positionOfFirstRoute = firstRouteIndex
        sm.positionOfSecondRoute = secondRouteIndex
        sm.positionOfFirstNode = firstNodeIndex
        sm.positionOfSecondNode = secondNodeIndex
        sm.costChangeFirstRt = costChangeFirstRoute
        sm.costChangeSecondRt = costChangeSecondRoute
        sm.moveCost = moveCost

    def CalculateTotalCost(self, sol):
        c = 0
        for i in range(0, len(sol.routes)):
            rt = sol.routes[i]
            for j in range(0, len(rt.sequenceOfNodes) - 1):
                a = rt.sequenceOfNodes[j]
                b = rt.sequenceOfNodes[j + 1]
                c += self.distanceMatrix[a.ID][b.ID]
        return c

    def InitializeOperators(self, rm, sm, top):
        rm.Initialize()
        sm.Initialize()
        top.Initialize()

    def FindBestTwoOptMove(self, top):
        maxobj_dif = 0
        for rtInd1 in range(0, len(self.sol.routes)):
            rt1: Route = self.sol.routes[rtInd1]
            for rtInd2 in range(rtInd1, len(self.sol.routes)):
                rt2: Route = self.sol.routes[rtInd2]
                for nodeInd1 in range(0, len(rt1.sequenceOfNodes) - 1):
                    start2 = 0
                    if (rt1 == rt2):
                        start2 = nodeInd1 + 2

                    for nodeInd2 in range(start2, len(rt2.sequenceOfNodes) - 1):
                        moveCost = 10 ** 9

                        A = rt1.sequenceOfNodes[nodeInd1]
                        B = rt1.sequenceOfNodes[nodeInd1 + 1]
                        K = rt2.sequenceOfNodes[nodeInd2]
                        L = rt2.sequenceOfNodes[nodeInd2 + 1]

                        if rt1 == rt2:
                            if nodeInd1 == 0 and nodeInd2 == len(rt1.sequenceOfNodes) - 2:
                                continue
                            costAdded = self.distanceMatrix[A.ID][K.ID] + self.distanceMatrix[B.ID][L.ID]
                            for n in range(nodeInd1 + 1, nodeInd2):
                                a = rt1.sequenceOfNodes[n]
                                b = rt1.sequenceOfNodes[n + 1]
                                costAdded += self.distanceMatrix[b.ID][a.ID]
                            costRemoved = 0
                            for n in range(nodeInd1, nodeInd2 + 1):
                                a = rt1.sequenceOfNodes[n]
                                b = rt1.sequenceOfNodes[n + 1]
                                costRemoved += self.distanceMatrix[a.ID][b.ID]
                            #costRemoved = self.distanceMatrix[A.ID][B.ID] + self.distanceMatrix[K.ID][L.ID] + self.distanceMatrix[B.ID][K.ID]
                            moveCost = costAdded - costRemoved
                        else:
                            if nodeInd1 == 0 and nodeInd2 == 0:
                                continue
                            if nodeInd1 == len(rt1.sequenceOfNodes) - 2 and nodeInd2 == len(rt2.sequenceOfNodes) - 2:
                                continue

                            if self.CapacityIsViolated(rt1, nodeInd1, rt2, nodeInd2):
                                continue
                            costAdded = self.distanceMatrix[A.ID][L.ID] + self.distanceMatrix[K.ID][B.ID]
                            costRemoved = self.distanceMatrix[A.ID][B.ID] + self.distanceMatrix[K.ID][L.ID]
                            moveCost = costAdded - costRemoved


                        toptesting = TwoOptMove()
                        self.StoreBestTwoOptMove(rtInd1, rtInd2, nodeInd1, nodeInd2, moveCost, toptesting)
                        obj_dif = round(self.max_route_cost(self.sol) - self.max_route_cost(self.clonedSol_appliedtop(toptesting)), 2)
                        #if moveCost < top.moveCost and abs(moveCost) > 0.0001:

                        if obj_dif > maxobj_dif or (abs(obj_dif - maxobj_dif) <= 0.00000000000001 and moveCost < top.moveCost and moveCost < -0.01):
                            self.StoreBestTwoOptMove(rtInd1, rtInd2, nodeInd1, nodeInd2, moveCost, top)
                            # tabu_list.append(sol.routes[rtInd1].sequenceOfNodes[nodeInd1].ID,moveCost)
                            maxobj_dif = obj_dif

    def CapacityIsViolated(self, rt1, nodeInd1, rt2, nodeInd2):

        rt1FirstSegmentLoad = 0
        for i in range(0, nodeInd1 + 1):
            n = rt1.sequenceOfNodes[i]
            rt1FirstSegmentLoad += n.demand
        rt1SecondSegmentLoad = rt1.load - rt1FirstSegmentLoad

        rt2FirstSegmentLoad = 0
        for i in range(0, nodeInd2 + 1):
            n = rt2.sequenceOfNodes[i]
            rt2FirstSegmentLoad += n.demand
        rt2SecondSegmentLoad = rt2.load - rt2FirstSegmentLoad

        if (rt1FirstSegmentLoad + rt2SecondSegmentLoad > rt1.capacity):
            return True
        if (rt2FirstSegmentLoad + rt1SecondSegmentLoad > rt2.capacity):
            return True

        return False

    def StoreBestTwoOptMove(self, rtInd1, rtInd2, nodeInd1, nodeInd2, moveCost, top):
        top.positionOfFirstRoute = rtInd1
        top.positionOfSecondRoute = rtInd2
        top.positionOfFirstNode = nodeInd1
        top.positionOfSecondNode = nodeInd2
        top.moveCost = moveCost


    def clonedSol_appliedtop(self, top):
        cloneSol = self.cloneSolution(self.sol)
        rt1: Route = cloneSol.routes[top.positionOfFirstRoute]
        rt2: Route = cloneSol.routes[top.positionOfSecondRoute]

        if rt1 == rt2:
            # reverses the nodes in the segment [positionOfFirstNode + 1,  top.positionOfSecondNode]
            reversedSegment = reversed(rt1.sequenceOfNodes[top.positionOfFirstNode + 1: top.positionOfSecondNode + 1])
            # lst = list(reversedSegment)
            # lst2 = list(reversedSegment)
            rt1.sequenceOfNodes[top.positionOfFirstNode + 1: top.positionOfSecondNode + 1] = reversedSegment
            ##rt1.sequenceOfNodes[top.positionOfFirstNode + 1], rt1.sequenceOfNodes[top.positionOfSecondNode] = rt1.sequenceOfNodes[top.positionOfSecondNode], rt1.sequenceOfNodes[top.positionOfFirstNode + 1]

            # reversedSegmentList = list(reversed(rt1.sequenceOfNodes[top.positionOfFirstNode + 1: top.positionOfSecondNode + 1]))
            # rt1.sequenceOfNodes[top.positionOfFirstNode + 1: top.positionOfSecondNode + 1] = reversedSegmentList

            rt1.cost += top.moveCost

        else:
            # slice with the nodes from position top.positionOfFirstNode + 1 onwards
            relocatedSegmentOfRt1 = rt1.sequenceOfNodes[top.positionOfFirstNode + 1:]

            # slice with the nodes from position top.positionOfFirstNode + 1 onwards
            relocatedSegmentOfRt2 = rt2.sequenceOfNodes[top.positionOfSecondNode + 1:]

            del rt1.sequenceOfNodes[top.positionOfFirstNode + 1:]
            del rt2.sequenceOfNodes[top.positionOfSecondNode + 1:]

            rt1.sequenceOfNodes.extend(relocatedSegmentOfRt2)
            rt2.sequenceOfNodes.extend(relocatedSegmentOfRt1)

            self.UpdateRouteCostAndLoad(rt1)
            self.UpdateRouteCostAndLoad(rt2)

        cloneSol.cost += top.moveCost
        return cloneSol

    def ApplyTwoOptMove(self, top):
        rt1: Route = self.sol.routes[top.positionOfFirstRoute]
        rt2: Route = self.sol.routes[top.positionOfSecondRoute]

        if rt1 == rt2:
            # reverses the nodes in the segment [positionOfFirstNode + 1,  top.positionOfSecondNode]
            reversedSegment = reversed(rt1.sequenceOfNodes[top.positionOfFirstNode + 1: top.positionOfSecondNode + 1])
            # lst = list(reversedSegment)
            # lst2 = list(reversedSegment)
            rt1.sequenceOfNodes[top.positionOfFirstNode + 1: top.positionOfSecondNode + 1] = reversedSegment
            ##rt1.sequenceOfNodes[top.positionOfFirstNode + 1], rt1.sequenceOfNodes[top.positionOfSecondNode] = rt1.sequenceOfNodes[top.positionOfSecondNode], rt1.sequenceOfNodes[top.positionOfFirstNode + 1]

            # reversedSegmentList = list(reversed(rt1.sequenceOfNodes[top.positionOfFirstNode + 1: top.positionOfSecondNode + 1]))
            # rt1.sequenceOfNodes[top.positionOfFirstNode + 1: top.positionOfSecondNode + 1] = reversedSegmentList

            rt1.cost += top.moveCost

        else:
            # slice with the nodes from position top.positionOfFirstNode + 1 onwards
            relocatedSegmentOfRt1 = rt1.sequenceOfNodes[top.positionOfFirstNode + 1:]

            # slice with the nodes from position top.positionOfFirstNode + 1 onwards
            relocatedSegmentOfRt2 = rt2.sequenceOfNodes[top.positionOfSecondNode + 1:]

            del rt1.sequenceOfNodes[top.positionOfFirstNode + 1:]
            del rt2.sequenceOfNodes[top.positionOfSecondNode + 1:]

            rt1.sequenceOfNodes.extend(relocatedSegmentOfRt2)
            rt2.sequenceOfNodes.extend(relocatedSegmentOfRt1)

            self.UpdateRouteCostAndLoad(rt1)
            self.UpdateRouteCostAndLoad(rt2)

        self.sol.cost += top.moveCost

    def UpdateRouteCostAndLoad(self, rt: Route):
        tc = 0
        tl = 0
        for i in range(0, len(rt.sequenceOfNodes) - 1):
            A = rt.sequenceOfNodes[i]
            B = rt.sequenceOfNodes[i + 1]
            tc += self.distanceMatrix[A.ID][B.ID]
            tl += A.demand
        rt.load = tl
        rt.cost = tc

    def TestSolution(self):
        totalSolCost = 0
        for r in range(0, len(self.sol.routes)):
            rt: Route = self.sol.routes[r]
            rtCost = 0
            rtLoad = 0
            for n in range(0, len(rt.sequenceOfNodes) - 1):
                A = rt.sequenceOfNodes[n]
                B = rt.sequenceOfNodes[n + 1]
                rtCost += self.distanceMatrix[A.ID][B.ID]
                rtLoad += A.demand
            if abs(rtCost - rt.cost) > 0.001:
                print ('Route Cost problem- diff: ',rtCost - rt.cost )
            if rtLoad != rt.load:
                print('Route Load problem')

            totalSolCost += rt.cost

        # if abs(totalSolCost - self.sol.cost) > 0.0001:
        #     print('Solution Cost problem')

    def IdentifyBestInsertionAllPositions(self, bestInsertion, rt, itr=0):
        random.seed(itr)
        rcl = []
        for i in range(0, len(self.customers)):
            candidateCust: Node = self.customers[i]
            if candidateCust.isRouted is False:
                if rt.load + candidateCust.demand <= rt.capacity:
                    for j in range(0, len(rt.sequenceOfNodes) - 1):
                        A = rt.sequenceOfNodes[j]
                        B = rt.sequenceOfNodes[j + 1]
                        costAdded = self.distanceMatrix[A.ID][candidateCust.ID] + self.distanceMatrix[candidateCust.ID][
                            B.ID]
                        costRemoved = self.distanceMatrix[A.ID][B.ID]
                        trialCost = costAdded - costRemoved

                        if len(rcl) < self.rcl_size:
                            new_tup = (trialCost, candidateCust, rt, j)
                            rcl.append(new_tup)
                            rcl.sort(key=lambda x: x[0])
                        elif trialCost < rcl[-1][0]:
                            rcl.pop(len(rcl) - 1)
                            new_tup = (trialCost, candidateCust, rt, j)
                            rcl.append(new_tup)
                            rcl.sort(key=lambda x: x[0])
        tup_index = random.randint(0, len(self.rcl) - 1)
        tpl = rcl[tup_index]
        bestInsertion.cost = tpl[0]
        bestInsertion.customer = tpl[1]
        bestInsertion.route = tpl[2]
        bestInsertion.insertionPosition = tpl[3]

    def ApplyCustomerInsertionAllPositions(self, insertion):
        insCustomer = insertion.customer
        rt = insertion.route
        # before the second depot occurrence
        insIndex = insertion.insertionPosition
        rt.sequenceOfNodes.insert(insIndex + 1, insCustomer)
        rt.cost += insertion.cost
        self.sol.cost += insertion.cost
        rt.load += insCustomer.demand
        insCustomer.isRouted = True

