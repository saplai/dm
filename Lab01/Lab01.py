
import networkx as nx
import matplotlib.pyplot as plot
import numpy as np


def getFile():
    text = []
    with open("l1-3.txt", "r") as file:
        lines = file.readlines()
        for i in lines:
            text.append(i.replace('\n', ''))
    for i in range(len(text)):
        text[i] = text[i].split(" ")

    return text


def addNewPlot(arrayOld):
    newPlot = nx.Graph()
    for i in arrayOld:
        newPlot.add_edge(i[0], i[1], weight=i[2])

    pos = nx.spring_layout(newPlot)
    labels = nx.get_edge_attributes(newPlot, 'weight')

    nx.draw(newPlot, pos, with_labels=True)
    nx.draw_networkx_edge_labels(newPlot, pos, edge_labels=labels)
    nx.draw_networkx_nodes(newPlot, pos, node_color='green', node_size=500)

    plot.show()


class Edge:
    def __init__(self, weight, src, dest):
        self.weight = weight
        self.dest = dest
        self.src = src
        self.next = None


class State:
    def __init__(self, parent, rank):
        self.parent = parent
        self.rank = rank


class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.graphEdge = []
        i = 0
        while (i < self.vertices):
            self.graphEdge.append([])
            i += 1

    def addEdge(self, src, dest, w):
        if (dest < 0 or dest >= self.vertices or src < 0 or src >= self.vertices):
            return

        # додати ребро
        self.graphEdge[src].append(Edge(w, src, dest))
        if (dest == src):
            return

        self.graphEdge[dest].append(Edge(w, dest, src))

    def printGraph(self):
        print("\n Список суміжності графів ", end="")
        i = 0
        while (i < self.vertices):
            print(" \n [", i, "] :", end="")
            j = 0
            # ітерація вузла i
            while (j < len(self.graphEdge[i])):
                print("  ", self.graphEdge[i][j].dest, end="")
                j += 1

            i += 1

    def find(self, subsets, i):
        if (subsets[i].parent != i):
            subsets[i].parent = self.find(subsets, subsets[i].parent)

        return subsets[i].parent

    def findUnion(self, subsets, x, y):
        a = self.find(subsets, x)
        b = self.find(subsets, y)
        if (subsets[a].rank < subsets[b].rank):
            subsets[a].parent = b
        elif (subsets[a].rank > subsets[b].rank):
            subsets[b].parent = a
        else:
            subsets[b].parent = a
            subsets[a].rank += 1

    def boruvka(self):
        arr = np.empty((0, 3), int)
        # містить суму ваги
        result = 0
        selector = self.vertices
        subsets = [None] * (self.vertices)
        cheapest = [None] * (self.vertices)
        v = 0
        while (v < self.vertices):
            subsets[v] = State(v, 0)
            v += 1

        while (selector > 1):
            v = 0
            while (v < self.vertices):
                cheapest[v] = None
                v += 1

            k = 0
            while (k < self.vertices):
                i = 0
                while (i < len(self.graphEdge[k])):
                    set1 = self.find(subsets, self.graphEdge[k][i].src)
                    set2 = self.find(subsets, self.graphEdge[k][i].dest)
                    if (set1 != set2):
                        if (cheapest[k] == None):
                            cheapest[k] = self.graphEdge[k][i]
                        elif (cheapest[k].weight > self.graphEdge[k][i].weight):
                            cheapest[k] = self.graphEdge[k][i]

                    i += 1

                k += 1

            i = 0
            while (i < self.vertices):
                if (cheapest[i] != None):
                    set1 = self.find(subsets, cheapest[i].src)
                    set2 = self.find(subsets, cheapest[i].dest)
                    if (set1 != set2):
                        selector -= 1
                        self.findUnion(subsets, set1, set2)
                        # відобразити з’єднання
                        print("\n Включити ребро (", cheapest[i].src, " - ", cheapest[i].dest, ") вагою",
                              cheapest[i].weight, end="")
                        # додати вагу
                        result += cheapest[i].weight

                        newRow = [cheapest[i].src,
                                  cheapest[i].dest, cheapest[i].weight]
                        arr = np.vstack([arr, newRow])
                i += 1
        arrayOld = getFile()
        arrayOld = [[int(j) if '.' not in j else float(j)
                     for j in i] for i in arrayOld]
        numberOfNodes = arrayOld[0][0]
        arrayNumbers = np.empty((0, numberOfNodes), int)
        for i in arrayOld[1:]:
            arrayNumbers = np.vstack([arrayNumbers, i])

        res = np.where(arrayNumbers != 0)
        oneArray = np.column_stack((res[0], res[1]))

        arrayDef = np.empty((0, 1), int)
        for i in oneArray:
            newRow = arrayNumbers[i[0]][i[1]]
            arrayDef = np.vstack([arrayDef, newRow])
        oneArray2 = np.column_stack((res[0], res[1], arrayDef))

        newPlot = nx.Graph()
        for i in oneArray2:
            newPlot.add_edge(i[0], i[1], weight=i[2])

        newOnePLot = nx.Graph()

        for i in arr:
            newOnePLot.add_edge(i[0], i[1], weight=i[2])
        edges_list = list(newOnePLot.edges)

        pos = nx.spring_layout(newPlot)
        labels1 = nx.get_edge_attributes(newOnePLot, 'weight')
        labels2 = nx.get_edge_attributes(newPlot, 'weight')

        nx.draw_networkx_edge_labels(newPlot, pos, edge_labels=labels1)
        nx.draw_networkx_edge_labels(newOnePLot, pos, edge_labels=labels2)
        nx.draw_networkx_edges(newPlot, pos, edge_color="blue", width=0.5)

        nx.draw(newOnePLot, pos, with_labels=True,
                edgelist=edges_list,
                edge_color="green",
                node_size=200,
                width=3,)

        plot.show()

        print("\n Розрахована загальна вага складає ", result)


if __name__ == "__main__":

    arrayOld = getFile()
    arrayOld = [[int(j) if '.' not in j else float(j)
                 for j in i] for i in arrayOld]
    numberOfNodes = arrayOld[0][0]
    arrayNumbers = np.empty((0, numberOfNodes), int)
    for i in arrayOld[1:]:
        arrayNumbers = np.vstack([arrayNumbers, i])

    res = np.where(arrayNumbers != 0)
    oneArray = np.column_stack((res[0], res[1]))

    arrayDef = np.empty((0, 1), int)
    for i in oneArray:
        newRow = arrayNumbers[i[0]][i[1]]
        arrayDef = np.vstack([arrayDef, newRow])
    oneArray2 = np.column_stack((res[0], res[1], arrayDef))

    g = Graph(numberOfNodes)

    for i in oneArray2:
        g.addEdge(i[0], i[1], i[2])

    g.boruvka()
