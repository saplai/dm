from prettytable import PrettyTable
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

global numberOfNodes
global pairs


def getFile():
    text = []
    with open("l1-3.txt", "r") as file:
        lines = file.readlines()
        for i in lines:
            text.append(i.replace('\n', ''))
    for i in range(len(text)):
        text[i] = text[i].split(" ")

    text = [[int(j) if '.' not in j else float(j) for j in i] for i in text]

    global numberOfNodes
    numberOfNodes = text[0][0]
    arrayNumbers = np.empty((0, numberOfNodes), int)
    for i in text[1:]:
        arrayNumbers = np.vstack([arrayNumbers, i])

    return arrayNumbers


def dijkstra(matrix, source, dest):
    shortest = [0 for i in range(len(matrix))]
    selected = [source]
    N = len(matrix)

    inf = 9999999
    min = inf
    for i in range(N):
        if (i == source):
            shortest[source] = 0
        else:
            if (matrix[source][i] == 0):
                shortest[i] = inf
            else:
                shortest[i] = matrix[source][i]
                if (shortest[i] < min):
                    min = shortest[i]
                    ind = i

    if (source == dest):
        return 0

    selected.append(ind)
    while (ind != dest):
        for i in range(N):
            if i not in selected:
                if (matrix[ind][i] != 0):
                    if ((matrix[ind][i] + min) < shortest[i]):
                        shortest[i] = matrix[ind][i] + min
        tempMin = 9999999

        for j in range(N):
            if j not in selected:
                if (shortest[j] < tempMin):
                    tempMin = shortest[j]
                    ind = j
        min = tempMin
        selected.append(ind)

    return shortest[dest]


def getOdd(matrix):
    degrees = [0 for i in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] != 0:
                degrees[i] += 1

    odds = [i for i in range(len(degrees)) if degrees[i] % 2 != 0]
    print('Непарні вузли:', odds)

    return odds


def createPairs(odds):
    global pairs
    pairs = []
    for i in range(len(odds) - 1):
        pairs.append([])
        for j in range(i + 1, len(odds)):
            pairs[i].append([odds[i], odds[j]])

    print('Наступні пари:',  end=' ')
    for i in pairs:
        print(i, end=', ')
    print('')

    return pairs


def sumEdges(matrix):
    sum = 0
    l = len(matrix)
    for i in range(l):
        for j in range(i, l):
            sum += matrix[i][j]
    return sum


def chinesePostman(matrix):
    odds = getOdd(matrix)
    if (len(odds) == 0):
        return sumEdges(matrix)
    pairs = createPairs(odds)
    l = (len(pairs) + 1) // 2

    pairingsSum = []

    def getPairs(pairs, done=[], final=[]):
        if (pairs[0][0][0] not in done):
            done.append(pairs[0][0][0])

            for i in pairs[0]:
                f = final[:]
                val = done[:]
                if (i[1] not in val):
                    f.append(i)
                else:
                    continue

                if (len(f) == l):
                    pairingsSum.append(f)
                    return
                else:
                    val.append(i[1])
                    getPairs(pairs[1:], val, f)
        else:
            getPairs(pairs[1:], done, final)

    getPairs(pairs)
    minSums = []

    for i in pairingsSum:
        s = 0
        for j in range(len(i)):
            s += dijkstra(matrix, i[j][0], i[j][1])
        minSums.append(s)

    minimum = min(minSums)
    chinese_dis = minimum + sumEdges(matrix)

    return chinese_dis


if __name__ == '__main__':
    array = getFile()

    res = np.where(array != 0)
    oneArray = np.column_stack((res[0], res[1]))
    arrayDef = np.empty((0, 1), int)
    for i in oneArray:
        newRow = array[i[0]][i[1]]
        arrayDef = np.vstack([arrayDef, newRow])
    oneArray2 = np.column_stack((res[0], res[1], arrayDef))

    table = PrettyTable([chr(i) for i in range(65, 65 + numberOfNodes)])
    for i in array:
        table.add_row(i)
    print(table)

    print('Мінімальна дистанція алгоритму:', chinesePostman(array))

    graph = nx.DiGraph()
    for i in oneArray2:
        graph.add_edge(i[0], i[1], weight=i[2])
    pos = nx.spring_layout(graph, seed=numberOfNodes)

    labels = nx.get_edge_attributes(graph, 'weight')
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(graph, pos, ax=ax)
    nx.draw_networkx_labels(graph, pos, ax=ax)

    curved_edges = [edge for edge in graph.edges() if reversed(edge)
                    in graph.edges()]
    straight_edges = list(set(graph.edges()) - set(curved_edges))
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=straight_edges)
    arc_rad = 0.3
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=curved_edges,
                           connectionstyle=f'arc3, rad = {arc_rad}')

    edge_weights = nx.get_edge_attributes(graph, 'weight')
    curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
    straight_edge_labels = {
        edge: edge_weights[edge] for edge in straight_edges}
    nx.draw_networkx_edge_labels(graph, pos, ax=ax, label_pos=0.1,
                                 font_size=5, edge_labels=straight_edge_labels, rotate=False)

    graph1 = nx.DiGraph()
    newOneArray = np.concatenate(pairs, axis=0)
    print(newOneArray)
    for i in newOneArray:
        graph1.add_edge(i[0], i[1])
    nx.draw_networkx_edges(graph1, pos, ax=ax, connectionstyle=f'arc3, rad = {arc_rad}',
                           edge_color="green",
                           node_size=200,
                           width=3, )

    plt.axis('off')
    plt.show()
