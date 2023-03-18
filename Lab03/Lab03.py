import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx

global numberOfNodes


def getFile():
    text = []
    with open("l3-3.txt", "r") as file:
        lines = file.readlines()
        for i in lines:
            text.append(i.replace('\n', ''))
    for i in range(len(text)):
        text[i] = text[i].split(" ")

    text2 = [[int(j) if '.' not in j else float(j) for j in i] for i in text]

    global numberOfNodes
    numberOfNodes = text2[0][0]
    return text2


fileArr = getFile()
arrayNumbers = np.empty((0, numberOfNodes), int)
for i in fileArr[1:]:
    arrayNumbers = np.vstack([arrayNumbers, i])
print(arrayNumbers)

newFileArr = fileArr[1:]


def Min(lst, myindex):
    return min(x for idx, x in enumerate(lst) if idx != myindex)


def Delete(matrix, index1, index2):
    del matrix[index1]
    for i in matrix:
        del i[index2]
    return matrix


def PrintMatrix(matrix):
    print("---------------")
    for i in range(len(matrix)):
        print(matrix[i])
    print("---------------")


n = numberOfNodes
matrix = []
H = 0
PathLenght = 0
Str = []
Stb = []
res = []
result = []
StartMatrix = []

for i in range(n):
    Str.append(i)
    Stb.append(i)

for i in range(n):
    StartMatrix.append(newFileArr[i].copy())

for i in range(n):
    newFileArr[i][i] = float('inf')

arrayNumbers = np.empty((0, numberOfNodes), int)
for i in newFileArr:
    arrayNumbers = np.vstack([arrayNumbers, i])
print(arrayNumbers)

newRes = np.where(arrayNumbers != float('inf'))
oneArray = np.column_stack((newRes[0], newRes[1]))

arrayDef = np.empty((0, 1), int)
for i in oneArray:
    newRow = newFileArr[i[0]][i[1]]
    arrayDef = np.vstack([arrayDef, newRow])
oneArray2 = np.column_stack((newRes[0], newRes[1], arrayDef))

while True:
    for i in range(len(newFileArr)):
        temp = min(newFileArr[i])
        H += temp
        for j in range(len(newFileArr)):
            newFileArr[i][j] -= temp

    for i in range(len(newFileArr)):
        temp = min(row[i] for row in newFileArr)
        H += temp
        for j in range(len(newFileArr)):
            newFileArr[j][i] -= temp

    NullMax = 0
    index1 = 0
    index2 = 0
    tmp = 0
    for i in range(len(newFileArr)):
        for j in range(len(newFileArr)):
            if newFileArr[i][j] == 0:
                tmp = Min(newFileArr[i], j) + Min((row[j]
                                                   for row in newFileArr), i)
                if tmp >= NullMax:
                    NullMax = tmp
                    index1 = i
                    index2 = j

    res.append(Str[index1] + 1)
    res.append(Stb[index2] + 1)

    oldIndex1 = Str[index1]
    oldIndex2 = Stb[index2]
    if oldIndex2 in Str and oldIndex1 in Stb:
        NewIndex1 = Str.index(oldIndex2)
        NewIndex2 = Stb.index(oldIndex1)
        newFileArr[NewIndex1][NewIndex2] = float('inf')
    del Str[index1]
    del Stb[index2]
    newFileArr = Delete(newFileArr, index1, index2)
    if len(newFileArr) == 1:
        break

for i in range(0, len(res) - 1, 2):
    if res.count(res[i]) < 2:
        result.append(res[i])
        result.append(res[i + 1])
for i in range(0, len(res) - 1, 2):
    for j in range(0, len(res) - 1, 2):
        if result[len(result) - 1] == res[j]:
            result.append(res[j])
            result.append(res[j + 1])

for i in range(0, len(result) - 1, 2):
    if i == len(result) - 2:
        PathLenght += StartMatrix[result[i] - 1][result[i + 1] - 1]
        PathLenght += StartMatrix[result[i + 1] - 1][result[0] - 1]
    else:
        PathLenght += StartMatrix[result[i] - 1][result[i + 1] - 1]
for i in range(len(result)):
    # print('- ', result[i])
    result[i] -= 1
print("----------------------------------")
tspArray = np.array(result)
tspArray.resize((numberOfNodes - 1, 2), refcheck=False)
print('Path:', ' -> '.join(map(str, tspArray)))
print(PathLenght)
print("----------------------------------")

graph = nx.DiGraph()
for i in oneArray2:
    graph.add_edge(i[0], i[1], weight=i[2])
pos = nx.spring_layout(graph, seed=399)
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
straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}
nx.draw_networkx_edge_labels(graph, pos, ax=ax, label_pos=0.1, font_size=5, edge_labels=straight_edge_labels,
                             rotate=False)

graph2 = nx.DiGraph()

for i in tspArray:
    graph2.add_edge(i[0], i[1])
nx.draw_networkx_edges(graph2, pos,  connectionstyle=f'arc3, rad = {arc_rad}',
                       edge_color="green",
                       node_size=200,
                       width=3, )

plt.show()
