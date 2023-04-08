import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from prettytable import PrettyTable


def graph_order(matrix):
    if len(matrix) != len(matrix[0]):
        return -1
    else:
        return len(matrix)


def matrix_degrees(matrix):
    degrees = []
    for i in range(len(matrix)):
        degrees.append(sum(matrix[i]))
    degrees.sort(reverse=True)
    return degrees


def permutations(matrix):
    if graph_order(matrix) > 8:
        return -1
    all_matrix = []
    idx = list(range(len(matrix)))
    possible_combinations = [list(i)
                             for i in itertools.permutations(idx, len(idx))]
    for i in possible_combinations:
        a = matrix
        a = a[i]
        a = np.transpose(np.transpose(a)[i])
        all_matrix.append({"perm_vertex": i, "adj_matrix": a})

    return all_matrix


def graph_isomporphism(first_matrix, second_matrix):
    first_matrix_degrees = matrix_degrees(first_matrix)
    second_matrix_degrees = matrix_degrees(second_matrix)
    if graph_order(first_matrix) != graph_order(first_matrix):
        return 'НЕ ІЗОМОРФІЧНИМИ'
    elif np.array_equal(first_matrix_degrees, second_matrix_degrees) == False:
        return 'НЕ ІЗОМОРФІЧНИМИ'
    else:
        for i in list(map(lambda matrix: matrix["adj_matrix"], permutations(second_matrix))):
            if np.array_equal(first_matrix, i) == True:
                return 'ІЗОМОРФІЧНИМИ'
    return False


if __name__ == '__main__':
    fig = plt.figure()

    with open('matrix_a.txt', 'r') as f:
        a_matrix = np.loadtxt(f)

    with open('matrix_b.txt', 'r') as f:
        b_matrix = np.loadtxt(f)

    print('Мaтриця A:')
    print(a_matrix)
    arrA = nx.Graph()
    resA = np.where(a_matrix != 0)
    oneArrayA = np.column_stack((resA[0], resA[1]))
    for i in oneArrayA:
        arrA.add_edge(i[0], i[1])

    ax1 = fig.add_subplot(121)
    ax1.set_title('Мaтриця A')
    nx.draw(arrA, with_labels=True, font_weight='bold', node_color='yellow')

    print('Мaтриця B:')
    print(b_matrix)
    arrB = nx.Graph()
    resB = np.where(b_matrix != 0)
    oneArrayB = np.column_stack((resB[0], resB[1]))
    for i in oneArrayB:
        arrB.add_edge(i[0], i[1])

    ax2 = fig.add_subplot(122)
    ax2.set_title('Мaтриця B')
    nx.draw(arrB, with_labels=True, font_weight='bold', node_color='orange')

    print('Результат вручну: Дані графи є',
          graph_isomporphism(a_matrix, b_matrix))
    if nx.is_isomorphic(arrA, arrB) == True:
        print('NetworkX: Дані графи є ізоморфічними')
    else:
        print('NetworkX: Дані графи є НЕ ізоморфічними')

    plt.show()
