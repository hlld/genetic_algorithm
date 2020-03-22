"""
Brief: Genetic algorithm demo for eight queens game.
Author: Lang Hu
Student Number: S190301066
"""
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpt

def calc_conflict(location_0, location_1):
    if abs(location_0[0] - location_1[0]) != abs(location_0[1] - location_1[1]) \
            and location_0[0] != location_1[0] and location_0[1] != location_1[1]:
        return True
    return False

def calc_resilience(unity):
    length = len(unity)
    result = 0
    for ri in range(length):
        for ci in range(ri + 1, length):
            if calc_conflict([ri, unity[ri]], [ci, unity[ci]]):
                result = result + 1
    return result

def calc_best_unity(group_list):
    best_result = -1
    index = 0
    for k in range(len(group_list)):
        result = calc_resilience(group_list[k])
        if result > best_result:
            best_result = result
            index = k
    return group_list[index]

def random_group_list(num_unity=16, num_queen=8):
    group_list = [[0] * num_queen] * num_unity
    for k in range(num_unity):
        group_list[k] = list(np.random.choice(num_queen, size=num_queen, replace=False))
    return group_list

def calc_resiliences(group_list):
    num = len(group_list)
    result = [0] * num
    for k in range(num):
        result[k] = calc_resilience(group_list[k])
    return result

def normalize(resiliences):
    num = len(resiliences)
    v = sum(resiliences) + 1e-7
    result = [0.0] * num
    for k in range(num):
        result[k] = resiliences[k] / v
    return result

def calc_cumulate(probabilities):
    num = len(probabilities)
    result = [0.0] * num
    result[0] = probabilities[0]
    for k in range(1, num):
        result[k] = result[k - 1] + probabilities[k]
    return result

def random_match(cumulate):
    rand = random.random()
    index = 0
    for k in range(len(cumulate)):
        if rand < cumulate[k]:
            index = k
            break
    return index

def random_select(group_list):
    num = len(group_list)
    rows = len(group_list[0])
    result = [[0] * rows] * num
    resiliences = calc_resiliences(group_list)
    probabilities = normalize(resiliences)
    cumulate = calc_cumulate(probabilities)
    for k in range(num):
        index = random_match(cumulate)
        result[k] = group_list[index][:]
    return result

def compete_select(group_list):
    num = len(group_list)
    rows = len(group_list[0])
    result = [[0] * rows] * num
    resiliences = calc_resiliences(group_list)
    # top_k = np.array(resiliences).argsort()[::-1]
    for k in range(num):
        index = np.random.choice(num, size=2, replace=False)
        winner = index[0]
        if resiliences[index[0]] < resiliences[index[1]]:
            winner = index[1]
        result[k] = group_list[winner][:]
    return result

def do_with_probability(probability):
    if random.random() < probability:
        return True
    return False

def crossover(group_list, probability):
    num = len(group_list)
    sa = len(group_list[0])
    result = [[0] * sa] * num
    for k in range(num):
        result[k] = group_list[k][:]
    if do_with_probability(probability):
        index = np.random.choice(num, size=num, replace=False)
        for k in range(num // 2):
            parent_0 = group_list[index[k * 2 + 0]]
            parent_1 = group_list[index[k * 2 + 1]]
            child_0 = parent_0[:]
            child_1 = parent_1[:]
            rand = np.random.choice(sa, size=2, replace=False)
            if rand[0] > rand[1]:
                t = rand[0]
                rand[0] = rand[1]
                rand[1] = t
            sz = rand[1] - rand[0] + 1
            idx = np.random.choice(sz, size=sz, replace=False)
            for n in range(0, rand[1] - rand[0] + 1):
                child_0[n + rand[0]] = parent_0[idx[n] + rand[0]]
                child_1[n + rand[0]] = parent_1[idx[n] + rand[0]]
            result[k * 2 + 0] = child_0[:]
            result[k * 2 + 1] = child_1[:]
    return result

def mutate(group_list, probability):
    num = len(group_list)
    sa = len(group_list[0])
    result = [[0] * sa] * num
    for k in range(num):
        result[k] = group_list[k][:]
        if do_with_probability(probability):
            parent = group_list[k]
            child = parent[:]
            index = np.random.choice(sa, size=2, replace=False)
            child[index[0]] = parent[index[1]]
            child[index[1]] = parent[index[0]]
            result[k] = child[:]
    return result

def draw_circle(axes, row, col, color='r'):
    center = (0.5, 0.5)
    circle = mpt.Circle(center, radius=0.4, color=color, fill=True)
    axes[row, col].add_patch(circle)

def draw_chessboard(axes, rows=8, cols=8):
    for y in range(0, rows):
        for x in range(0, cols):
            # set the background of canvas.
            if y % 2 == 0 and x % 2 == 0:
                axes[y, x].set_facecolor('k')
            if y % 2 != 0 and x % 2 != 0:
                axes[y, x].set_facecolor('k')
            # close all the figure xes.
            axes[y, x].set_xticks([])
            axes[y, x].set_yticks([])
            # set all the edge labels.
            if x == 0:
                axes[y, x].set_ylabel(y + 1, rotation='horizontal', ha='right', va='center', fontsize=12)
            if y == 0:
                axes[y, x].set_title(x + 1, ha='center', va='center', fontsize=12)

def update_monitor(axes, unity, last_unity, color='r', rows=8, cols=8):
    # clear all the last locations.
    for k in range(rows):
        axes[k, last_unity[k]].cla()

    draw_chessboard(axes, rows, cols)
    # redraw all the new locations.
    for k in range(len(unity)):
        draw_circle(axes, row=k, col=unity[k], color=color)

def start_monitor(unities=16, p_crossover=0.5, p_mutate=0.05, iterations=100):
    rows = cols = 8
    fig, axes = plt.subplots(rows, cols, figsize=(5, 5))
    # set window title and figure title.
    fig.canvas.set_window_title('Genetic Algorithm')
    fig.suptitle('Eight Queens', fontsize=16)
    plt.ion()

    group_list = random_group_list(num_unity=unities, num_queen=rows)
    last_unity = [0] * rows
    for k in range(iterations):
        unity = calc_best_unity(group_list)
        resilience = calc_resilience(unity)
        print('unity: ' + str(unity))
        print('resilience: ' + str(resilience))
        if resilience == 28:
            color = 'b'
        else:
            color = 'r'
        update_monitor(axes, unity, last_unity, color, rows, cols)
        if resilience == 28:
            break
        else:
            group_list = compete_select(group_list)
            group_list = crossover(group_list, p_crossover)
            group_list = mutate(group_list, p_mutate)
        last_unity = unity[:]
        plt.pause(0.25)
    # stay until close the window manually.
    print('Monitor has done.')
    # stop and close the window.
    plt.pause(0)
    # plt.ioff()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments of genetic algorithm.')
    parser.add_argument('--unities', type=int, default=16, help='Number of group members.')
    parser.add_argument('--p_crossover', type=float, default=0.5, help='Probability of gene crossover.')
    parser.add_argument('--p_mutate', type=float, default=0.05, help='Probability of gene mutation.')
    parser.add_argument('--iterations', type=int, default=100, help='Iterations of algorithm.')
    args = parser.parse_args()
    if args.unities < 1 or args.unities > 1000:
        print('Bad argument --unities, too small or too large')
        exit(0)
    if args.p_crossover < 0.0 or args.p_crossover > 1.0:
        print('Bad argument --p_crossover, too small or too large')
        exit(0)
    if args.p_mutate < 0.0 or args.p_mutate > 1.0:
        print('Bad argument --p_mutate, too small or too large')
        exit(0)
    if args.iterations < 1 or args.iterations > 1000:
        print('Bad argument --iterations, too small or too large')
        exit(0)
    start_monitor(unities=args.unities, p_crossover=args.p_crossover,
                  p_mutate=args.p_mutate, iterations=args.iterations)
