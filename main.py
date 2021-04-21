import copy
import random

import numpy as np

eval_nodes_global_list = set()


def play_game(game):
    original_game = copy.deepcopy(game)
    tokens_left_on_board = np.arange(1, game[0] + 1, dtype=int).tolist()
    for num in game[2]:
        tokens_left_on_board.remove(num)

    winner = "J"
    loser = "D"

    if game[-1] == 0:
        game[-1] = -1
    depth = game[-1] - 1

    if winner == "MIN" or winner == "MAX":
        print(game)
        print(tokens_left_on_board)
        print(winner + " won and " + loser + " lost")
    else:
        game_traversal = make_game_traversal(depth, tokens_left_on_board, game, [])
        correct_move, alpha_score, total_visited_nodes, evaluated_nodes, max_depth_reached, avg_branching = \
            build_a_tree(game_traversal)
        print_output(correct_move[original_game[1]], alpha_score, total_visited_nodes, len(eval_nodes_global_list) + 1,
                     max_depth_reached, avg_branching)


def print_output(move: str, value: float, visited_nodes: int, evaluated_nodes: int, max_depth_reach: int,
                 avg_branching: float):
    print(f'Move: {move}')
    print(f'Value: {value:.1f}')
    print(f'Number of nodes visited: {str(visited_nodes)}')
    print(f'Number of nodes evaluated: {str(evaluated_nodes)}')
    print(f'Max depth reach: {str(max_depth_reach)}')
    print(f'Avg Effective Branching Factor: {avg_branching:.1f}')


def make_game_traversal(depth, tokens_left_on_board, game, invalid_choices):
    game_traversal = [[copy.deepcopy(tokens_left_on_board), copy.deepcopy(game)]]

    if depth == -1:
        depth = -2

    while depth > 0 or depth == -2:
        if game[1] == 0:  # MAX starts first
            res = make_first_move(tokens_left_on_board, game, invalid_choices)
        else:
            tokens_left_on_board = [x for x in tokens_left_on_board if x not in invalid_choices]
            res = take_a_turn(tokens_left_on_board, game)
        if not res:
            break
        if depth != -2:
            depth -= 1
        game_traversal.append(
            [copy.deepcopy(list(set(tokens_left_on_board + invalid_choices))), copy.deepcopy(game)])

    return game_traversal


def update_max_depth(game, max_depth):
    if len(game[2]) > max_depth:
        max_depth = len(game[2])
    return max_depth


def build_a_tree(game_traversal):
    parent_alpha = -np.inf  # more then or equal to
    parent_beta = +np.inf  # less than or equal to

    child_alpha = -np.inf
    child_beta = +np.inf
    total_evaluated_nodes = 0
    previous_parent_turned_child = None

    max_depth = 0
    visited = set()
    for item in game_traversal:
        visited.add(tuple(item[1][2]))
        max_depth = update_max_depth(item[1], max_depth)

    winning_path = game_traversal[-1][1][2]

    # iterate depth times
    for parent_index, parent in enumerate((reversed(game_traversal)), start=0):
        visited.add(tuple(parent[1][2]))

        if make_game_move(copy.deepcopy(parent[0]), copy.deepcopy(parent[1])):

            # keep track of previous parent node
            moves_chosen = []
            amount_of_tokens_on_board = len(parent[0])

            if previous_parent_turned_child is not None:

                if parent[1][1] % 2 == 0:
                    if parent_beta == 1:
                        # automatic loss, prune
                        previous_parent_turned_child = parent
                        if parent_alpha == -np.inf:
                            parent_alpha = parent_beta
                            parent_beta = np.inf
                        elif parent_beta == +np.inf:
                            parent_beta = parent_alpha
                            parent_alpha = -np.inf
                        continue
                else:
                    if parent_alpha == -1:
                        # automatic win, prune
                        previous_parent_turned_child = parent
                        if parent_alpha == -np.inf:
                            parent_alpha = parent_beta
                            parent_beta = np.inf
                        elif parent_beta == +np.inf:
                            parent_beta = parent_alpha
                            parent_alpha = -np.inf
                        continue

                amount_of_tokens_on_board -= 1
                moves_chosen.append(previous_parent_turned_child[-1][2][-1])

                new_game_traversal = make_game_traversal(parent_index - 1, copy.deepcopy(parent[0]),
                                                         copy.deepcopy(parent[1]),
                                                         moves_chosen)
                # if new_game_traversal[0] == parent and len(new_game_traversal) == 1:
                #     continue

                # pass is as beta,alpha but accept as alpha,beta
                visited, winning_path, total_evaluated_nodes, max_depth, parent_alpha, parent_beta = traverse_children(
                    parent_alpha,
                    parent_beta,
                    new_game_traversal,
                    visited,
                    moves_chosen,
                    total_evaluated_nodes,
                    max_depth,
                    winning_path)
                previous_parent_turned_child = parent
            else:

                parent_alpha, parent_beta, visited, winning_path, total_evaluated_nodes, max_depth = produce_points_for_children(
                    copy.deepcopy(parent), amount_of_tokens_on_board, total_evaluated_nodes, visited, max_depth)

                if parent_alpha == -np.inf:
                    parent_alpha = parent_beta
                    parent_beta = np.inf
                elif parent_beta == +np.inf:
                    parent_beta = parent_alpha
                    parent_alpha = -np.inf

                previous_parent_turned_child = parent
        else:
            if parent[1][1] % 2 == 0:
                parent_alpha = -1
            else:
                parent_beta = +1
            winning_path = parent[1][2]
            total_evaluated_nodes += 1

        print(parent)
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        previous_parent_turned_child = parent

    print("\n\n\n\n")
    print("Winning path: " + str(winning_path))

    if previous_parent_turned_child[1][1] % 2 == 0:
        value = parent_beta
    else:
        value = parent_alpha

    return winning_path if winning_path else None, value, len(
        visited), total_evaluated_nodes, max_depth - 1, calculate_branching_factor_from_visited_nodes(visited)


def traverse_children(parent_alpha, parent_beta, game_traversal, visited,
                      moves_chosen,
                      total_evaluated_nodes, max_depth, winning_path):
    if parent_alpha == -np.inf:
        grandpa_alpha = parent_beta
        grandpa_beta = np.inf
    elif parent_beta == +np.inf:
        grandpa_beta = parent_alpha
        grandpa_alpha = -np.inf

    children = []
    for parent_index, parent in enumerate((reversed(game_traversal)), start=0):

        amount_of_children = 0
        visited.add(tuple(parent[1][2]))

        amount_of_tokens_on_board = len(parent[0])

        while amount_of_tokens_on_board != 0:
            tokens_on_board = copy.deepcopy(parent[0])
            current_parent = copy.deepcopy(parent[1])

            if len(moves_chosen) != 0:
                tokens_on_board = [x for x in tokens_on_board if x not in moves_chosen]

            response = make_game_move(tokens_on_board, current_parent)

            visited.add(tuple(current_parent[2]))
            max_depth = update_max_depth(current_parent, max_depth)

            if current_parent not in children:
                print(current_parent)
                tokens_on_board = tokens_on_board + moves_chosen
                if not response:
                    break
                if not make_game_move(copy.deepcopy(tokens_on_board), copy.deepcopy(current_parent)):
                    print("prune the rest")

                    if current_parent[1] % 2 == 0:
                        parent_alpha = -1
                        parent_beta = +np.inf
                    else:
                        parent_beta = +1
                        parent_alpha = -np.inf

                    # 1,6,3 vs 1,5
                    winning_path = current_parent[2]
                    return visited, winning_path, total_evaluated_nodes, max_depth, parent_alpha, parent_beta
                else:

                    depth = max_depth - 1 - parent_index - 1
                    child_score, visited, total_evaluated_nodes = depth_children(depth, tokens_on_board, current_parent,
                                                                                 grandpa_alpha, grandpa_beta, visited,
                                                                                 total_evaluated_nodes, max_depth)
                moves_chosen.append(current_parent[2][-1])
                tokens_on_board += moves_chosen
                total_evaluated_nodes += 1
                children.append(current_parent)

                if current_parent[1] % 2 == 0:
                    parent_alpha = child_score
                else:
                    parent_beta = child_score

                if grandpa_alpha != -np.inf:  # more than
                    if parent_beta < grandpa_alpha:
                        print("Prune rest because grandparent")
                        print("parent_alpha: " + str(parent_alpha) + " parent_beta: " + str(parent_beta))
                        print("grandpa_alpha: " + str(grandpa_alpha) + " grandpa_beta: " + str(grandpa_beta))
                        winning_path = current_parent[2]
                        continue
                else:
                    if parent_alpha > grandpa_beta:
                        winning_path = current_parent[2]
                        print("Prune rest because grandparent")
                        print("parent_alpha: " + str(parent_alpha) + " parent_beta: " + str(parent_beta))
                        print("grandpa_alpha: " + str(grandpa_alpha) + " grandpa_beta: " + str(grandpa_beta))
                        continue

                    amount_of_children += 1

    return visited, winning_path, total_evaluated_nodes, max_depth, parent_alpha, parent_beta


def depth_children(depth, tokens_on_board_original, game_original, grandpa_alpha, grandpa_beta, visited,
                   total_evaluated_nodes, max_depth):
    winning_path = []
    parent_alpha = -np.inf
    parent_beta = +np.inf

    children = []
    choices = []
    while True:
        tokens_on_board = copy.deepcopy(tokens_on_board_original)
        game = copy.deepcopy(game_original)

        if len(choices) != 0:
            tokens_on_board = [x for x in tokens_on_board if x not in choices]

        res = make_game_move(tokens_on_board, game)
        if not res:
            print("no children left")
            if game[1] % 2 == 0:
                return parent_beta, visited, total_evaluated_nodes
            else:
                return parent_alpha, visited, total_evaluated_nodes

        depth -= 1
        child = [tokens_on_board, game]
        if child not in children:
            children.append(child)
            total_evaluated_nodes += 1
            visited.add(tuple(game[2]))
            max_depth = update_max_depth(game, max_depth)
            print(child)
            if depth > 0:
                child_score, visited, total_evaluated_nodes = depth_children(depth - 1, tokens_on_board, game,
                                                                             parent_beta, parent_beta, visited,
                                                                             total_evaluated_nodes, max_depth)
            # do stuff
            else:
                choices = choices + (game[[2][-1]])
                child_score = determine_node_score(tokens_on_board, game)
                parent_alpha, parent_beta, visited, winning_path, total_evaluated_nodes, max_depth = evaluate_node(
                    parent_alpha, parent_beta, tokens_on_board, game, winning_path, total_evaluated_nodes, visited,
                    max_depth)

                if parent_beta == 1 or parent_alpha == -1:
                    continue

            if game[1] % 2 == 0:
                parent_alpha = child_score
            else:
                parent_beta = child_score

            if parent_alpha == -np.inf:
                parent_alpha = parent_beta
                parent_beta = np.inf
            elif parent_beta == +np.inf:
                parent_beta = parent_alpha
                parent_alpha = -np.inf

            if grandpa_alpha != -np.inf:  # more than
                if parent_beta < grandpa_alpha:
                    print("Prune rest because grandparent")
                    print("parent_alpha: " + str(parent_alpha) + " parent_beta: " + str(parent_beta))
                    print("grandpa_alpha: " + str(grandpa_alpha) + " grandpa_beta: " + str(grandpa_beta))
                    return parent_beta, visited, total_evaluated_nodes
            else:
                if parent_alpha > grandpa_beta:
                    print("Prune rest because grandparent")
                    print("parent_alpha: " + str(parent_alpha) + " parent_beta: " + str(parent_beta))
                    print("grandpa_alpha: " + str(grandpa_alpha) + " grandpa_beta: " + str(grandpa_beta))
                    return parent_alpha, visited, total_evaluated_nodes


def process_child_into_children(amount_of_tokens_on_board, child, children, moves_chosen):
    # decrement the amount of tokens on board
    amount_of_tokens_on_board -= 1
    # add a copy of the child into children
    children.append(copy.deepcopy(child))
    # add the number that was chosen, its state is represented by child
    moves_chosen.append(children[-1][2][-1])

    return amount_of_tokens_on_board


def make_first_move(tokens_left_on_board, game, invalid_choices):
    # round up with a .5
    limit = int(game[0] / 2 + .5)

    choice = first_move_check(limit, invalid_choices, tokens_left_on_board)
    if choice == -1:
        return False
    tokens_left_on_board.remove(choice)
    game[1] += 1
    game[2].append(choice)
    return True


def first_move_check(limit, invalid_choices, tokens_left_on_board):
    attempts = []
    # generate random off number between 1 and limit (inclusively)
    while True:
        choice = random.randrange(1, limit, 2)
        attempts.append(choice)
        attempts = np.unique(attempts).tolist()

        if choice not in invalid_choices and choice in tokens_left_on_board:
            return choice

        if len(attempts) >= limit / 2:
            return -1
    # choice = 3


def take_a_turn(tokens_left_on_board, game):
    last_chosen = game[2][-1]
    choice = None

    for token in tokens_left_on_board:
        if (token % last_chosen == 0 or last_chosen % token == 0) and token in tokens_left_on_board:
            choice = token
            break

        if token == 1 and token in tokens_left_on_board:
            choice = token
            break

    # remove choice from board and update game
    if choice is None:
        return False

    tokens_left_on_board.remove(choice)
    game[1] += 1
    game[2].append(choice)
    return True


def produce_points_for_children(parent, amount_of_tokens_on_board,
                                total_evaluated_nodes, visited, max_depth):
    winning_path = []
    children = []

    parent_alpha = -np.inf
    parent_beta = +np.inf

    moves_chosen = []

    while amount_of_tokens_on_board > 0:
        tokens_on_board = copy.deepcopy(parent[0])
        game = copy.deepcopy(parent[1])

        # modify the current game with past move as not to repeat the number
        if len(moves_chosen) != 0:
            tokens_on_board = [x for x in tokens_on_board if x not in moves_chosen]

        # make move
        response = make_game_move(tokens_on_board, game)

        child = [tokens_on_board, game]

        if child not in children:

            print(child[1])
            if not response:
                return parent_alpha, parent_beta, visited, winning_path, total_evaluated_nodes, max_depth
            visited.add(tuple(child[1][2]))
            max_depth = update_max_depth(child[1], max_depth)

            # add previously removed choices back
            child[0].extend(moves_chosen)

            amount_of_tokens_on_board = process_child_into_children(amount_of_tokens_on_board,
                                                                    child[1], children,
                                                                    moves_chosen)

            parent_alpha, parent_beta, visited, winning_path, total_evaluated_nodes, max_depth = evaluate_node(
                parent_beta, parent_alpha, tokens_on_board, game, winning_path, total_evaluated_nodes, visited,
                max_depth)

            if parent_alpha == -1 or parent_beta == 1:
                # continue
                return parent_alpha, parent_beta, visited, winning_path, total_evaluated_nodes, max_depth

    return parent_alpha, parent_beta, visited, winning_path, total_evaluated_nodes, max_depth


def evaluate_node(child_alpha, child_beta, tokens_on_board, game, winning_path, total_evaluated_nodes, visited,
                  max_depth):
    child_alpha, child_beta, winning_path = evaluate_alpha_beta(tokens_on_board, game, child_alpha, child_beta,
                                                                winning_path)
    total_evaluated_nodes += 1

    if not make_game_move(copy.deepcopy(tokens_on_board), copy.deepcopy(game)):
        print("Prune rest")
        winning_path = game[2]
        if len(game[2]) % 2 == 0:
            return -1, +np.inf, visited, winning_path, total_evaluated_nodes, max_depth
        else:
            return -np.inf, +1, visited, winning_path, total_evaluated_nodes, max_depth

    return child_alpha, child_beta, visited, winning_path, total_evaluated_nodes, max_depth


def evaluate_alpha_beta(tokens_on_board, game, parent_alpha, parent_beta, winning_path):
    # are you a base node
    child_point = determine_node_score(tokens_on_board, game)

    # evaluate child_alpha and child_beta
    if game[1] % 2 == 0:  # even means child_alpha
        if parent_alpha == -np.inf or parent_alpha > child_point:
            parent_alpha = child_point
            parent_beta = np.inf
            winning_path = game[2]
    else:  # odd means child_beta
        if parent_beta == np.inf or parent_beta < child_point:
            parent_beta = child_point
            parent_alpha = -np.inf
            winning_path = game[2]

    return parent_alpha, parent_beta, winning_path


def make_game_move(tokens_on_board, game):
    if game[1] == 0:  # MAX starts first
        return make_first_move(tokens_on_board, game, [])
    else:
        return take_a_turn(tokens_on_board, game)


def determine_node_score(tokens_left_on_board, game):
    player_mutator = 1 if len(game[2]) % 2 == 0 else -1
    last_chosen = game[2][-1]

    global eval_nodes_global_list
    eval_nodes_global_list.add(tuple(game[2]))

    # if 1 in tokens_left_on_board:
    value = 0
    if 1 in tokens_left_on_board:
        value = 0
    elif last_chosen == 1:
        if len(tokens_left_on_board) % 2 == 0:
            value = -0.5
        else:
            value = 0.5

    elif is_prime(last_chosen):
        possible_multiples_count = 0
        for value in tokens_left_on_board:
            if value != last_chosen and value % last_chosen == 0:
                possible_multiples_count += 1

        if possible_multiples_count % 2 == 0:
            value = -0.7
        else:
            value = 0.7
    else:
        prime_factor = largest_prime_factor(last_chosen)
        possible_multiples_count = 0

        for value in tokens_left_on_board:
            if value != last_chosen and value % prime_factor == 0:
                possible_multiples_count += 1

        if possible_multiples_count % 2 == 0:
            value = -0.6
        else:
            value = 0.6

    return value * player_mutator


def run_tests():
    for case in read_test_cases():
        play_game(case)


def read_test_cases():
    cases = []
    with open("resources/testcase.txt") as f:
        lines = f.readlines()
        for cur_line in lines:
            if "input" not in cur_line:  # assume player name has input in it
                cases.append(process_line(cur_line.strip().split(" ")))
    return cases


def process_line(parts):
    taken_tokens = int(parts[2])
    list_of_taken_tokens = []
    for index in range(taken_tokens):
        list_of_taken_tokens.append(int(parts[3 + index]))

    return [int(parts[1]), taken_tokens, list_of_taken_tokens, int(parts[-1])]


# Source: https://www.reddit.com/r/Python/comments/5vwctk/what_is_the_fastest_python_isprime_methodfunction/
def is_prime(n):
    if n < 2:
        return False
    for x in range(2, int(n ** 0.5) + 1):
        if n % x == 0:
            return False
    return True


def calculate_branching_factor_from_visited_nodes(visited_nodes: {tuple}):
    list_of_relations = list()
    number_of_branch = 0

    if tuple() in visited_nodes:
        for item in visited_nodes:
            if len(item) == 1:
                number_of_branch += 1

        list_of_relations.append(number_of_branch)
        number_of_branch = 0

    for x in visited_nodes:
        for y in visited_nodes:
            if x and y and len(x) == len(y) - 1 and is_part_of(x, y):
                number_of_branch += 1
        list_of_relations.append(number_of_branch)
        number_of_branch = 0

    clean_list = list()

    for item in list_of_relations:
        if item != 0:
            clean_list.append(item)

    return sum(clean_list) / len(clean_list)


def is_part_of(a: tuple, b: tuple):
    part_of = True

    for x in range(len(a)):
        if a[x] != b[x]:
            part_of = False

    return part_of


def largest_prime_factor(n):
    prime_factor = 1
    i = 2

    while i <= n / i:
        if n % i == 0:
            prime_factor = i
            n /= i
        else:
            i += 1

    if prime_factor < n:
        prime_factor = n

    return prime_factor


if __name__ == '__main__':
    # if len(sys.argv) > 1:
    #     line = []
    #     for i in range(3, len(sys.argv)):
    #         line.append(sys.argv[i])
    #     play_game(process_line(line))
    # else:
    # while True:
    # play_game([7, 0, [], 2])
    # play_game([3, 0, [], 0])
    play_game([7, 1, [1], 2])
    # play_game([7, 2, [1, 4], 0])
    # play_game([10, 5, [3, 1, 8, 4, 2], 0])
    # play_game(read_test_cases()[2]) 3 0 [] 0
# if player_choices in possible_multiples:
#     return game[0] + " lost"
# possible_multiples = list(range(1, last_chosen, game[1]))
# possible_factors = []
