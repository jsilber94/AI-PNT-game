import copy
import random

import numpy as np


def play_game(game):
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
        total_visited_nodes = build_a_tree(game_traversal)
        print(total_visited_nodes)


def make_game_traversal(depth, tokens_left_on_board, game, invalid_choices):
    game_traversal = [[copy.deepcopy(tokens_left_on_board), copy.deepcopy(game)]]

    if depth == -1:
        depth = 999

    while depth > 0:
        if game[1] == 0:  # MAX starts first
            res = make_first_move(tokens_left_on_board, game, invalid_choices)
            # elif game[1] % 2 == 0:  # MAX's turn
        else:
            tokens_left_on_board = [x for x in tokens_left_on_board if x not in invalid_choices]
            res = take_a_turn(tokens_left_on_board, game)
        if not res:
            break
        if depth != -1:
            depth -= 1
        game_traversal.append([copy.deepcopy(list(set(tokens_left_on_board + invalid_choices))), copy.deepcopy(game)])

    return game_traversal


def build_a_tree(game_traversal):
    parent_alpha = -np.inf  # more then or equal to
    parent_beta = +np.inf  # less than or equal to

    child_alpha = -np.inf
    child_beta = +np.inf
    total_visited_nodes = 0
    previous_parent_turned_child = None

    # iterate depth times
    for parent_index, parent in enumerate((reversed(game_traversal)), start=0):
        parent_alpha = child_alpha
        parent_beta = child_beta

        # keep track of previous parent node
        moves_chosen = []
        children = []
        amount_of_tokens_on_board = len(parent[0])

        if previous_parent_turned_child is not None:

            children.append(previous_parent_turned_child)  # we have a score so its children dont matter
            amount_of_tokens_on_board -= 1
            moves_chosen.append(previous_parent_turned_child[-1][2][-1])

            new_game_traversal = make_game_traversal(parent_index, copy.deepcopy(parent[0]),
                                                     copy.deepcopy(parent[1]),
                                                     moves_chosen)

            new_game_traversal.pop()
            # pass is as beta,alpha but accept as alpha,beta
            visited_nodes = produce_points_for_children_with_parent_value(parent_alpha, parent_beta,
                                                                          new_game_traversal,
                                                                          total_visited_nodes, moves_chosen)

        else:
            child_alpha, child_beta, visited_nodes = produce_points_for_children(parent, amount_of_tokens_on_board,
                                                                                 moves_chosen, total_visited_nodes)
            # update parent alpha/beta but switch them because its switching game turns
            if child_alpha is -np.inf:
                child_alpha = child_beta
                child_beta = np.inf
            elif child_beta is np.inf:
                child_beta = child_alpha
                child_alpha = -np.inf
            previous_parent_turned_child = parent
            # total_visited_nodes += visited_nodes
            print(parent)
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        previous_parent_turned_child = parent
    return total_visited_nodes


def produce_points_for_children_with_parent_value(grandpa_alpha, grandpa_beta, game_traversal, total_visited_nodes,
                                                  moves_chosen_new):
    parent_alpha = -np.inf
    parent_beta = +np.inf
    children = []
    last_chosen = 0

    for parent_index, parent in enumerate((reversed(game_traversal)), start=0):
        tokens_on_board = copy.deepcopy(parent[0])
        # add loop here for other children
        amount_of_tokens_on_board = len(parent[0])

        # if len(moves_chosen_new) != 0:
        #     tokens_on_board = [x for x in tokens_on_board if x not in moves_chosen_new]

        moves_chosen = []
        if last_chosen > 0:
            moves_chosen.append(last_chosen)

        while amount_of_tokens_on_board != 0:
            current_parent_game = copy.deepcopy(parent[1])
            # modify the current game with past move as not to repeat the number tokens - moves
            if len(moves_chosen) != 0:
                tokens_on_board = [x for x in tokens_on_board if x not in moves_chosen]

            if len(moves_chosen_new) != 0:
                tokens_on_board_new = [x for x in tokens_on_board if x not in moves_chosen_new]

            # check if starting places exist
            response = make_game_move(tokens_on_board_new, current_parent_game)

            if not response:
                break

            if current_parent_game not in children:
                # get children

                current_node_game_traversal = make_game_traversal(parent_index, copy.deepcopy(tokens_on_board_new),
                                                                  copy.deepcopy(current_parent_game),
                                                                  moves_chosen)
                current_node_game_traversal[0][0] = list(set(current_node_game_traversal[0][0] + moves_chosen))
                res = get_valued_children(current_node_game_traversal)
                print(current_parent_game)
                if res is None:
                    break
                # get possibly updated value
                child_score = determine_node_score(tokens_on_board, current_parent_game)
                # # determine whos turn
                if current_parent_game[1] % 2 == 0:
                    parent_alpha = child_score
                else:
                    parent_beta = child_score

                if grandpa_alpha != -np.inf:  # more than
                    if parent_beta < grandpa_alpha:
                        print("Prune rest")
                        break
                        # return total_visited_nodes
                else:
                    if parent_alpha > grandpa_beta:
                        print("Prune rest")
                        break
                        # return total_visited_nodes
                total_visited_nodes, amount_of_tokens_on_board = process_child_into_children(total_visited_nodes,
                                                                                             amount_of_tokens_on_board,
                                                                                             current_parent_game,
                                                                                             children, moves_chosen)
        if len(parent[1][2]) > 0:
            last_chosen = parent[1][2][-1]
        print(parent)
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    return total_visited_nodes


def get_valued_children(current_node_game_traversal):
    for game_index, game in enumerate((reversed(current_node_game_traversal)), start=0):

        if not make_game_move(copy.deepcopy(game[0]), copy.deepcopy(game[1])):
            print("No children rest")
            return None

        children = []
        choices = []
        amount_of_tokens_on_board = len(game[0])

        # cur_game[0] = [x for x in [1,2,3,4,5,6,7] if x not in cur_game[0]]

        while amount_of_tokens_on_board > 0:
            cur_game = copy.deepcopy(game)
            cur_game[0] = [x for x in cur_game[0] if x not in choices]
            # check if starting places exist
            response = make_game_move(cur_game[0], cur_game[1])

            if not response:
                return False
            if cur_game[1] not in children:
                choices.append(cur_game[1][2][-1])
                children.append(cur_game)
                amount_of_tokens_on_board -= 1
                print(cur_game)

    return True




def process_child_into_children(total_visited_nodes, amount_of_tokens_on_board, child, children, moves_chosen):
    # increment total visited nodes
    total_visited_nodes += 1
    # decrement the amount of tokens on board
    amount_of_tokens_on_board -= 1
    # add a copy of the child into children
    children.append(copy.deepcopy(child))
    # add the number that was chosen, its state is represented by child
    moves_chosen.append(children[-1][2][-1])

    return total_visited_nodes, amount_of_tokens_on_board


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
    player_choices = []
    last_chosen = game[2][-1]

    while True:
        choice = random.randrange(1, game[0] + 1)
        player_choices.append(choice)

        if (choice % last_chosen == 0 or last_chosen % choice == 0) and choice in tokens_left_on_board:
            break

        if choice == 1 and choice in tokens_left_on_board:
            break

        if set(tokens_left_on_board).issubset(list(set(player_choices))):
            return False

    # remove choice from board and update game
    tokens_left_on_board.remove(choice)
    game[1] += 1
    game[2].append(choice)
    return True


def produce_points_for_children(parent, amount_of_tokens_on_board, moves_chosen, total_visited_nodes):
    child_alpha = -np.inf
    child_beta = +np.inf
    children = []

    parent_alpha = -np.inf
    parent_beta = +np.inf

    # produce possible children
    while amount_of_tokens_on_board > 0:
        tokens_on_board = copy.deepcopy(parent[0])
        game = copy.deepcopy(parent[1])

        # modify the current game with past move as not to repeat the number
        if len(moves_chosen) != 0:
            tokens_on_board = [x for x in tokens_on_board if x not in moves_chosen]

        # make move
        response = make_game_move(tokens_on_board, game)
        if not response:
            print("Prune rest")
            if len(game[2]) % 2 == 0:
                return +1, +np.inf, total_visited_nodes
            else:
                return -np.inf, -1, total_visited_nodes

        child = [tokens_on_board, game]
        if child not in children:
            print(child[1])
            # add previously removed choices back
            child[0].extend(moves_chosen)

            # add the most current choice
            # moves_chosen.append(child[1][2][-1])
            # # add to evaluated children
            # children.append(child)
            # amount_of_tokens_on_board -= 1
            # total_visited_nodes += 1

            total_visited_nodes, amount_of_tokens_on_board = process_child_into_children(total_visited_nodes,
                                                                                         amount_of_tokens_on_board,
                                                                                         child[1], children,
                                                                                         moves_chosen)

            # get possibly updated value
            child_alpha, child_beta = evaluate_alpha_beta(tokens_on_board, game,
                                                          child_alpha, child_beta)

            if not make_game_move(copy.deepcopy(tokens_on_board), copy.deepcopy(game)):
                print("Prune rest")
                if len(game[2]) % 2 == 0:
                    return +1, +np.inf, total_visited_nodes
                else:
                    return -np.inf, -1, total_visited_nodes
    #             set alpha or beta to be +/-1
    return child_alpha, child_beta, total_visited_nodes


def evaluate_alpha_beta(tokens_on_board, game, child_alpha, child_beta):
    # are you a base node
    child_point = determine_node_score(tokens_on_board, game)

    # evaluate child_alpha and child_beta
    if game[1] % 2 == 0:  # even means child_alpha
        if child_alpha == -np.inf or child_alpha < child_point:
            child_alpha = child_point
            child_beta = np.inf
    else:  # odd means child_beta
        if child_beta == np.inf or child_beta > child_point:
            child_beta = child_point
            child_alpha = -np.inf
    return child_alpha, child_beta


def make_game_move(tokens_on_board, game):
    if game[1] == 0:  # MAX starts first
        return make_first_move(tokens_on_board, game, [])
    else:
        return take_a_turn(tokens_on_board, game)


def determine_node_score(tokens_left_on_board, game):
    player_mutator = 1 if len(game[2]) % 2 == 0 else -1
    last_chosen = game[2][-1]

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


def print_output(move: str, value: float, visited_nodes: int, evaluated_nodes: int, max_depth_reach: int,
                 avg_branching: float):
    print(f'Move: {move}')
    print(f'Value: {value:.1f}')
    print(f'Number of nodes visited: {str(visited_nodes)}')
    print(f'Number of nodes evaluated: {str(evaluated_nodes)}')
    print(f'Max depth reach: {str(max_depth_reach)}')
    print(f'Avg Effective Branching Factor: {avg_branching:.1f}')


if __name__ == '__main__':
    # if len(sys.argv) > 1:
    #     line = []
    #     for i in range(3, len(sys.argv)):
    #         line.append(sys.argv[i])
    #     play_game(process_line(line))
    # else:
    # while True:
    play_game([7, 0, [], 3])
    # play_game([7, 3, [1, 4, 2], 3])
    # play_game(read_test_cases()[2])
# if player_choices in possible_multiples:
#     return game[0] + " lost"
# possible_multiples = list(range(1, last_chosen, game[1]))
# possible_factors = []
