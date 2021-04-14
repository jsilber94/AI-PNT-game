import copy
import random

import numpy as np


def play_game(game):
    tokens_left_on_board = np.arange(1, game[0] + 1, dtype=int).tolist()
    for num in game[2]:
        tokens_left_on_board.remove(num)

    print(game)
    print(tokens_left_on_board)
    print("````````````````")

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
        visited_nodes = build_a_tree(game_traversal)
        print(visited_nodes)


def make_game_traversal(depth, tokens_left_on_board, game, invalid_choices):
    game_traversal = [[copy.deepcopy(tokens_left_on_board), copy.deepcopy(game)]]
    while depth != 0:
        if game[1] == 0:  # MAX starts first
            make_first_move(tokens_left_on_board, game, invalid_choices)
        # elif game[1] % 2 == 0:  # MAX's turn
        else:
            res = take_a_turn(tokens_left_on_board, game)
            if not res:
                break

        depth -= 1
        game_traversal.append([copy.deepcopy(tokens_left_on_board), copy.deepcopy(game)])

    return game_traversal


def build_a_tree(game_traversal):
    parent_alpha = -np.inf  # more then or equal to
    parent_beta = +np.inf  # less than or equal to

    child_alpha = -np.inf
    child_beta = +np.inf
    visited_nodes = 0
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
            game_traversal = make_game_traversal(parent_index, parent[0], parent[1], moves_chosen)

            nodies = do_something(parent_alpha, parent_beta, game_traversal, visited_nodes)
        else:
            child_alpha, child_beta, nodies = produce_points_for_children(parent, amount_of_tokens_on_board,
                                                                          moves_chosen, visited_nodes)
            visited_nodes += nodies
            # update parent alpha/beta but switch them because its switching game turns
            if child_alpha is -np.inf:
                child_alpha = child_beta
                child_beta = np.inf
            elif child_beta is np.inf:
                child_beta = child_alpha
                child_alpha = -np.inf
            previous_parent_turned_child = parent
        visited_nodes += nodies
    return visited_nodes


def do_something(grandpa_alpha, grandpa_beta, game_traversal, visited_nodes):
    parent_alpha = -np.inf
    parent_beta = +np.inf

    # same thing as before but with grandpa
    for parent_index, parent in enumerate((reversed(game_traversal)), start=0):
        tokens_on_board = copy.deepcopy(parent[0])
        game = copy.deepcopy(parent[1])

        response = make_game_move(tokens_on_board, game)
        visited_nodes += 1
        if not response:
            return visited_nodes
        # get possibly updated value
        child_score = determine_node_score(tokens_on_board, game)
        # determine whos turn
        if game[1] % 2 == 0:
            parent_alpha = child_score
        else:
            parent_beta = child_score

        if grandpa_alpha != -np.inf:  # more than
            if parent_beta < grandpa_alpha:
                return visited_nodes
        else:
            if parent_alpha > grandpa_beta:
                return visited_nodes

    return visited_nodes


def make_first_move(tokens_left_on_board, game, invalid_choices):
    # round up with a .5
    limit = int(game[0] / 2 + .5)

    # generate random off number between 1 and limit (inclusively)
    while True:
        choice = random.randrange(1, limit, 2)
        if choice not in invalid_choices:
            break
    # choice = 3
    tokens_left_on_board.remove(choice)
    game[1] += 1
    game[2].append(choice)
    return True


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


def produce_points_for_children(parent, amount_of_tokens_on_board, moves_chosen, visited_nodes):
    child_alpha = -np.inf
    child_beta = +np.inf
    children = []

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
            break

        child = [tokens_on_board, game]
        if child not in children:
            # add previously removed choices back
            child[0].extend(moves_chosen)
            # add the most current choice
            moves_chosen.append(child[1][2][-1])
            # add to evaluated children
            children.append(child)
            amount_of_tokens_on_board -= 1
            visited_nodes += 1
            # get possibly updated value
            child_alpha, child_beta = evaluate_alpha_beta(tokens_on_board, game,
                                                          child_alpha, child_beta)

    return child_alpha, child_beta, visited_nodes


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

    if last_chosen == 1:
        if len(tokens_left_on_board) % 2 == 0:
            value = -0.5
        else:
            value = 0.5

    if is_prime(last_chosen):
        possible_multiples = list(range(1, last_chosen, game[1]))
        if len(possible_multiples) % 2 == 0:
            value = -0.7
        else:
            value = 0.7
    else:
        # "If the last move is a composite number (i.e., not prime), find the largest prime that can divide last move,
        # count the multiples of that prime, including the prime number itself if it hasn't already been taken,
        # in all the" possible successors. If the count is odd, return 0.6; otherwise, return-0.6."
        value = .6

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


if __name__ == '__main__':
    # if len(sys.argv) > 1:
    #     line = []
    #     for i in range(3, len(sys.argv)):
    #         line.append(sys.argv[i])
    #     play_game(process_line(line))
    # else:
    # play_game([7, 0, [], 2])
    # play_game([7, 3, [1, 4, 2], 3])
    play_game(read_test_cases()[2])
# if player_choices in possible_multiples:
#     return game[0] + " lost"
# possible_multiples = list(range(1, last_chosen, game[1]))
# possible_factors = []
