import sys
import random
import copy
import numpy as np


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
    depth = game[-1]

    while depth != 0:
        if game[1] == 0:  # MAX starts first
            make_first_move(tokens_left_on_board, game)
        elif game[1] % 2 == 0:  # MAX's turn
            res = take_a_turn(tokens_left_on_board, game)
            if not res:
                winner = "MIN"
                loser = "MAX"
                break
        else:  # MIN's turn
            res = take_a_turn(tokens_left_on_board, game)
            if not res:
                winner = "MAX"
                loser = "MIN"
                break

        depth -= 1

    if winner == "MIN" or winner == "MAX":
        print(game)
        print(tokens_left_on_board)
        print(winner + " won and " + loser + " lost")
        return

    build_a_tree(tokens_left_on_board, game)
    x = 6
    # print("````````````````")


def make_first_move(tokens_left_on_board, game):
    # round up with a .5
    limit = int(game[0] / 2 + .5)
    # generate random off number between 1 and limit (inclusively)
    choice = random.randrange(1, limit, 2)
    tokens_left_on_board.remove(choice)
    game[1] += 1
    game[2].append(choice)


def take_a_turn(tokens_left_on_board, game):
    last_chosen = game[2][-1]

    player_choices = []
    # 3 6 2

    while True:
        choice = random.randrange(1, game[0] + 1)
        player_choices.append(choice)

        if (choice % last_chosen == 0 or last_chosen % choice == 0) and choice in tokens_left_on_board:
            break

        if choice == 1 and choice in tokens_left_on_board:
            break

        if set(tokens_left_on_board).issubset(list(set(player_choices))):
            return False

    tokens_left_on_board.remove(choice)
    game[1] += 1
    game[2].append(choice)

    return True


def build_a_tree(tokens_left_on_board, game):
    # careful!!!
    # opposite of the other mutator
    # a turn is made based on a copy of the original board
    # that switches the player therefore switch the mutator because out new board

    parent_node = 1
    # node_score = evaluate_board(tokens_left_on_board, game)
    children = {}


    while True:
        tokens_on_board_copy = copy.deepcopy(tokens_left_on_board)
        game_copy = copy.deepcopy(game)
        take_a_turn(tokens_on_board_copy, game_copy)

        point = evaluate_board(tokens_on_board_copy, game_copy)
        children[[tokens_on_board_copy, game_copy]] = point
        player_mutator = 1 if len(game_copy[2]) % 2 == 0 else -1

        if len(children) == 0:
            parent_node = point
        else:
            parent_node = point * player_mutator

def evaluate_board(tokens_left_on_board, game):
    player_mutator = 1 if len(game[2]) % 2 == 0 else -1

    last_chosen = game[2][-1]
    if 1 in tokens_left_on_board:
        value = 0

    if last_chosen == 1:
        if len(tokens_left_on_board[3]) % 2 == 0:
            value = -0.5
        else:
            value = 0.5

    if isPrime(last_chosen):
        possible_multiples = list(range(1, last_chosen, game[1]))
        if len(possible_multiples) % 2 == 0:
            value = -0.7
        else:
            value = 0.7
    else:
        # "If the last move is a composite number (i.e., not prime), find the largest prime that can divide last move, " \
        # "count the multiples of that prime, including the prime number itself if it hasn’t already been taken, in all the" \
        # " possible successors. If the count is odd, return 0.6; otherwise, return-0.6."
        value = .6

    return value * player_mutator


# Source: https://www.reddit.com/r/Python/comments/5vwctk/what_is_the_fastest_python_isprime_methodfunction/
def isPrime(n):
    if n < 2: return False
    for x in range(2, int(n ** 0.5) + 1):
        if n % x == 0:
            return False
    return True


if __name__ == '__main__':
    if len(sys.argv) > 1:
        line = []
        for i in range(3, len(sys.argv)):
            line.append(sys.argv[i])
        play_game(process_line(line))
    else:
        play_game([7, 0, [], 3])
        # play_game(["Player", 7, 3, [1, 2, 5], 3])

# if player_choices in possible_multiples:
#     return game[0] + " lost"
# possible_multiples = list(range(1, last_chosen, game[1]))
# possible_factors = []
