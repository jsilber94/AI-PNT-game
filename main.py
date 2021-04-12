import sys
import random

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
    print(game)
    print(tokens_left_on_board)
    print(winner + " won and " + loser + " lost")
    print("````````````````")


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


if __name__ == '__main__':
    if len(sys.argv) > 1:
        line = []
        for i in range(3, len(sys.argv)):
            line.append(sys.argv[i])
        play_game(process_line(line))
    else:
        play_game([7, 0, [], 0])
        # play_game(["Player", 7, 3, [1, 2, 5], 3])

# if player_choices in possible_multiples:
#     return game[0] + " lost"
# possible_multiples = list(range(1, last_chosen, game[1]))
# possible_factors = []
