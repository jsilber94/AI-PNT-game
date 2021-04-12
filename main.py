import sys
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

    return [parts[0], int(parts[1]), taken_tokens, list_of_taken_tokens, int(parts[-1])]


def play_game(case):
    tokens_on_board = np.arange(1, case[1] + 1)
    print(case)
    print(tokens_on_board)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        line = []
        for i in range(2, len(sys.argv)):
            line.append(sys.argv[i])
        play_game(process_line(line))
    else:
        # play_game(["Player", 7, 0, [], 0])
        play_game(["Player", 7, 2, [3, 6], 0])
