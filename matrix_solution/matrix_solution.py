#!/usr/bin/python3
import numpy
import argparse
import copy
import yaml

# Set numpy print options
numpy.set_printoptions(threshold=numpy.inf, linewidth=numpy.inf)

# Process arguments
argparse.ArgumentParser()
parser = argparse.ArgumentParser(description='Process matrix size.')
parser.add_argument(
    '--matrix_size', type=int, help='Size of the matrix', default=5, required=False
)
parser.add_argument(
    '--test_mode', help='Test mode', action='store_true', default=False, required=False
)
parser.add_argument(
    '--free_move',
    help='Free movement of the robot in all direction',
    action='store_true',
    default=False,
    required=False,
)
args = parser.parse_args()


class MatrixSolution:
    # Class attribute
    mov_viz_char = '>'

    def __init__(self, matrix_size: int, test_mode: bool, free_move: bool) -> None:
        """Question: Given a matrix of size N*N, find the shortest path from top left to bottom right.
        Each cell of the matrix has a boolean value, True or False. Where True means that the cell is blocked.
        Only right and down jumps are allowed.
        """
        # Log the class description
        print(self.__init__.__doc__)

        # Initialize variables
        self.result = []
        self.free_move = free_move
        self.test_mode = test_mode
        # Character to visualize the movement of the robot

        if not test_mode:
            # Input matrix (8x8) with random True/False values
            self.np_matrix = numpy.random.choice(
                [True, False], size=(matrix_size, matrix_size), p=[0.2, 0.8]
            )
            # Convert numpy matrix to list of lists
            self.input_orig = self.np_matrix.tolist()
        else:
            # Log
            print('Test mode is on. Reading input from input.yaml file.')
            # Read input from input.yaml file
            with open('input.yaml') as file:
                input_dict = yaml.load(file, Loader=yaml.FullLoader)
            self.input_orig = input_dict['input']
            # Convert 0/1 to True/False
            self.input_orig = list(map(lambda x: list(map(bool, x)), self.input_orig))

        # Copy input matrix, as we will be modifying it
        self.input = copy.deepcopy(self.input_orig)
        self.matrix_size = len(self.input)

        # Generate a visualization matrix where we overlap input and result while in test_mode
        if test_mode:
            self.io_overlap = copy.deepcopy(input_dict['input'])

    def is_blocked(self, x, y) -> bool:
        """Check if the cell is blocked or not."""
        return self.input[x][y]

    def get_last_jump(self) -> str:
        """Get the last jump direction."""
        if len(self.result) <= 1:
            return None
        else:
            i2, j2 = self.result[-1]
            i1, j1 = self.result[-2]
            if i2 == i1:
                if j2 > j1:
                    return 'right'
                else:
                    return 'left'
            elif j2 == j1:
                if i2 > i1:
                    return 'down'
                else:
                    return 'up'
            else:
                print(f'Something is wrong in the entries of self.result.{self.result}')

    def simulate_right_down(
        self, last_jump: str, right: bool, down: bool, i: int, j: int
    ) -> str:
        # If last jump was down (first jump), prioritize right movement,
        # else if jump was right, prioritize down movement
        if last_jump is None or last_jump == 'down':
            if right is None or right is True:
                if down is None or down is True:
                    if (i, j) == (self.matrix_size - 1, self.matrix_size - 1):
                        return 'last_cell'
                    else:
                        return 'go_back'
                else:
                    return 'down'
            else:
                return 'right'
        elif last_jump == 'right':
            if down is None or down is True:
                if right is None or right is True:
                    if (i, j) == (self.matrix_size - 1, self.matrix_size - 1):
                        return 'last_cell'
                    else:
                        return 'go_back'
                else:
                    return 'right'
            else:
                return 'down'

    def simulate_left_up(
        self, last_jump: str, left: bool, up: bool, i: int, j: int
    ) -> str:
        # Since we cannot prioritize like the right_down simulation given the last jump direction,\
        # We could say that the best next move is to move in a direction other than the last jump,
        # Eg: If last_jump is right, then going left is basically the "go_back" move, so to have more
        # Possibility of reaching the goal, we can do go "up" and vice versa.
        if last_jump is None:
            return 'go_back'
        elif last_jump == 'right':
            # Prioritize up
            if up is None or up is True:
                if (i, j) == (self.matrix_size - 1, self.matrix_size - 1):
                    return 'last_cell'
                else:
                    return 'go_back'
            else:
                return 'up'
        elif last_jump == 'down':
            # Prioritize left
            if left is None or left is True:
                if (i, j) == (self.matrix_size - 1, self.matrix_size - 1):
                    return 'last_cell'
                else:
                    return 'go_back'
            else:
                return 'left'

    def find_best_next_cell(self, i, j) -> str:
        """Find the best next cell to jump to."""
        # Initialize direction variables
        left, up, right, down = None, None, None, None
        # Simulate jump down
        if i + 1 <= self.matrix_size - 1:
            down = self.input[i + 1][j]
        # Simulate jump right
        if j + 1 <= self.matrix_size - 1:
            right = self.input[i][j + 1]

        if self.free_move:
            # Simulate jump up
            if i - 1 >= 0:
                up = self.input[i - 1][j]
            # Simulate jump left
            if j - 1 >= 0:
                left = self.input[i][j - 1]

        # Get last jump direction
        last_jump = self.get_last_jump()

        # Check if free movement is allowed
        if self.free_move:
            # First priority is still right and down
            # But if status is "go_back" then we try to go up and left
            status = self.simulate_right_down(last_jump, right, down, i, j)
            if status == 'go_back':
                return self.simulate_left_up(last_jump, left, up, i, j)
            else:
                return status
        else:
            return self.simulate_right_down(last_jump, right, down, i, j)

    def print_result(self) -> None:
        """Print the self.result."""
        # Print input matrix
        print('##############################')
        print('Input Matrix')
        input_mod = list(map(lambda x: list(map(int, x)), self.input_orig))
        print(numpy.array(input_mod))
        # Print solution
        if self.result:
            print('##############################')
            print('Solution')
            pretty_result = numpy.full(
                (self.matrix_size, self.matrix_size), '0', dtype='U1'
            )
            for item in self.result:
                pretty_result[item[0]][item[1]] = MatrixSolution.mov_viz_char
            print(pretty_result)

    def recursion(self, i, j) -> None:
        """Recursion function."""
        # Stop recursion
        if i > self.matrix_size - 1 or j > self.matrix_size - 1:
            return None
        else:
            # Check if the current place is blocked or not
            is_blocked_curr_cell = self.is_blocked(i, j)

            # Edge case 1
            if is_blocked_curr_cell:
                # If first cell is blocked, then there are no solutions
                # First cell block,
                if (i, j) == (0, 0):
                    print('First cell is blocked. There are no viable solutions.')
                    self.result = None
                    return None
            else:
                # The cell is clear, it's a good cell
                self.result.append((i, j))
                # Visualize result for feedback if in test_mode
                if self.test_mode:
                    self.io_overlap[i][j] = MatrixSolution.mov_viz_char
                    print(f'\r{numpy.matrix(self.io_overlap)}', end='', flush=True)

                # Check best path
                best_path = self.find_best_next_cell(i, j)
                # print(best_path)
                if best_path == 'right':
                    # Jump right
                    self.recursion(i, j + 1)
                elif best_path == 'down':
                    # Jump down
                    self.recursion(i + 1, j)
                elif best_path == 'left':
                    # Jump left
                    self.recursion(i, j - 1)
                elif best_path == 'up':
                    # Jump up
                    self.recursion(i - 1, j)
                elif best_path == 'go_back':
                    # Come back one step
                    # If needs to go back from the first element, then there are no solutions
                    if (i, j) == (0, 0):
                        print(
                            'All paths to the destination are either blocked or cannot be achieved by only moving down and right.'
                        )
                        self.result = None
                        return None
                    self.result.pop()
                    # Reset io_overlap
                    self.io_overlap[i][j] = '0'
                    # Blacklist that cell
                    self.input[i][j] = True
                    # Restart from last known good location
                    i, j = self.result[-1]
                    self.result.pop()
                    # Reset io_overlap
                    self.io_overlap[i][j] = '0'
                    self.recursion(i, j)
                elif best_path == 'last_cell':
                    return None


# Run the recursion
matrix_solution = MatrixSolution(args.matrix_size, args.test_mode, args.free_move)
matrix_solution.recursion(0, 0)
matrix_solution.print_result()
