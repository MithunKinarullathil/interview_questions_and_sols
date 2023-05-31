# Matrix solution
Question: Given a square matrix area where each cell is either empty or has an obstacle. A #robot starts from the left top corner (0, 0) and its goal is the cell that is the diagonal opposite. The output of the algorithm is the shortest path the robot should take given that it can only move in the right and down directions unless there is a dead end.

## Running the solution
```
python3 matrix_solution.py
```
### Optional args
- `--matrix_size=<number>`: Given an int generates a random NxN matrix as input.
- `--file_input`: The input is loaded from the `input.yaml` file.
- `--free_nav`: Free movement of the robot in all directions.
- `--mov_viz_char`: Character to visualize the movement of the robot.
- `--visualize`: Visualize the jumps in realtime. Use it with the --speed arg to control iteration speed.
- `--speed`: [0-1] Speed of processing. 1 is fastest, 0 is slowest.
