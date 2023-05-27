# interview_questions_and_sols
Solutions to coding questions that are usually asked in interviews

# How to run the scripts
Simply call them with python3

# Matrix solution
Question: Given a square matrix area where each cell is either empty or has an obstacle. A #robot starts from the left top corner (0, 0) and its goal is the cell that is the diagonal opposite. The output of the algorithm is the shortest path the robot should take given that it can only move in the right and down directions unless there is a dead end.

## Running the solution
```
python3 matrix_solution.py
```
### Optional args
- `--matrix_size=<number>`: Given an int generates a random NxN matrix as input.
- `--test_mode`: The input is loaded from the `input.yaml` file.
