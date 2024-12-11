import numpy as np
import os

def is_symbole_possible(grid_symbole, x, y, symbole):
	"""
	Check if placing a given symbol (1 = sun, 2 = moon) at position (x,y) is possible
	without violating the Tango puzzle constraints:
	- No more than 3 identical symbols in a row or column.
	- No line of 3 identical symbols in a row or column.
	"""
	old_symbole = grid_symbole[y, x]
	grid_symbole[y, x] = symbole

	# Check counts in the current row and column
	symbole_in_line = (grid_symbole[y] == symbole).sum()
	symbole_in_col = (grid_symbole[:, x] == symbole).sum()
	if symbole_in_line > 3 or symbole_in_col > 3:
		grid_symbole[y, x] = old_symbole
		return False

	# Check no 3 identical symbols consecutively in the row
	for i in range(min(0, x-2), max(6, x+2)):
		if (grid_symbole[y, i:i+3] == symbole).sum() == 3:
			grid_symbole[y, x] = old_symbole
			return False

	# Check no 3 identical symbols consecutively in the column
	for i in range(min(0, y-2), max(6, y+2)):
		if (grid_symbole[i:i+3, x] == symbole).sum() == 3:
			grid_symbole[y, x] = old_symbole
			return False

	# Restore original symbol and return True if no rule was violated
	grid_symbole[y, x] = old_symbole
	return True

def possible_symboles(grid, x, y):
	"""
	Return which symbols (1 or 2) can be placed at (x,y) under the Tango rules.
	"""
	possible = []
	for symbole in range(1, 3):
		if is_symbole_possible(grid, x, y, symbole):
			possible.append(symbole)
	return possible

def solve_tango(solutions, grid, x, y):
	"""
	Backtracking solver for the Tango puzzle:
	Tries to fill the entire 6x6 grid with valid symbols.
	Appends each valid complete solution to 'solutions'.
	"""
	if x == 6:
		# If we filled all columns in a row, move to the next column
		solutions.append(grid.copy())
		return
	if y == 6:
		# Move to the next column once we reach beyond the last row
		solve_tango(solutions, grid, x+1, 0)
		return

	# Try each possible symbol in the current cell
	for symbole in possible_symboles(grid, x, y):
		grid[y, x] = symbole
		solve_tango(solutions, grid, x, y+1)
		grid[y, x] = 0

def generate_all_tango_solutions():
	"""
	Generate all possible valid solutions for a 6x6 Tango grid
	without any additional constraints. The solutions are stored
	as a numpy array for later use.
	"""
	grid = np.zeros((6, 6), dtype=np.uint8)
	solutions = []
	solve_tango(solutions, grid, 0, 0)
	return np.array(solutions)

def is_board_valid(line):
	"""
	Validate a line (for the Queen puzzle variant):
	- All elements should be unique.
	- No two consecutive elements differ by 1.
	(This is a custom constraint from the provided code.)
	"""
	if len(set(line)) != len(line):
		return False
	for i in range(len(line) - 1):
		if abs(line[i] - line[i + 1]) == 1:
			return False
	return True

def solve_queens(avaible_rows, lines, line, i):
	"""
	Backtracking solver for the Queen puzzle variant:
	- 'avaible_rows' tracks which rows are free.
	- 'line' represents a possible queen placement configuration.
	- 'lines' accumulates all valid solutions.
	"""
	if i == len(line):
		# If we've placed all queens and the line passes validation, save it
		if is_board_valid(line):
			lines.append(line.copy())
		return

	# Try placing a queen in each available row of the current column
	for j in np.where(avaible_rows == 1)[0]:
		line[i] = j
		avaible_rows[j] = 0
		solve_queens(avaible_rows, lines, line.copy(), i + 1)
		avaible_rows[j] = 1

def generate_all_queen_solutions(n):
	"""
	Generate all valid solutions for the Queen puzzle variant for an n x n board.
	Solutions are stored in a numpy array.
	"""
	solutions = []
	line = np.array([0] * n)
	avaible_rows = np.ones(n)
	solve_queens(avaible_rows, solutions, line, 0)
	solutions = np.array(solutions)
	return solutions

def main():
	"""
	Main entry point:
	- If tango_solution.npy does not exist, generate all Tango solutions and save them.
	- For each n from 5 to 10, if queen_solution_n.npy does not exist, generate and save it.
	This script should be run once before using the puzzle solver scripts.
	"""
	if not os.path.exists("tango_solution.npy"):
		solutions = generate_all_tango_solutions()
		np.save("tango_solution.npy", solutions)

	for i in range(5, 11):
		if not os.path.exists(f"queen_solution_{i}.npy"):
			solutions = generate_all_queen_solutions(i)
			np.save(f"queen_solution_{i}.npy", solutions)

if __name__ == "__main__":
	main()
