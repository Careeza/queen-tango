import cv2
import numpy as np
import mss
from collections import defaultdict
import threading
from pynput import keyboard
import pyautogui

have_highdpi = True    # Flag indicating if the screen uses high DPI scaling
double_click = False   # Flag to determine if double-click is needed to place a queen
stop_flag = False      # Flag to stop the main loop
capture_flag = False   # Flag to trigger puzzle capture and solving

def solution_valid_for_paving(paving, solution):
	"""
	Check if a given queen placement (solution) is valid for the given paving (color arrangement).
	Ensures that each color region receives exactly one queen.
	"""
	# For each placed queen, record the color index of that cell
	queens_color = [paving[queen_x, queen_y] for queen_x, queen_y in enumerate(solution)]
	# If there's a duplicate color, the solution is invalid
	if len(set(queens_color)) != len(queens_color):
		return False
	return True

def solve(paving):
	"""
	Attempt to solve the N-Queen puzzle for a given paving (color-coded grid).
	Loads a precomputed set of solutions from a .npy file and checks each
	against the paving to find a valid match.
	"""
	n = len(paving)
	try:
		# Load precomputed solutions for the given n
		solutions = np.load(f"queen_solution_{n}.npy")
	except FileNotFoundError:
		print(f"Solution file for {n}x{n} not found. Please run generate_solutions.py first")
		return None
	print(f"Solving {n}x{n}...")
	# Test each known solution until a valid one is found
	for solution in solutions:
		if solution_valid_for_paving(paving, solution):
			return solution
	print(f"No valid solution found for {n}x{n}")
	return None

def detect_grid(image):
	"""
	Detect the puzzle grid area in a given screenshot image.
	Returns the cropped grid image and its bounding rectangle if found.
	"""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	edges = cv2.Canny(blurred, 50, 150)

	contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Sort contours by area and assume the largest one is the puzzle grid
	contours = sorted(contours, key=cv2.contourArea, reverse=True)
	if len(contours) == 0:
		return None, (0, 0, 0, 0)

	grid_contour = contours[0]
	x, y, w, h = cv2.boundingRect(grid_contour)

	grid = image[y:y+h, x:x+w]
	return grid, (x, y, w, h)

def is_duplicate(square1, square2, tolerance=10):
	"""
	Check if two identified squares are essentially the same, within a given tolerance.
	Used to avoid counting the same cell multiple times.
	"""
	x1, y1, w1, h1 = square1
	x2, y2, w2, h2 = square2

	return (
		abs(x1 - x2) <= tolerance and
		abs(y1 - y2) <= tolerance and
		abs(w1 - w2) <= tolerance and
		abs(h1 - h2) <= tolerance
	)

def align_squares(squares, tolerance=10):
	"""
	Align detected squares so that cells line up properly on a grid.
	This helps correct slight offsets in detected cell coordinates.
	"""
	x_coords = [x for x, _, _, _ in squares]
	y_coords = [y for _, y, _, _ in squares]

	# Group similar coordinates and take their median to "snap" them to a grid
	unique_x = np.unique([np.median([x for x in x_coords if abs(x - ref_x) <= tolerance]) for ref_x in x_coords])
	unique_y = np.unique([np.median([y for y in y_coords if abs(y - ref_y) <= tolerance]) for ref_y in y_coords])

	aligned_squares = []
	for x, y, w, h in squares:
		# Snap coordinates to the nearest "unique" line
		closest_x = min(unique_x, key=lambda ref_x: abs(x - ref_x))
		closest_y = min(unique_y, key=lambda ref_y: abs(y - ref_y))
		aligned_squares.append((int(closest_x), int(closest_y), w, h))

	return aligned_squares

def extract_cells(grid):
	"""
	Extract individual cells from the puzzle grid image.
	Attempts to identify square cells by contour analysis and filtering.
	"""
	gray = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	_, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
	contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	squares = []
	for contour in contours:
		# Approximate the contour and check if it's a square
		epsilon = 0.05 * cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, epsilon, True)

		if len(approx) == 4 and cv2.contourArea(approx) > 50:
			x, y, w, h = cv2.boundingRect(approx)
			aspect_ratio = float(w) / h
			# Filter by approximate square shape
			if 0.9 <= aspect_ratio <= 1.1:
				squares.append((x, y, w, h))

	# Filter out squares that are too large or too small compared to median area
	if len(squares) > 0:
		areas = [w * h for _, _, w, h in squares]
		median_area = np.median(areas)
		area_tolerance = 0.2
		squares = [
			(x, y, w, h) for (x, y, w, h) in squares
			if abs(w * h - median_area) / median_area <= area_tolerance
		]

	# Remove duplicate or overlapping square detections
	filtered_squares = []
	for square in squares:
		if not any(is_duplicate(square, fsq) for fsq in filtered_squares):
			filtered_squares.append(square)
	squares = filtered_squares

	# Align squares to form a neat grid
	squares = align_squares(squares)
	return squares

def extract_colors(grid, squares):
	"""
	Given the identified cells (squares) and the full grid image, determine the color
	associated with each cell. Cells with similar colors are grouped together.
	"""
	color_map = defaultdict(list)
	colors_detected = []

	for square_idx, (x, y, w, h) in enumerate(squares):
		# Sample the center region of the cell to determine its average color
		margin = 0.4
		x_center_start = int(x + margin * w)
		x_center_end = int(x + (1 - margin) * w)
		y_center_start = int(y + margin * h)
		y_center_end = int(y + (1 - margin) * h)

		center_region = grid[y_center_start:y_center_end, x_center_start:x_center_end]
		avg_color = cv2.mean(center_region)[:3]

		# Find if this color matches a previously detected color region
		matched = False
		for idx, ref_color in enumerate(colors_detected):
			if np.linalg.norm(np.array(avg_color) - np.array(ref_color)) < 1:
				color_map[idx].append((square_idx, (x, y, w, h)))
				matched = True
				break
		# If no match, this is a new color group
		if not matched:
			colors_detected.append(avg_color)
			color_map[len(colors_detected) - 1].append((square_idx, (x, y, w, h)))

	return color_map, colors_detected

def extract_grid(grid_image):
	"""
	Extract the positions and color-coded layout of the N-Queen puzzle grid.
	Returns a list of cell coordinates and a 2D array representing the color layout.
	"""
	cells = extract_cells(grid_image)
	if len(cells) == 0:
		return None, None

	# Check if we have a square number of cells (e.g., 25 cells for 5x5)
	if len(cells) != int(np.sqrt(len(cells))) ** 2:
		return None, None

	n = int(np.sqrt(len(cells)))
	if n < 5:
		return None, None

	# Sort cells row-wise and determine the color mapping
	cells = sorted(cells, key=lambda sq: (sq[1], sq[0]))
	color_map, colors_detected = extract_colors(grid_image, cells)

	# We expect exactly n color groups for an n×n puzzle
	if len(color_map) != n:
		return None, None

	grid = np.zeros((n, n), dtype=int)
	# Assign each cell its color index
	for idx, squares_group in color_map.items():
		for square_idx, (x, y, w, h) in squares_group:
			grid[square_idx // n, square_idx % n] = idx

	return cells, grid

def click_mouse(x, y):
	"""
	Click at the given coordinates using PyAutoGUI.
	"""
	pyautogui.click(x, y, _pause=False, interval=0)

def solve_puzzle(offset_x, offset_y, cells, grid):
	"""
	Solve the puzzle and simulate mouse clicks to place the queens.
	Uses the detected cell coordinates and the computed solution.
	"""
	global have_highdpi
	global double_click

	solution = solve(grid)
	if solution is None:
		return

	# For each queen in the solution, find the corresponding cell and click on it
	for i in range(len(solution)):
		index = i * len(solution) + solution[i]
		x, y, w, h = cells[index]

		# Adjust for DPI scaling and place the queen
		if have_highdpi:
			click_mouse(offset_x + x // 2 + 10, offset_y + y // 2 + 10)
			if double_click:
				click_mouse(offset_x + x // 2 + 10, offset_y + y // 2 + 10)
		else:
			click_mouse(offset_x + x + 10, offset_y + y + 10)
			if double_click:
				click_mouse(offset_x + x + 10, offset_y + y + 10)

def on_press(key):
	"""
	Keyboard callback:
	- ESC key: stop and exit.
	- 'c' key: toggle capture_flag to start/stop puzzle solving attempts.
	"""
	global stop_flag
	global capture_flag
	try:
		if key == keyboard.Key.esc:
			stop_flag = True
			print("Exiting...")
			return False
		if key.char == 'c':
			capture_flag = not capture_flag
			print(f"Capture flag: {capture_flag}")
	except AttributeError:
		pass

def keyboard_listener():
	"""
	Launch a keyboard listener thread to handle key press events.
	"""
	with keyboard.Listener(on_press=on_press) as listener:
		listener.join()

def main():
	"""
	Main loop:
	- Starts keyboard listener in a separate thread.
	- Continuously captures the screen.
	- When capture_flag is set, attempts to detect and solve the puzzle.
	- Uses stop_flag to break the loop and exit.
	"""
	global stop_flag
	global capture_flag
	global have_highdpi

	listener_thread = threading.Thread(target=keyboard_listener)
	listener_thread.start()

	last_x, last_y = 0, 0

	with mss.mss() as sct:
		while not stop_flag:
			# Capture the entire screen
			screen = np.array(sct.grab(sct.monitors[0]))

			if capture_flag:
				# Attempt to detect the puzzle grid
				result = detect_grid(screen)
				if result is not None:
					grid_image, (x, y, w, h) = result
					# Extract cells and color layout
					cells, grid = extract_grid(grid_image)
					if grid is not None:
						# If the grid is stable (same position as last time), solve it
						if x == last_x and y == last_y:
							if have_highdpi:
								solve_puzzle(x//2, y//2, cells, grid)
							else:
								solve_puzzle(x, y, cells, grid)
							capture_flag = False
					last_x, last_y = x, y

			cv2.waitKey(1)

		cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
