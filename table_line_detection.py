import cv2
import numpy as np
import os
import json

# Input image path
IMAGE_PATH = "pictures/IMG_6620.jpeg"  # You can change this to any table image
# Output image with detected lines
OUTPUT_IMAGE = "detected_lines_IMG_6620.jpeg"
# Output file for line positions
OUTPUT_LINES = "table_lines_IMG_6620.json"

# Read the image
img = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Edge detection
edges = cv2.Canny(gray, 100, 150, apertureSize=3)
# Detect lines using Hough Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

verti_lst = []
horiz_lst = []
if lines is not None:
    for i in lines:
        x1, y1, x2, y2 = i[0]
        # Vertical line
        if abs(x1 - x2) < 10:
            verti_lst.append(x1)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Horizontal line
        if abs(y1 - y2) < 10:
            horiz_lst.append(y1)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Remove duplicates and sort
verti_lst = sorted(list(set(verti_lst)))
horiz_lst = sorted(list(set(horiz_lst)))

# Convert numpy int to Python int for JSON serialization
verti_lst = [int(x) for x in verti_lst]
horiz_lst = [int(y) for y in horiz_lst]

# Save the image with detected lines
cv2.imwrite(OUTPUT_IMAGE, img)

# Save the line positions to a JSON file
lines_info = {"vertical": verti_lst, "horizontal": horiz_lst}
with open(OUTPUT_LINES, "w") as f:
    json.dump(lines_info, f, indent=2)

# --- Node visualization ---
# Draw and label each intersection (node) on a copy of the image
img_nodes = cv2.imread(IMAGE_PATH)
for x in verti_lst:
    for y in horiz_lst:
        # Draw a small circle at each node
        cv2.circle(img_nodes, (x, y), 8, (0, 0, 255), -1)
        # Label the node with its coordinates
        label = f"({x},{y})"
        cv2.putText(
            img_nodes,
            label,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
# Draw green circles for unique x positions (vertical lines), skipping those too close to the previous
GREEN_THRESHOLD = 20  # Minimum distance between green points
last_x = None
for x in verti_lst:
    if last_x is None or abs(x - last_x) > GREEN_THRESHOLD:
        # Draw a green circle at the top of each unique vertical line
        cv2.circle(img_nodes, (x, horiz_lst[0]), 12, (0, 255, 0), 2)
        last_x = x
# Save the node visualization image
NODE_IMAGE = "visualized_nodes_IMG_6620.jpeg"
cv2.imwrite(NODE_IMAGE, img_nodes)

# Print detected distances for reference
print("Vertical line positions (x):", verti_lst)
print("Horizontal line positions (y):", horiz_lst)
print(f"Detected lines image saved as {OUTPUT_IMAGE}")
print(f"Line positions saved as {OUTPUT_LINES}")
print(f"Node visualization saved as {NODE_IMAGE}")
