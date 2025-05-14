import cv2
import json
import os

# Input image and JSON file
IMAGE_PATH = "pictures/IMG_6620.jpeg"
LINES_JSON = "table_lines_IMG_6620.json"
OUTPUT_IMAGE = "visualized_nodes_IMG_6620.jpeg"

# Load image
img = cv2.imread(IMAGE_PATH)

# Load line positions
with open(LINES_JSON, "r") as f:
    lines_info = json.load(f)
vertical = lines_info["vertical"]
horizontal = lines_info["horizontal"]

# Draw and label each intersection
for x in vertical:
    for y in horizontal:
        # Draw a small circle at each node
        cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
        # Label the node with its coordinates
        label = f"({x},{y})"
        cv2.putText(
            img,
            label,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

# Save the output image
cv2.imwrite(OUTPUT_IMAGE, img)
print(f"Node visualization saved as {OUTPUT_IMAGE}")
