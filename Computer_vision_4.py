import cv2
import numpy as np
import matplotlib.pyplot as plt

# BASIC IMAGE PROCESSING
lhu_img = cv2.imread('C:/Users/tancs/PycharmProjects/Computer vision/lhu.jpg')

# Resize using the function cv2.resize
# (0, 0) no explicit size given
# Using scaling factors fx and fy parameters
# The image will be half the size of the original
lhu_resize = cv2.resize(lhu_img, (0, 0), fx = 0.5, fy = 0.5)


# Displaying resized image
cv2.imshow('Original uni image', lhu_img)
cv2.imshow('Resized uni image', lhu_resize)

cv2.waitKey(0)
cv2.destroyAllWindows()

# EDGE DETECTION
# Read the image
cat_img = cv2.imread('C:/Users/tancs/PycharmProjects/Computer vision/kittyyy.jpg')

# Convert from BGR to RGB
# Matplotlib takes RGB
cat_rgb = cv2.cvtColor(cat_img, cv2.COLOR_BGR2RGB)

# Apply Canny edge detection
# --> pixels with intensity gradients within the thresholds are considered edges
edge_detection = cv2.Canny(image = cat_img, threshold1=50, threshold2=100 )

# Creating subplots
# 1 row and 2 columns of subplots
# figsize() sets the size of the figure, width and height
fig, axs = plt.subplots(1, 2, figsize=(7, 4))

# Plot original cat in the first subplot
axs[0].imshow(cat_rgb)
axs[0].set_title('Original cat')

# Plot the edges of the cat
# cmap = 'gray' for rendering
axs[1].imshow(edge_detection, cmap = 'gray')
axs[1].set_title('Edges cat')

# Remove the ticks from the subplots
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

# Display
plt.tight_layout()
plt.show()