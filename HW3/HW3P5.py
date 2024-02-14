# 24-677 Modern Control Theory
# HW3 Programming
# Ryan Wu
# ID: weihuanw

import cv2
import numpy as np

# Reading input image
input_image = cv2.imread("CMU_Grayscale.png", cv2.IMREAD_GRAYSCALE)

U, S, Vh = np.linalg.svd(input_image, full_matrices=False)

# Image compression function
def compress(singularValues, percentages):
    k = int(np.ceil(percentages * len(singularValues) / 100))
    print(f"Number of Singular Values used for {percentages}% compression: {k}")
    compressed_singularValues = singularValues[:k]
    compressed_image = np.dot(U[:, :k], np.dot(np.diag(compressed_singularValues), Vh[:k, :]))
    compressed_image = compressed_image.astype(np.uint8)

    return compressed_image

# Performing compressions at varying percentages
compressed_50 = compress(S, 50)
compressed_10 = compress(S, 10)
compressed_5 = compress(S, 5)

# Viewing and saving compressed images
cv2.imshow('Image compressed to 50%', compressed_50)
cv2.waitKey(0)
cv2.imwrite("compressed_50.png", compressed_50)
cv2.imshow('Image compressed to 10%', compressed_10)
cv2.waitKey(0)
cv2.imwrite("compressed_10.png", compressed_10)
cv2.imshow('Image compressed to 5%', compressed_5)
cv2.waitKey(0)
cv2.imwrite("compressed_5.png", compressed_5)
