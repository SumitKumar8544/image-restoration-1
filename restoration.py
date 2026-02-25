# =====================================================
# Name: Sumit Kumar
# Roll No: 2301010297
# Course: Image Processing & Computer Vision
# Unit: Image Restoration
# Assignment: Noise Modeling and Image Restoration
# Date: 12-Feb-2026
# =====================================================

import cv2
import os
import numpy as np
import math


# -----------------------------------------------------
# Welcome Message
# -----------------------------------------------------
def welcome():
    print("=" * 60)
    print(" IMAGE RESTORATION FOR SURVEILLANCE SYSTEM ")
    print("=" * 60)
    print("Final Version: Noise + Filtering + Evaluation\n")


# -----------------------------------------------------
# Create Output Folder
# -----------------------------------------------------
def create_output_folder():
    if not os.path.exists("outputs"):
        os.makedirs("outputs")


# -----------------------------------------------------
# Load Image
# -----------------------------------------------------
def load_image(path):

    img = cv2.imread(path)

    if img is None:
        print("Error loading image:", path)
        return None

    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray


# -----------------------------------------------------
# Noise Models
# -----------------------------------------------------
def add_gaussian_noise(image):

    mean = 0
    std = 25
    gaussian = np.random.normal(mean, std, image.shape)
    noisy = image + gaussian

    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper(image):

    noisy = image.copy()
    prob = 0.02

    salt = np.random.rand(*image.shape) < prob
    noisy[salt] = 255

    pepper = np.random.rand(*image.shape) < prob
    noisy[pepper] = 0

    return noisy


# -----------------------------------------------------
# Filters
# -----------------------------------------------------
def mean_filter(image):
    return cv2.blur(image, (5, 5))


def median_filter(image):
    return cv2.medianBlur(image, 5)


def gaussian_filter(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


# -----------------------------------------------------
# Performance Metrics
# -----------------------------------------------------
def calculate_mse(original, restored):
    return np.mean((original - restored) ** 2)


def calculate_psnr(original, restored):

    mse = calculate_mse(original, restored)

    if mse == 0:
        return float("inf")

    return 10 * math.log10((255 ** 2) / mse)


# -----------------------------------------------------
# Analytical Comparison
# -----------------------------------------------------
def analyze_performance(results, noise_type):

    print(f"\n===== Performance Comparison ({noise_type}) =====")

    best_filter = None
    best_psnr = 0

    for name, metrics in results.items():

        mse_val, psnr_val = metrics
        print(f"{name} -> MSE: {mse_val:.2f}, PSNR: {psnr_val:.2f} dB")

        if psnr_val > best_psnr:
            best_psnr = psnr_val
            best_filter = name

    print(f"\nBest filter for {noise_type}: {best_filter}")
    print("===============================================")


# -----------------------------------------------------
# Main Function
# -----------------------------------------------------
def main():

    welcome()
    create_output_folder()

    files = os.listdir("images")

    if len(files) == 0:
        print("No images found!")
        return

    for file in files:

        print("\nProcessing:", file)

        path = os.path.join("images", file)

        original = load_image(path)

        if original is None:
            continue

        # -------- Add Noise --------
        noisy_g = add_gaussian_noise(original)
        noisy_sp = add_salt_pepper(original)

        # -------- Apply Filters --------
        mean_g = mean_filter(noisy_g)
        median_g = median_filter(noisy_g)
        gauss_g = gaussian_filter(noisy_g)

        mean_sp = mean_filter(noisy_sp)
        median_sp = median_filter(noisy_sp)
        gauss_sp = gaussian_filter(noisy_sp)

        # -------- Save Outputs --------
        name = file.split(".")[0]

        cv2.imwrite(f"outputs/{name}_original.png", original)
        cv2.imwrite(f"outputs/{name}_gaussian_noise.png", noisy_g)
        cv2.imwrite(f"outputs/{name}_salt_pepper.png", noisy_sp)

        cv2.imwrite(f"outputs/{name}_mean_g.png", mean_g)
        cv2.imwrite(f"outputs/{name}_median_g.png", median_g)
        cv2.imwrite(f"outputs/{name}_gauss_g.png", gauss_g)

        cv2.imwrite(f"outputs/{name}_mean_sp.png", mean_sp)
        cv2.imwrite(f"outputs/{name}_median_sp.png", median_sp)
        cv2.imwrite(f"outputs/{name}_gauss_sp.png", gauss_sp)

        # -------- Evaluation --------
        results_gaussian = {
            "Mean Filter": (
                calculate_mse(original, mean_g),
                calculate_psnr(original, mean_g)
            ),
            "Median Filter": (
                calculate_mse(original, median_g),
                calculate_psnr(original, median_g)
            ),
            "Gaussian Filter": (
                calculate_mse(original, gauss_g),
                calculate_psnr(original, gauss_g)
            )
        }

        results_sp = {
            "Mean Filter": (
                calculate_mse(original, mean_sp),
                calculate_psnr(original, mean_sp)
            ),
            "Median Filter": (
                calculate_mse(original, median_sp),
                calculate_psnr(original, median_sp)
            ),
            "Gaussian Filter": (
                calculate_mse(original, gauss_sp),
                calculate_psnr(original, gauss_sp)
            )
        }

        # -------- Analytical Discussion --------
        analyze_performance(results_gaussian, "Gaussian Noise")
        analyze_performance(results_sp, "Salt & Pepper Noise")

    print("\nAnalysis Complete Successfully!")


# -----------------------------------------------------
# Program Entry
# -----------------------------------------------------
if __name__ == "__main__":
    main()
