# File Name: plot_contours_and_radial_points.py
# Purpose: Given a all_contour_radial_points.txt file and n number of contour_line_XXXXXXXX.txt files, plots the radial points and their corresponding contours. 

import matplotlib.pyplot as plt
import csv
import os
import glob

def read_contour_data(filename):
    r_data = []
    z_data = []
    labels = []

    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)

        num_cols = len(header)
        if num_cols % 2 != 0:
            raise ValueError("Expected even number of columns for r,z pairs.")

        num_pairs = num_cols // 2
        r_data = [[] for _ in range(num_pairs)]
        z_data = [[] for _ in range(num_pairs)]
        labels = [header[2 * i] for i in range(num_pairs)]

        for row in reader:
            for i in range(num_pairs):
                try:
                    r_val = float(row[2 * i])
                    z_val = float(row[2 * i + 1])
                    if r_val == 0.0 and z_val == 0.0:
                        continue
                    r_data[i].append(r_val)
                    z_data[i].append(z_val)
                except (ValueError, IndexError):
                    continue
    return r_data, z_data, labels

def read_data(filename):
    x_vals = []
    y_vals = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    x, y = map(float, parts)
                    x_vals.append(x)
                    y_vals.append(y)
                except ValueError:
                    continue
    return x_vals, y_vals

def plot_all_contours(folder_path, base_contour_file, rmagx, zmagx):
    r_data, z_data, labels = read_contour_data(base_contour_file)

    # Find all contour files and sort them
    contour_files = glob.glob(os.path.join(folder_path, "contour_line_*.txt"))
    contour_files.sort()

    plt.figure(figsize=(4, 6))  
    num_pairs = len(r_data)
    colormap = plt.get_cmap('tab10')
    num_colors = colormap.N

    # Plot radial point sets and matching (contrasting) contour lines
    for i in range(num_pairs):
        if i >= len(contour_files):
            break  # No corresponding contour file

        # Skip empty radial sets
        if all(v == 0 for v in r_data[i]):
            print(f"Skipping {labels[i]} (all zero)")
            continue

        base_color = colormap(i % num_colors)
        contrast_color = colormap((i + num_colors // 2) % num_colors)

        # Plot radial points
        plt.plot(r_data[i], z_data[i], 'o', markersize=5, color=base_color)

        # Plot corresponding contour
        x_extra, y_extra = read_data(contour_files[i])
        if x_extra and y_extra:
            filename = os.path.basename(contour_files[i])
            contour_label = filename.replace("contour_line_", "").replace(".txt", "")
            plt.plot(x_extra, y_extra, 'o', markersize=1, linewidth=0.8, color=contrast_color, label=f'Psi: {contour_label}')

    # Magnetic Axis Point
    plt.plot(rmagx, zmagx, 'k*', markersize=8, label='Magnetic Axis')
    plt.xlim(4, 8.5)
    plt.ylim(-4, 5.5)
    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    plt.title("Contour Lines with Radial Points")
    plt.legend(loc='best', fontsize=8, markerscale=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

# User should update the folder path and magnetic axis coordinates as needed
folder = r"C:\Users\u\OneDrive\Desktop\OSPO_MFEM_Project\glvis-4.4\Contours_Plot"
rmagx = 6.20492; 
zmagx = 0.67054; 

base_file = os.path.join(folder, "all_contour_radial_points.txt")
plot_all_contours(folder, base_file, rmagx, zmagx)


