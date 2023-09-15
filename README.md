# MaciezeKroprowe
Python_dot matrix

This code performs several tasks related to data processing and visualization using Python libraries like Pandas, NumPy, Matplotlib, and scikit-learn's DBSCAN. Here's a breakdown of what each part of the code does:

Imports necessary libraries:

sklearn.cluster.DBSCAN for DBSCAN clustering.
numpy for numerical operations.
matplotlib.pyplot for creating plots.
pandas for data manipulation.
Reads data from a CSV file called "rand.mums" into a Pandas DataFrame (df) with specific column names.

Splits the data into two tables, "> 1p_2s_rand_100nt" and "> 1p_2s_rand_100nt Reverse," and stores them in dictionaries (tables and tables_new).

Converts specific columns of the tables to integer data types.

Constructs matrices (matrix_data and matrix_data_reverse) based on the converted data, where each entry represents a point in space.

Converts these matrices into NumPy arrays for further processing.

Extracts data columns from the NumPy arrays (x, y, x_reverse, and y_reverse) for both tables.

Determines the maximum values of X and Y coordinates for both tables.

Initializes zero-filled matrices (matrix and matrix_reverse) with dimensions based on the maximum X and Y coordinates.

Populates the matrices with '1's at specific locations based on the data points.

Creates separate plots for each table using Matplotlib:

Uses ax.imshow to display the binary matrices as images.
Uses ax.scatter to overlay points on the images.
Sets labels, titles, and gridlines for the plots.
Saves the plots as image files (PNG format).

The data in the "rand.mums" file actually contains nucleotide amounts of spatial points for the array, this code allows you to visualize this data graphically. Each point in the matrix corresponds to a certain location in space, which can be analyzed in the context of examining the structure or properties of these nucleotides.
