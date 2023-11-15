import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

def FindCoefficients1(positions, flocking_radius, velocity_orientations):
    """
    Calculate the alignment and clustering coefficients for a configuration.

    Parameters:
        positions (numpy.ndarray): Array of shape (N, 2) representing the positions of N particles.
        flocking_radius (float): Flocking area radius.
        velocity_orientations (numpy.ndarray): Array of shape (N,) representing the orientation of velocity for each particle.

    Returns:
        tuple: Alignment coefficient and clustering coefficient.
    """
    N = len(positions)

    # Calculate Voronoi tessellation
    vor = Voronoi(positions)

    # Calculate alignment coefficient
    alignment_sum = np.sum(np.cos(velocity_orientations))
    alignment_coefficient = alignment_sum / N

    # Calculate clustering coefficient
    clustering_count = 0
    for i, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        if not -1 in region and len(region) > 2:  # Ensure at least 3 vertices for a valid polygon
            area = 0.5 * np.abs(np.dot(vor.vertices[region, 0], np.roll(vor.vertices[region, 1], 1)) - np.dot(vor.vertices[region, 1], np.roll(vor.vertices[region, 0], 1)))
            if area < np.pi * flocking_radius**2:
                clustering_count += 1

    clustering_coefficient = clustering_count / N

    return alignment_coefficient, clustering_coefficient
