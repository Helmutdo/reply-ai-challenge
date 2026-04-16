"""Geospatial utility functions for fraud detection."""
import math


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two GPS coordinates in km.

    Args:
        lat1, lon1: First point in decimal degrees.
        lat2, lon2: Second point in decimal degrees.

    Returns:
        Distance in kilometres.
    """
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
