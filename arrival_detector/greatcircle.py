import numpy as np

# approximate distance between two point with the haversine equation


def haversine(lon1: float, lat1: float,
              lon2: float, lat2: float,
              earth_radius: float = 6_371_000) -> float:
    """
    Calculate the great circle distance in meters between two points 
    on the earth 
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    distance = earth_radius * c
    return distance


# estimate the time it will take a tsunami wave to bridge that distance
def estimate_travel_time(lon1: float, lat1: float,
                         lon2: float, lat2: float,
                         earth_radius: float = 6_371_000,
                         depth: float = 4_000) -> float:
    """
    Estimates the time it will take a tsunami wave to travel
    between point a (lon1, lat1) to point b (lon2, lat2) in an
    ocean of constant depth depth
    """
    distance = haversine(lon1, lat1, lon2, lat2, earth_radius)
    velocity = np.sqrt(9.81*depth)
    time = distance / velocity
    return time
