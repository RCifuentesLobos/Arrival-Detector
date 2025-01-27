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


# approximate distance between two points on a sphere in degrees
def distance_in_degrees(lat1: float, lon1: float, 
                        lat2: float, lon2: float):
    """
    Computes the distance between two points on a sphere in degrees

    Parameters:
    lat1, lon1: Coordinate of the first point in degrees
    lat2, lon2: Coordinate of the second point in degrees

    Returns:
    Angular distance in degrees
    """
    # convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    # longitude difference 
    delta_lon = lon2_rad - lon1_rad
    # spherical cosine
    cos_c = np.sin(lat1_rad) * np.sin(lat2_rad) + np.cos(lat1_rad) * np.cos(lat2_rad) * np.cos(delta_lon)
    # clip to avoid numerical errors
    cos_c = np.clip(cos_c, -1.0, 1.0)
    # angular distance
    c = np.arccos(cos_c)
    # back to degrees
    degrees_between = np.degrees(c)

    return degrees_between

# Ejemplo de uso
if __name__ == "__main__":
    lat1, lon1 = 0, 0  # Punto 1: Ecuador y Greenwich
    lat2, lon2 = 10, 10  # Punto 2

    distancia = distancia_en_grados(lat1, lon1, lat2, lon2)
    print(f"La distancia angular entre los puntos es: {distancia:.6f} grados")


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
