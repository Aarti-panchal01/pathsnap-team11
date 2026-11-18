"""Choose local metric projections for the geographically distributed trip data."""

from collections.abc import Sequence

import numpy as np
from pyproj import CRS, Transformer


def pick_metric_crs(longitudes: Sequence[float], latitudes: Sequence[float]) -> CRS:
    """Select a trip-local UTM zone; the full training set spans multiple continents."""
    lon = np.asarray(longitudes, dtype=float)
    lat = np.asarray(latitudes, dtype=float)
    valid = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lon) <= 180) & (np.abs(lat) <= 90)
    if not valid.any():
        raise ValueError("At least one finite, in-range longitude/latitude pair is required.")
    center_lon = float((lon[valid].min() + lon[valid].max()) / 2)
    center_lat = float((lat[valid].min() + lat[valid].max()) / 2)
    if not -80 <= center_lat <= 84:
        raise ValueError(f"UTM is not defined at this trip's center latitude: {center_lat}")
    zone = min(60, max(1, int((center_lon + 180) // 6) + 1))
    epsg = (32600 if center_lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def lonlat_to_metres(
    longitudes: Sequence[float], latitudes: Sequence[float], crs: CRS
) -> tuple[np.ndarray, np.ndarray]:
    """Project longitude/latitude arrays into the supplied metre-based CRS."""
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(longitudes, latitudes)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def metres_to_lonlat(
    x: Sequence[float], y: Sequence[float], crs: CRS
) -> tuple[np.ndarray, np.ndarray]:
    """Convert projected arrays back to longitude/latitude for maps and exports."""
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)


def trip_metric_coordinates(
    longitudes: Sequence[float], latitudes: Sequence[float]
) -> tuple[CRS, np.ndarray, np.ndarray]:
    """Apply one trip-local UTM CRS consistently to its coordinates."""
    crs = pick_metric_crs(longitudes, latitudes)
    x, y = lonlat_to_metres(longitudes, latitudes, crs)
    return crs, x, y
