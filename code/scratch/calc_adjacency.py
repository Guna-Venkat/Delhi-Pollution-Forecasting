import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# Use the coordinates we added to the dashboard
STATION_COORDS = {
    "Ashok_Vihar":      (28.6955, 77.1824),
    "Anand_Vihar":      (28.6476, 77.3025),
    "Bawana":           (28.7761, 77.0511),
    "Dwarka-Sector_8":  (28.5734, 77.0673),
    "Jahangirpuri":     (28.7344, 77.1691),
    "Mundka":           (28.6811, 77.0298),
    "Punjabi_Bagh":     (28.6675, 77.1325),
    "Rohini":           (28.7413, 77.1154),
    "Wazirpur":         (28.6997, 77.1654),
}

stations = list(STATION_COORDS.keys())
coords = np.array(list(STATION_COORDS.values()))

# Calculate distance matrix (approximate degrees to km: 1 deg ~ 111km)
dist_matrix = cdist(coords, coords) * 111

# Create Adjacency Matrix (Threshold = 10km)
# Two stations are "connected" if they are within 10km of each other
adj_matrix = (dist_matrix < 10).astype(int)

# Create a clean DataFrame for the report
adj_df = pd.DataFrame(adj_matrix, index=stations, columns=stations)

print("--- STATION ADJACENCY MATRIX (10km Threshold) ---")
print(adj_df)

# Save to CSV so user can use it
adj_df.to_csv("station_adjacency.csv")
print("\nSaved to station_adjacency.csv")
