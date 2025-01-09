import pandas as pd
import numpy as np
import struct
from shapely.geometry import Point

class DataProcessor:
    def __init__(self, config):
        self.config = config

    def process(self):
        # Load data
        bus_data = pd.read_csv(self.config.BUS_DATA_CSV)
        stops_data = pd.read_csv(self.config.BUS_STOPS_CSV)
        delay_data = pd.read_csv(self.config.BUS_STOP_DELAY_CSV)

        # Convert timestamps
        bus_data["time_at"] = pd.to_datetime(bus_data["time_at"])
        delay_data["datetime"] = pd.to_datetime(delay_data["time"])

        # Decode postgis
        def decode(hex):
            coord_hex = hex[18:]
            coord_bytes = bytes.fromhex(coord_hex)
            x, y = struct.unpack("<dd", coord_bytes)
            return Point(x, y)
        
        stops_data["geometry"] = stops_data["pt"].apply(lambda pt: decode(pt))
        stops_data["lon"] = stops_data["geometry"].apply(lambda x: x.x if x else None)
        stops_data["lat"] = stops_data["geometry"].apply(lambda x: x.y if x else None)

        bus_data["geometry"] = bus_data["pt"].apply(lambda pt: decode(pt))
        bus_data["lon"] = bus_data["geometry"].apply(lambda x: x.x if x else None)
        bus_data["lat"] = bus_data["geometry"].apply(lambda x: x.y if x else None)

        # Clean redundant columns
        bus_data = bus_data.drop(columns = ["iteration", "pt", "geometry"])
        stops_data = stops_data.drop(columns = ["last_bus_stop_time_visit", "pt", "geometry"])
        delay_data = delay_data.drop(columns = ["delay"])

        # Add temporal features
        delay_data["hour"] = delay_data["datetime"].dt.hour
        delay_data["minute"] = delay_data["datetime"].dt.minute
        delay_data["time_of_day"] = delay_data["hour"] + delay_data["minute"]/60

        ## Add cyclical time features
        delay_data["hour_sin"] = np.sin(2 * np.pi * delay_data["hour"] / 24)
        delay_data["hour_cos"] = np.cos(2 * np.pi * delay_data["hour"] / 24)

        delay_data["minute_sin"] = np.sin(2 * np.pi * delay_data["minute"] / 60)
        delay_data["minute_cos"] = np.cos(2 * np.pi * delay_data["minute"] / 60)

        delay_data["time_of_day_sin"] = np.sin(2 * np.pi * delay_data["time_of_day"] / 24)
        delay_data["time_of_day_cos"] = np.cos(2 * np.pi * delay_data["time_of_day"] / 24)

        delay_data["is_rush_hour"] = ((delay_data["hour"] >= 7) & (delay_data["hour"] <= 10) | 
                                      (delay_data["hour"] >= 16) & (delay_data["hour"] <= 19))
        
        # Stop features
        stop_features = delay_data.groupby("bus_stop_id").agg({
            "id": "count",
            "hour": "mean",
            "time_of_day": ["mean", "std"]
        }).reset_index()

        stop_features.columns = ["bus_stop_id", "stop_delay_count", "stop_avg_delay_hour", "stop_avg_delay_time", "stop_std_delay_time"]

        ## Calculate if stop is frequent
        mean_delays = stop_features["stop_delay_count"].mean()
        stop_features["is_frequent_delay_stop"] = stop_features["stop_delay_count"] > mean_delays

        # Bus features
        bus_features = delay_data.groupby("bus_id").agg({
            "id": "count",
            "bus_stop_id": pd.Series.nunique,
            "time_of_day": "mean"
        }).reset_index()

        bus_features.columns = ["bus_id", "bus_total_delays", "bus_unique_stops_delayed", "bus_avg_delay_time"]

        # Route features
        route_features = bus_data.groupby("route_number").agg({
            "id": "count",
            "latest_bus_stop_id": pd.Series.nunique
        }).reset_index()

        route_features.columns = ["route_number", "route_total_buses", "route_unique_stops"]

        # Merge features
        features = delay_data.merge(stops_data[['id', 'lon', 'lat']],
                                    left_on='bus_stop_id',
                                    right_on='id',
                                    suffixes=('', '_stop'))
        features = features.merge(bus_features, on='bus_id', how='left')
        features = features.merge(stop_features, on='bus_stop_id', how='left')
        features = features.merge(bus_data[['id', 'route_number']],
                                  left_on='bus_id',
                                  right_on='id',
                                  suffixes=('', '_bus'))
        features = features.merge(route_features, on='route_number', how='left')

        # Conclusion
        features_df = features[[
            # Temporal
            "hour", "minute", "time_of_day", "is_rush_hour", 
            "hour_sin", "hour_cos", 
            "minute_sin", "minute_cos", 
            "time_of_day_sin", "time_of_day_cos",

            # Stop features
            "lon", "lat", "stop_delay_count", "stop_avg_delay_hour",
            "stop_avg_delay_time", "stop_std_delay_time",
            "is_frequent_delay_stop", "bus_stop_id",

            # Bus features
            "bus_id", "bus_total_delays", "bus_unique_stops_delayed", "bus_avg_delay_time",

            # Route features
            "route_number", "route_total_buses", "route_unique_stops"
        ]]

        # Save to data/processed/features.csv
        features_df.to_csv(self.config.PROCESSED_DATA_CSV, index = False)

        print(f"Extracted {features_df.shape[1]} features for {features_df.shape[0]} instances")

        
        



        

        