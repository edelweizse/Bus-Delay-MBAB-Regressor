import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from shapely.geometry import Point
import struct
import folium

class Visualizer:
    def process_data(self):
        bus_data = pd.read_csv("data/raw/bus.csv")
        bus_stops = pd.read_csv("data/raw/bus_stop.csv")

        def decode(hex):
            coord_hex = hex[18:]
            coord_bytes = bytes.fromhex(coord_hex)
            x, y = struct.unpack("<dd", coord_bytes)
            return Point(x, y)
        
        bus_data["time_at"] = pd.to_datetime(bus_data["time_at"])

        bus_stops["geometry"] = bus_stops["pt"].apply(lambda pt: decode(pt))
        bus_stops["lon"] = bus_stops["geometry"].apply(lambda x: x.x if x else None)
        bus_stops["lat"] = bus_stops["geometry"].apply(lambda x: x.y if x else None)

        bus_data["geometry"] = bus_data["pt"].apply(lambda pt: decode(pt))
        bus_data["lon"] = bus_data["geometry"].apply(lambda x: x.x if x else None)
        bus_data["lat"] = bus_data["geometry"].apply(lambda x: x.y if x else None)
        
        bus_data = bus_data.drop(columns = ["iteration", "pt", "geometry"])
        bus_stops = bus_stops.drop(columns = ["last_bus_stop_time_visit", "pt", "geometry"])

        return bus_data, bus_stops
    
    def explanatory_analysis(self):
        bus_data, _ = self.process_data()
        plt.figure(figsize=(15, 5))

        bus_data["hour"] = bus_data["time_at"].dt.hour
        hourly_counts = bus_data.groupby("hour").size()
        
        plt.subplot(1, 2, 1)
        hourly_counts.plot(kind = "bar")
        plt.title("Bus activity by hours")
        plt.xlabel("Hour")
        plt.ylabel("Records")

        plt.subplot(1, 2, 2)
        route_counts = bus_data["route_number"].value_counts()
        route_counts.head(10).plot(kind = "bar")
        plt.title("Top 10 routes")
        plt.xlabel("Route number")
        plt.ylabel("Records")

        plt.tight_layout()
        plt.savefig("outputs/data_analysis/bus_analysis.png")
        plt.close()

    def create_map(self):
        bus_data, bus_stops = self.process_data()
        center_lat = bus_stops['lat'].mean()
        center_lon = bus_stops['lon'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        for _, stop in bus_stops.iterrows():
            folium.CircleMarker(
                location=[stop['lat'], stop['lon']],
                radius=5,
                color='red',
                fill=True,
                popup=f'Bus Stop ID: {stop["id"]}'
            ).add_to(m)
        
        colors = plt.cm.Set3(np.linspace(0, 1, bus_data['route_number'].nunique()))
        for (route_num, bus_id), group in bus_data.groupby(['route_number', 'bus_id']).head(100).groupby(['route_number', 'bus_id']):
            color = colors[route_num % len(colors)]
            color_hex = '#%02x%02x%02x' % tuple(int(c * 255) for c in color[:3])
            
            route_coords = [[row['lat'], row['lon']] 
                        for _, row in group.sort_values('time_at').iterrows()]
            
            if len(route_coords) > 1:
                folium.PolyLine(
                    route_coords,
                    weight=2,
                    color=color_hex,
                    opacity=0.8,
                    popup=f'Route {route_num}, Bus {bus_id}'
                ).add_to(m)
        
        m.save("outputs/data_analysis/map.html")

    def features_heatmap(self):
        df = pd.read_csv("data/processed/features.csv")

        correlation_matrix = df.corr()

        plt.figure(figsize = (24, 16))
        sns.heatmap(correlation_matrix, annot = True, cmap = "coolwarm", fmt = ".2f", linewidths = 0.5)
        plt.title("Features correlation")
        plt.savefig("outputs/data_analysis/heatmap.png")
        plt.close()

    def data_analysis(self):
        self.explanatory_analysis()
        self.create_map()
        self.features_heatmap()

    def predictions_vs_actuals(self, results):
        predictions = results["predictions"].flatten()
        actuals = results["actuals"].flatten()
        
        plt.figure(figsize=(10, 6))
        plt.scatter(actuals, predictions, alpha=0.6, label="Predicted vs Actual")
        
        min_val = min(min(actuals), min(predictions))
        max_val = max(max(actuals), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Prediction Line")

        for actual, predicted in zip(actuals, predictions):
            intersection = (actual + predicted) / 2
            plt.plot([actual, intersection], [predicted, intersection], 'b-', alpha=0.2, linewidth = 0.5)
        
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Predictions vs Actuals")
        plt.legend()
        plt.grid()
        plt.savefig("outputs/prediction_analysis/pr_vs_ac.png")
        plt.close()

    def error_distribution(self, results):
        predictions = results["predictions"].flatten()
        actuals = results["actuals"].flatten()
        errors = predictions - actuals
        
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', label="Zero Error")
        
        plt.xlabel("Prediction Errors")
        plt.ylabel("Frequency")
        plt.title("Error Distribution")
        plt.legend()
        plt.grid()
        plt.savefig("outputs/prediction_analysis/er_dist.png")
        plt.close()

    def error_metrics(self, results):
        metrics = {
            "MAE": results["mae"],
            "RMSE": results["rmse"],
            "p25": results["percentile_errors"]["p25"],
            "p50": results["percentile_errors"]["p50"],
            "p75": results["percentile_errors"]["p75"],
            "p90": results["percentile_errors"]["p90"],
            "p95": results["percentile_errors"]["p95"]
        }
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(metrics.keys(), metrics.values(), color=['skyblue', 'salmon', 'lightgreen', 'gold', 'violet', 'cyan', 'pink'], edgecolor='black')
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.3f}", ha="center", va="bottom", fontsize=9)
        
        plt.ylabel("Error Value")
        plt.title("Error Metrics")
        plt.grid(axis='y')
        plt.savefig("outputs/prediction_analysis/metrics.png")
        plt.close()

    def prediction_analysis(self, results):
        self.predictions_vs_actuals(results)
        self.error_distribution(results)
        self.error_metrics(results)