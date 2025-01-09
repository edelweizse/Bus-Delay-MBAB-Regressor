import os

class Config():
    RAW_DATA = "data/raw"
    PROCESSED_DATA = "data/processed"

    BUS_DATA_CSV = os.path.join(RAW_DATA, "bus.csv")
    BUS_STOPS_CSV = os.path.join(RAW_DATA, "bus_stop.csv")
    BUS_STOP_DELAY_CSV = os.path.join(RAW_DATA, "bus_stop_delay_history.csv")

    PROCESSED_DATA_CSV = os.path.join(PROCESSED_DATA, "features.csv")