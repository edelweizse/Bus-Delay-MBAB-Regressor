import pandas as pd
import numpy as np
from processing.processor import DataProcessor
from processing.preprocessor import Preprocessor
from visualization.visualizer import Visualizer
from model.trainer import Trainer
from model.dataset import Dataset
from model.regressor import Regressor
from config import Config

def main():
    config = Config()
    processor = DataProcessor(config)
    visualizer = Visualizer()
    preprocessor = Preprocessor()
    trainer = Trainer()
    model = Regressor(input_size = 24)

    #processor.process()
    df = pd.read_csv("data/processed/features.csv")
    print("Data loaded")

    #visualizer.data_analysis()

    features_scaled = preprocessor.process(df)
    print("Data preprocessed")

    dataset = Dataset(features_scaled, df["stop_avg_delay_time"])
    print("Dataset created")

    train_loader, val_loader, test_loader = trainer.prepare_data(dataset)
    print("Loaders created")

    #trainer.train_regressor(model, train_loader, val_loader, epochs = 50)

    res = trainer.evaluate_predictions(model, test_loader)

    visualizer.prediction_analysis(res)
if __name__ == "__main__":
    main()