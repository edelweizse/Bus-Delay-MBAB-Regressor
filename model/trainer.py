import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

class Trainer:
    def prepare_data(self, dataset):
        dataset_size = len(dataset)
        train_size = int(0.7 * dataset_size)
        val_size = int(0.2 * dataset_size)
        test_size = dataset_size - train_size - val_size

        train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(
            train_data,
            32,
            True,
            num_workers = 3
        )

        val_loader = DataLoader(
            val_data,
            32,
            False,
            num_workers = 3
        )

        test_loader = DataLoader(
            test_data,
            32,
            False,
            num_workers = 3
        )

        return train_loader, val_loader, test_loader



    def train_regressor(self, model, train_loader, val_loader, epochs = 50):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        criterion = nn.MSELoss()
        l1_lambda = 0.01

        optimizer = optim.Adam(model.parameters(), lr = 0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 3, factor = 0.5)

        best_val_loss = float("inf")

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            train_mae = 0

            for features, delays in train_loader:
                features, delays = features.to(device), delays.to(device)

                optimizer.zero_grad()
                predictions = model(features)

                mse_loss = criterion(predictions, delays.unsqueeze(1))
                
                l1_reg = torch.tensor(0., device = device)
                for param in model.parameters():
                    l1_reg += torch.norm(param, 1)
                
                total_loss = mse_loss + l1_lambda * l1_reg
                total_loss.backward()

                optimizer.step()

                train_loss += mse_loss.item()
                train_mae += torch.mean(torch.abs(predictions - delays.unsqueeze(1))).item()

            avg_train_loss = train_loss / len(train_loader)
            avg_train_mae = train_mae / len(train_loader)

            model.eval()
            val_loss = 0
            val_mae = 0

            with torch.no_grad():
                for features, delays in val_loader:
                    features, delays = features.to(device), delays.to(device)
                    predictions = model(features)

                    val_loss += criterion(predictions, delays.unsqueeze(1)).item()
                    val_mae += torch.mean(torch.abs(predictions - delays.unsqueeze(1))).item()
            
            avg_val_loss = val_loss / len(val_loader)
            avg_val_mae = val_mae / len(val_loader)

            scheduler.step(avg_val_loss)

            print(f"Epoch {epoch + 1}/{epochs}:")
            print(f"Training - MSE: {avg_train_loss:.2f}, MAE: {avg_train_mae:.2f} minutes")
            print(f"Validation - MSE: {avg_val_loss:.2f}, MAE: {avg_val_mae:.2f} minutes")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "outputs/best_regressor.pt")

    def evaluate_predictions(self, model, test_loader):
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load("outputs/best_regressor.pt", weights_only = True))
        model.eval()

        predictions = []
        actuals = []

        with torch.no_grad():
            for features, delays in test_loader:
                features, delays = features.to(device), delays.to(device)
                pred = model(features)
                predictions.extend(pred.cpu().numpy())
                actuals.extend(delays.cpu().numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

        percentile_errors = np.percentile(np.abs(predictions - actuals), [25, 50, 75, 90, 95])

        return {
            "mae": mae,
            "rmse": rmse,
            "percentile_errors": {
                "p25": percentile_errors[0],
                "p50": percentile_errors[1],
                "p75": percentile_errors[2],
                "p90": percentile_errors[3],
                "p95": percentile_errors[4]
            }
        }

