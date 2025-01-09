# Overview
This project aims to predict bus delay using a Multi-Branch Attention-Based Regressor. This model uses temporal, spatial, historical and route-related features for prediction. By using attention mechanism, model dynamically focuses on relevant features.
## Development steps
1. Data collection
	Data was collected from https://github.com/Darwin939/bus_tracker
	Datasets that was used:
		`bus.csv` - contains time-based bus data
		`bus_stop.csv`- contains spatial data of bus stops
		`bus_stop_delay_history.csv` - contains historical records at bus stops
2. Data processing
	Processor module:
		- Formats timestamps and adds temporal features (cyclical features, is_rush_time etc.)
		- Decodes PostGIS into latitude and longitude
		- Statistical features (avg delay for stop, std and avg delay for buses and routes etc.)
		- In the end all features extracted and saved in `features.csv`
	Preprocessor module:
		- Handling missing values via replacing with mean values
		- Preparing data for model to use leveraging `StandardScaler` from Scikit-learn
3. Model design
	Multi-Branch architecture:
		- Temporal branch: encoding temporal features
		- Spatial branch: processes lon and lat
		- Historical branch: encodes statistical data
		- Route brach: encodes route and stops features
	Attention mechanism:
		- Highlights importance of features branches
		- Outputs weighted matrix for predictions
	Regressor: 
		- Fully connected layer with dynamic dropout for regularization
		- Outputs delay time in minutes
4. Training and evaluation
	Data splitting:
		- Training: 70%
		- Validation: 20%
		- Test: 10%
	Train process:
		- Loss function: MSE with l1 regularization
		- Optimizer: Adam with scheduled learning rate
		- Metrics: MAE, RMSE, Percentile error
	Best model saved based on validation loss
5. Visualization
	Data analysis:
		- Correlation heatmap for features
		- Interactive map of buses movement
		- Buses activity plot
	Prediction analysis:
		- Actual vs Predicted plot
		- Error distribution
		- Error metrics comparison

# Workflow
1. Data processing:
	- Input raw CSVs
	- Extract, clean and process features
	- Save for further analysis
2. Feature scaling
	Feature normalization using scikit
3. Dataset prep
	- PyTorch dataset and data loaders creation for training, validation and testing
4. Model training
	- Train the model
	- Monitor validation performance and save the best model
5. Evaluation
	- Test model
	- Calculate MAE, RMSE and percentile errors
6. Visuals
