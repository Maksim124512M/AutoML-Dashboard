# ML-Dashboard

Interactive Machine Learning Dashboard built with **Streamlit** and **scikit-learn**.
This project allows users to upload datasets, preprocess data, train multiple ML models, tune hyperparameters, visualize performance metrics, and export reports — all from an intuitive web interface.

## Features

- Upload your own CSV datasets for **classification** or **regression**
- Automatic **data preprocessing** (handling missing values, scaling)
- Train multiple ML models:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine
- Hyperparameter tuning via **GridSearchCV**
- Generate **performance metrics**: accuracy, F1, precision, recall
- Visualize results with **interactive plots**
- Export **reports** and **trained models** for later use
- Dockerized for **easy deployment**

## Technologies

- Python 3.12
- Pandas, NumPy, Matplotlib
- scikit-learn
- Streamlit
- Docker

## Getting Started

1. Clone the repo:
```bash
git clone https://github.com/your-username/ML-Dashboard.git
```
2. Navigate to the project directory:
```bash
cd ML-Dashboard
```
3. Build the Docker image:
```bash
docker build -t ml-dashboard .
```
4. Run the Docker container:
```bash
docker run -p 8501:8501 ml-dashboard
```
5. Open your browser and go to `http://localhost:8501` to access the dashboard.
## Usage
1. Upload your dataset in CSV format.
2. Select the target variable and task type (classification/regression).
3. Click "Train Models" to start training.
4. View performance metrics and visualizations.
5. Export reports and trained models as needed.
## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
