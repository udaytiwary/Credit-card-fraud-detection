# Credit-card-fraud-detection

## Task Objectives
This project aims to detect fraudulent credit card transactions using machine learning techniques. The dataset is highly imbalanced, so various techniques like SMOTE (Synthetic Minority Over-sampling Technique) and feature engineering are applied to improve model performance.

## Dataset
- The dataset used for this project is `creditcard.csv`.
- It contains transaction details, including `Time`, `Amount`, and anonymized features (V1-V28).
- The target variable `Class` indicates fraud (`1`) or non-fraud (`0`).

## Steps to Run the Project
1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```
2. **Install Dependencies**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
   - Open `CCFD_Project.ipynb` and execute the cells step by step.
   
4. **Run the Python Script (Alternative Approach)**
   ```bash
   python CCFD_Project.py
   ```
   This will execute the entire project workflow in a single script.

## Models Used
- **Logistic Regression**
- **Random Forest Classifier**
- **SMOTE for class balancing**
- **Feature Engineering (Transaction Frequency, Spending Patterns)**

## Evaluation Metrics
- **Accuracy**
- **Classification Report (Precision, Recall, F1-Score)**
- **Confusion Matrix**
- **ROC-AUC Score**

## Results & Observations
- The dataset is highly imbalanced, requiring oversampling techniques.
- Random Forest performed better in fraud detection than Logistic Regression.
- Feature engineering improved classification performance.

## Repository Structure
```
credit-card-fraud-detection/
│-- CCFD_Project.ipynb  # Jupyter Notebook version
│-- CCFD_Project.py     # Python script version
│-- README.md           # Project documentation
│-- creditcard.csv      # Dataset (not uploaded, needs to be downloaded separately)
```

## Notes
- Ensure you have `creditcard.csv` in the project directory before running the scripts.
- For any issues, please raise them in the repository.

## Contributors
- **Your Name** (Replace with your actual name)

## License
This project is for educational purposes only.

