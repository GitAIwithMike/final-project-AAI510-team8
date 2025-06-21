# MS-AAI
## AAI-510-02 Final Project - Group 8

### Final Project for the AAI-501 (Machine Learning: Fundamentals and Applications) Course in the **Master of Applied Artificial Intelligence** Program
### University of San Diego

## ���� Stock Price Prediction for S&P 500 Companies  
## Iman Hamdan, Michael De Leon, Matt Hashemi  

> **Note:** The machine learning models and data analysis are located in [`Final_Project_Section2_Team8.ipynb`](Final_Project_Section2_Team8.ipynb)

---

## �� Introduction

Accurate stock price prediction is essential for investors aiming to reduce risks, maximize returns, and make informed decisions in volatile markets.

This project leverages historical financial data and machine learning techniques to develop robust predictive models for **505 companies listed on the S&P 500 index**. By analyzing key metrics such as stock prices, earnings, and market indicators, we aim to build a model that delivers high accuracy and actionable insights.

---

## �� Project Goals

- **Predictive Modeling**: Develop ML models to forecast future stock prices.
- **Model Comparison**: Evaluate Random Forest, XGBoost, Decision Tree, and Linear Regression.
- **Feature Engineering**: Improve performance using moving averages and volatility metrics.
- **Data Visualization**: Provide visual insights into feature importance and residuals.
- **Deployment Planning**: Outline a strategy for deploying the model using AWS SageMaker.

---

## �� Technologies Used

| Technology | Purpose |
|----------|---------|
| Python (3.11.7) | Core programming language |
| Scikit-learn, XGBoost | Machine learning models |
| Pandas, NumPy | Data manipulation and preprocessing |
| Matplotlib, Seaborn | Data visualization |
| Jupyter Notebook | Interactive analysis |
| AWS SageMaker | Model deployment |

---

## �� Why This Matters?

Stock price movements are influenced by numerous factors including macroeconomic trends, company-specific events, and global news. Traditional methods often fall short in capturing complex patterns in financial data.

Using **machine learning**, our Random Forest model achieved an impressive **R² score of 0.98** on validation data, demonstrating its potential to be integrated into real-time investment decision-making systems.

---

## �� Dataset Overview

We used a dataset containing financial data for **505 S&P 500 companies**, including **14 key features**:

### �� Features Included:
| Feature | Description |
|--------|-------------|
| `Price` | Current stock price |
| `Price/Earnings` | P/E ratio |
| `Dividend Yield` | Dividend per share divided by price per share |
| `Earnings/Share` | Earnings per share |
| `52 Week Low/High` | Annual price range |
| `Market Cap` | Market capitalization |
| `EBITDA` | Earnings before interest, taxes, depreciation, and amortization |
| `Price/Sales`, `Price/Book` | Valuation ratios |
| `Sector` | Industry classification (encoded) |

🔗 **Dataset Source:** [S&P 500 Financial Data](https://www.kaggle.com/datasets/)  *(link placeholder – replace with actual link)*

---

## �� Machine Learning Models

We trained and evaluated four models:

1. **Linear Regression**
   - Baseline model for comparison
   - Interpretable coefficients and fast training

2. **Decision Tree**
   - Captures non-linear relationships
   - Prone to overfitting without pruning

3. **Random Forest**
   - Ensemble method combining multiple trees
   - Reduced overfitting and high accuracy (Best performing model)

4. **XGBoost**
   - Gradient boosting framework
   - Strong regularization but underperformed on this dataset

### �� Model Performance Summary

| Model | RMSE | MAE | R² |
|------|------|-----|-----|
| **Random Forest** | 0.01 | 0.01 | **0.98** ✅ |
| **Decision Tree** | 0.02 | 0.01 | 0.93 |
| **Linear Regression** | 0.02 | 0.01 | 0.84 |
| **XGBoost** | 0.07 | 0.01 | -0.30 ❌ |

---

## �� Visualizations

Key visualizations included in the notebook:

- **Correlation Matrix Heatmap**: Identifies important features related to stock prices
- **Actual vs. Predicted Scatter Plots**: Compares model predictions against true values
- **Residual Analysis**: Assesses model error distribution
- **Learning Curves**: Shows how model performance improves with more data
- **Feature Importance Plot**: Highlights which variables most influence predictions

---

## �� Deployment Plan

To ensure practical use of our model, we propose the following deployment plan:

- **Hosting Platform**: AWS SageMaker
- **Inference Type**: Batch processing (daily predictions)
- **Latency Requirements**: Low-latency response via REST API
- **Cost Optimization**: Auto-scaling based on demand
- **Monitoring**: Continuous evaluation and retraining on new data

---

## ��️ Ethical Considerations

- **Bias & Fairness**: Ensure the model does not disproportionately favor certain sectors or large-cap companies.
- **Transparency**: Clearly document assumptions, limitations, and interpretability of results.
- **Regulatory Compliance**: Align with SEC guidelines and data privacy laws when used in production environments.

---

## �� Acknowledgments

- **Dataset Source**: [S&P 500 Companies with Financial Information](https://www.kaggle.com/datasets/) 
- **References**:
  - *"Machine Learning for Finance"* by Marcos Lopez de Prado
  - *"Deep Learning for Time Series Forecasting"* by Jason Brownlee

---

## Project Setup Instructions

This project includes Python scripts and Jupyter notebooks for analyzing statistical data. Follow the instructions below to set up the environment and run the Jupyter Notebook.

---

## Prerequisites

Ensure you have the following installed on your system:
- Python 3.11.7
- `pip` (Python's package installer)

---

## Setup Instructions

1. **Clone or Download the Project Repository**  
   ```bash
   git clone https://github.com/yourusername/stock-price-prediction.git 
   cd stock-price-prediction
   ```

2. **Create a Virtual Environment (Optional but Recommended)**  
   Create a virtual environment to manage dependencies:
   ```bash
   python3.11 -m venv env
   ```

   Activate the environment:
   - macOS/Linux:
     ```bash
     source env/bin/activate
     ```
   - Windows:
     ```bash
     env\Scripts\activate
     ```

3. **Install the Dependencies**  
   Use the `requirements.txt` file to install the required packages:
   ```bash
   pip install -r notebook/requirements.txt
   ```
   If you encounter installation issues, try:
   ```bash
   pip install --force-reinstall -r notebook/requirements.txt
   ```

4. **Verify the Installations**  
   Check installed packages:
   ```bash
   pip list
   ```

5. **To Deactivate the Virtual Environment:**  
   ```bash
   deactivate
   ```

---

## Running the Jupyter Notebook

1. **Start the Jupyter Notebook Server**  
   ```bash
   jupyter notebook
   ```

2. **Open the Notebook File**  
   Navigate to the `.ipynb` file (e.g., `analysis.ipynb`) in your web browser and open it.

3. **Run the Code Cells**  
   Execute the code cells step by step or run all cells at once to perform the analysis.

---

## Notes

- Ensure your Python version is **3.11.7** to maintain compatibility with the dependencies.
- If you encounter issues, try updating `pip`:
  ```bash
  pip install --upgrade pip
  ```

---

## Converting Jupyter Notebooks to PDF

### Install LaTeX (Required for PDF Conversion)
LaTeX is required for Jupyter notebook PDF conversion. Install a LaTeX distribution if not already installed:

- **macOS:** Install MacTeX via the [MacTeX website](https://www.tug.org/mactex/).
- **Windows/Linux:** Install TeX Live or MikTeX from their official websites.

### Ensure `nbconvert` is Installed
```bash
pip install nbconvert
```

### Add LaTeX Binaries to PATH (macOS Users)
If using MacTeX, ensure LaTeX binaries are in your PATH:

1. Open a terminal and edit the `.zshrc` or `.bashrc` file:
   ```bash
   nano ~/.zshrc
   ```
   Or use another editor like `vim` or `code`.

2. Add the following line at the end of the file:
   ```bash
   export PATH="/usr/local/texlive/2024/bin/universal-darwin:$PATH"
   ```

3. Save and exit the file.

4. Reload your shell configuration:
   ```bash
   source ~/.zshrc
   ```

---

## Additional Requirements

This project requires Pandoc to convert Jupyter notebooks to PDF.

### Install Pandoc (macOS/Linux Users)
```bash
brew install pandoc
```

### Convert a Jupyter Notebook to PDF
To convert a Jupyter notebook (`analysis.ipynb`) to a PDF, run:
```bash
jupyter nbconvert --to pdf "Final_Project_Section2_Team8.ipynb"
```

Ensure that LaTeX is installed on your system for successful PDF generation.

### Convert a Jupyter Notebook to Text
Run this command to convert the notebook to plain text:
```bash
jupyter nbconvert --to script "Final_Project_Section2_Team8.ipynb"
```

### Convert a Jupyter Notebook to Markdown
Run this command to convert the notebook to Markdown (.md) file in the same directory:
```bash
jupyter nbconvert --to markdown "Final_Project_Section2_Team8.ipynb"
```

---

## Additional Notes
- Ensure all dependencies in the `requirements.txt` file are installed.
- The `nbconvert` tool also supports other formats like HTML, Markdown, and slides.
- If conversion fails, check your LaTeX installation and dependencies.

---

## Check Jupyter Notebook code for PEP8 compliance

auto-format your notebook according to PEP8 standards, use black:

Step 1: Install black
```bash
pip install 'black[jupyter]'
```

Step 2: Format Your Jupyter Notebook
```bash
black Final_Project_Section2_Team8.ipynb
```
---