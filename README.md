# MS-AAI
## AAI-501 Final Project - Group 8

### Final Project for the AAI-501 (Machine Learning: Fundamentals and Applications) Course in the **Master of Applied Artificial Intelligence** Program
### University of San Diego

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
   If using Git:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
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
   pip install -r requirements.txt
   ```
   If you encounter installation issues, try:
   ```bash
   pip install --force-reinstall -r requirements.txt
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
jupyter nbconvert --to pdf "Group_8_Final_Team_Project_Notebook.ipynb"
```

Ensure that LaTeX is installed on your system for successful PDF generation.

### Convert a Jupyter Notebook to Text
Run this command to convert the notebook to plain text:
```bash
jupyter nbconvert --to script "Group_8_Final_Team_Project_Notebook.ipynb"
```

### Convert a Jupyter Notebook to Markdown
Run this command to convert the notebook to Markdown (.md) file in the same directory:
```bash
jupyter nbconvert --to markdown "Group_8_Final_Team_Project_Notebook.ipynb"
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
black Group_8_Final_Team_Project_Notebook.ipynb
```
---