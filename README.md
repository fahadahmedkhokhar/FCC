# FCC
Fail Controlled Classifier


[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%20-blue.svg)]()

## Requirements

- Python 3.8
- PyTorch
- PyTorch-Lightning


## Installation

The code uses **Python 3.8**.

#### Create a Conda virtual environment:

```bash
conda create --name FCC python=3.8
conda activate FCC
```

#### Clone the project and install requirements:

1. Clone the repository:

```bash
git clone [https://github.com/fahadahmedkhokhar/FCC.git]
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run Image Classifier with FCCs:
```bash
python main.py 
```
4. Run Tabular Classifier with FCCs:
```bash
python tabular_main.py
```

4. Run Updated FCCs (RB, VT, WVT and STK):
```bash
python updated_FCC.py
```
## Compute the Results
For computing the Metrices:
```bash
python Results/compute_matrix.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

