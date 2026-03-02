# High-Throughput-Screening-of-HOIPs-via-Machine-Learning
Machine learning models for paper "High-Throughput Screening of Highly Piezoelectric Hybrid Organic–Inorganic Perovskites via Density-Functional Theory and Machine Learning"

The pre-trained models needed for prediction can be found at: https://zenodo.org/records/18839158

To use the pre-trained models, before running, please set up your development environment: ensure that all necessary libraries and dependencies are installed. You can then run:
```
conda create -n py39 python=3.9
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
python -m pip install "tensorflow<2.11"
pip install numpy==1.26.4
pip install matplotlib==3.8.0
pip install pandas==1.5.3
pip install rdkit==2023.9.1
pip install scikit-learn==1.3.0
pip install keras-tuner==1.4.5
pip install seaborn==0.13.0
pip install dscribe
```

To predict polymer viscosity using FNN and PINN models, save the dataframe containing the polymer's SMILES information and temperature into a .csv file, and then use a command like:

```
python predict_from_cif.py /path/to/your_structure.cif --model-dir /path/to/models
```
