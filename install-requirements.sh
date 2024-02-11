#!/bin/bash

# Function to install Miniconda
install_miniconda() {
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    conda init bash
    source ~/.bashrc
    echo "Miniconda installed."
}

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found. Installing Miniconda."
    install_miniconda
else
    echo "Conda is already installed."
fi

# Install requirements from different files with respective extra-index-urls
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib scikit-learn scipy seaborn
pip install -U pip
pip install --extra-index-url=https://example-url1.com -r requirements1.txt
pip install --extra-index-url=https://example-url2.com -r requirements2.txt
# Add more lines as needed for different requirements files and URLs
conda create --solver=libmamba -n rapids-23.12 -y -c rapidsai -c conda-forge -c nvidia  \
    rapids=23.12 python=3.10 cuda-version=11.8 \
    dask-sql jupyterlab dash graphistry tensorflow xarray-spatial pytorch
