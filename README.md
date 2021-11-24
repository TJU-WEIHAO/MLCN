# Disclaimer
This is a repository of Multi-Level Consistency Network (MLCN) for CVPR2022 blind review !!!

# MLCN — Official PyTorch Implementation
**Title: "A Multi-Level Consistency Network for High-Fidelity Virtual Try-On"**  
-------------------------------  Paper ID: 8343  -------------------------------
# Installation
Clone this repository:  
> git clone https://github.com/2022-CVPR-8343/MLCN.git  
> cd ./MLCN/

Create a virtual environment:
> conda create -n [ENV] python=3.8  
> conda activate [ENV]

Install PyTorch and other dependencies:  
> pytorch == 1.8  
> torchvision == 0.9.0  
> opencv == 4.5.3  
> scipy == 1.7.1  
> pillow == 8.3.1 

Install deformable convolution:  
> sh make.sh  

# Pre-trained networks
We provide pre-trained networks and the testing set from VITON dataset. Please download ./checkpoints the [CHECKPOINTS Google Drive folder](https://drive.google.com/drive/folders/1-CWgyodbc_kB0YCPIw89BSS6Oap6UtLc?usp=sharing) and put the downloaded files in ./checkpoints/ directory.
 
