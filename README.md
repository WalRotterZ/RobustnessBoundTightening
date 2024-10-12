# RobustnessBoundTightening

Requirements

Linux 
The instructions to install all the requirements for Linux are:

conda config --add channels conda-forge
conda create -n DeepSRGR python=3.7.8
conda activate DeepSRGR
conda install pkgconfig
conda install cvxopt cvxpy-base cvxpy glpk numpy scipy blas libblas libcblas coin-or-cbc
pip install cylp

Windows 
The instructions to install all the requirements for Windows are:

conda config --add channels conda-forge
conda create -n DeepSRGR python=3.7.8
conda activate DeepSRGR
conda install pkgconfig
conda install cvxopt cvxpy-base cvxpy glpk numpy scipy blas libblas libcblas
pip install cbcpy
pip install cylp
