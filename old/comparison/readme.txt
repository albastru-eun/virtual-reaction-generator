Run the code sequentially on Anaconda

conda create -n augment_1 python=3.10 -y
conda activate augment

conda install -c conda-forge rdkit -y
pip install rxnutils
conda install -c conda-forge rdkit -y

git clone https://github.com/hesther/templatecorr.git
cd templatecorr
pip install -e .
cd ..

After the installation, put raw_train.csv in the folder,
then run 1 to 4 using python, sequentially.

Now you can find augmented files at /output_datasets