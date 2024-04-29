# SealD-NeRF

## Install
```
git clone --recursive https://github.com/ashawkey/torch-ngp.git
cd torch-ngp
```
### Install with pip
```
pip install -r requirements.txt
```

### Install with conda
```
conda env create -f environment.yml
conda activate torch-ngp
```

### Build extension (optional)
By default, we use load to build the extension at runtime. However, this may be inconvenient sometimes. Therefore, we also provide the setup.py to build each extension:
```
# install all extension modules
bash scripts/install_ext.sh

# if you want to install manually, here is an example:
cd raymarching
python setup.py build_ext --inplace # build ext only, do not install (only can be used in the parent directory)
pip install . # install to python path (you still need the raymarching/ folder, since this only install the built extension.)
```

https://github.com/windingwind/seal-3d
