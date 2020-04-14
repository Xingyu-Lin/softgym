cd ${PYFLEXROOT}/bindings
rm -rf build
mkdir build
cd build
# Seuss 
if [[ $(hostname) = *"compute-0"* ]] || [[ $(hostname) = *"yertle"* ]]; then
    export CUDA_BIN_PATH=/usr/local/cuda-9.1
fi
cmake ..
make -j
