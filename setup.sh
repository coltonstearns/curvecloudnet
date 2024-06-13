CONDA_PATH=$1
CONDA_ENV_NAME=$2

# create conda environment
source ${CONDA_PATH}/bin/activate
conda create -n ${CONDA_ENV_NAME} python=3.10 -y
source ${CONDA_PATH}/bin/activate ${CONDA_ENV_NAME}

# install Pytorch and Open3d
yes | pip install numpy
yes | pip install open3d
pip install tensorboard~=2.8.0
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y

# install Pytorch Geometric
conda install pytorch-scatter==2.1.1 -c pyg -y
conda install pytorch-sparse==0.6.17 -c pyg -y
conda install pytorch-cluster==1.6.1 -c pyg -y
yes | pip install torch-geometric==2.3.0 --no-cache-dir

# Install Pytorch3D from scratch
yes | pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install all other Requirements
yes | pip install -r requirements.txt

# External dependencies
cd third_party/FRNN/external/prefix_sum
pip install .
cd ../..
pip install -e .
cd ../..

cd third_party/minimal_pytorch_rasterizer
pip install .
cd ../..

pip install --upgrade charset-normalizer
pip install shortuuid
