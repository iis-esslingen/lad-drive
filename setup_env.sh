eval "$(conda shell.bash hook)"

conda create -n lad_drive python=3.8 -y
conda activate lad_drive 

pip install --upgrade pip
pip install torch==2.0.1 torchvision
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html  
pip install pygame pyyaml scikit-image tqdm
pip install contexttimer decord "diffusers<=0.16.0" "einops>=0.4.1" "fairscale==0.4.4" ftfy iopath ipython omegaconf opendatasets packaging pandas plotly pre-commit pycocoevalcap pycocotools python-magic scikit-image sentencepiece streamlit webdataset peft
pip install spacy --prefer-binary
pip install easydict dictor "py-trees==0.8.3" "scikit-image==0.21.0" "networkx==3.1" "shapely>=1.7.1,<1.8" psutil "xmlschema==1.0.18" ephem tabulate "opencv-python==4.2.0.32" numpy matplotlib six "transformers==4.31.0" timm

conda install -c conda-forge shapely=1.6.4 libjpeg-turbo=2.1 "jpeg<9" tensorboardx

pip install torch_geometric

cd ./vision_encoder && python setup.py develop 
cd ./LAVIS && python setup.py develop

# ./setup_carla.sh
pip install carla

git lfs install

git clone https://huggingface.co/liuhaotian/llava-v1.5-7b
git clone https://huggingface.co/OpenDILabCommunity/LMDrive-llava-v1.5-7b-v1.0
git clone https://huggingface.co/OpenDILabCommunity/LMDrive-vision-encoder-r50-v1.0

sed -i 's/, *cached_download//g' /home/es/es_es/es_kafeit00/miniconda3/envs/lad_drive/lib/python3.8/site-packages/diffusers/utils/dynamic_modules_utils.py

echo "conda activate lad_drive" >> ~/.bashrc