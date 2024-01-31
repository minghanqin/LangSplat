#!/bin/bash

if [ $# -eq 0 ]; then
    echo "No argument passed"
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        -dn|--dataset_name) # name of the dataset
        dataset_name="$2"
        shift
        shift
        ;;
        -p|--dataset_abspath) # absolute path of the dataset
        dataset_abspath="$2"
        shift
        shift
        ;;
        -cn|--casename) # casename of the 3dgs train output (ex. 469867ee-e)
        casename="$2"
        shift
        shift
        ;;
        -ckpt|--ae_ckpt) # name of autoencoder checkpoint
        ae_ckpt="$2"
        shift
        shift
        ;;
        *)
        echo "Unknown option $1"
        exit 1 # incorrect format will terminate the process
        ;;
    esac
done

# get the language feature of the scene
python preprocess.py --dataset_path $dataset_abspath

# train the autoencoder
cd autoencoder
python train.py --dataset_name $dataset_name --dataset_path $dataset_abspath \
                --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 \
                --lr 0.0007 --output $ae_ckpt
# get the 3-dims language feature of the scene
python test.py --dataset_name $dataset_name --dataset_path $dataset_abspath --ckpt_name $ae_ckpt --output $ae_ckpt
cd ..

# ATTENTION: Before you train the LangSplat, please follow https://github.com/graphdeco-inria/gaussian-splatting
# to train the RGB 3D Gaussian Splatting model.
# put the path of your RGB model after '--start_checkpoint'

for level in 1 2 3
do
    python train.py -s $dataset_abspath -m output/${casename} --start_checkpoint $dataset_abspath/output/$casename/chkpnt30000.pth --feature_level ${level}
done

for level in 1 2 3
do
    python render.py -m output/${casename}_${level} # render rgb
    
    python render.py -m output/${casename}_${level} --include_feature # render language features
done