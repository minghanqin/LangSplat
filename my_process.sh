#!/bin/bash

# get the language feature of the scene
# python preprocess.py --dataset_name $dataset_path

# # train the autoencoder
# cd autoencoder
# python train.py --dataset_path $dataset_path --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name ae_ckpt
# # e.g. python train.py --dataset_path ../dataset/lerf_ovs/teatime --encoder_dims 256 128 64 32 4 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name teatime


# python train.py --dataset_path ../dataset/lerf_ovs/ramen --encoder_dims 256 128 64 32 4 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name ramen

# # get the 3-dims language feature of the scene
# python test.py --dataset_name $dataset_path --dataset_name $dataset_name
# # e.g. python test.py --dataset_path ../dataset/lerf_ovs/teatime --dataset_name teatime --encoder_dims 256 128 64 32 4 --decoder_dims 16 32 64 128 256 256 512

# # ATTENTION: Before you train the LangSplat, please follow https://github.com/graphdeco-inria/gaussian-splatting
# # to train the RGB 3D Gaussian Splatting model.
# # put the path of your RGB model after '--start_checkpoint'

# for level in 1 2 3
# do
# #     # python train.py -s $dataset_path -m output/${casename} --start_checkpoint $dataset_path/$casename/chkpnt30000.pth --feature_level ${level}
# #     # # e.g. python train.py -s data/sofa -m output/sofa --start_checkpoint data/sofa/sofa/chkpnt30000.pth --feature_level 3
# #     # # python train.py -s dataset/lerf_ovs/teatime -m output/teatime --start_checkpoint /datadrive/yingwei/gaussian-splatting/output/0e94792f-2/chkpnt30000.pth --feature_level 3
# #     # python train.py -s dataset/lerf_ovs/teatime -m dataset/lerf_ovs/teatime/output/teatime --start_checkpoint dataset/lerf_ovs/teatime/output/teatime/chkpnt30000.pth --feature_level ${level}
# #     # python train.py -s dataset/sofa -m dataset/sofa/sofa --start_checkpoint dataset/sofa/sofa/chkpnt30000.pth --feature_level 3
# #     python train.py -s dataset/sofa -m dataset/sofa/sofa --start_checkpoint dataset/sofa/sofa/chkpnt30000.pth --feature_level ${level}
#     python train.py -s dataset/lerf_ovs/ramen -m dataset/lerf_ovs/ramen/output/ramen --start_checkpoint dataset/lerf_ovs/ramen/output/ramen/chkpnt30000.pth --feature_level ${level}
# done

# for level in 1 2 3
# do
#     # render rgb
#     #python render.py -m /datadrive/yingwei/LangSplat/dataset/lerf_ovs/teatime/output/teatime_${level}
#     #python render.py -m dataset/sofa/sofa_${level}
#     python render.py -m /datadrive/yingwei/LangSplat/dataset/lerf_ovs/ramen/output/ramen_${level}

#     # render language features
#     #python render.py -m /datadrive/yingwei/LangSplat/dataset/lerf_ovs/teatime/output/teatime_${level} --include_feature
#     #python render.py -m dataset/sofa/sofa_${level} --include_feature
#     # e.g. python render.py -m output/sofa_3 --include_feature
#     python render.py -m /datadrive/yingwei/LangSplat/dataset/lerf_ovs/ramen/output/ramen_${level} --include_feature
# done


# dim discover WORKFLOW

# !!! CLEARN THE PREV OUTPUT BEFORE START NEW DIM !!!
# EDIT FILES:
# >> rasterization edit <<
# /datadrive/yingwei/LangSplat/submodules/langsplat-rasterization
# Step 1: delete build file and egg info
# Step 2: cuda_rasterizer/config.h  change NUM_CHANNELS_language_feature
# Step 3: pip uninstall diff-gaussian-rasterization -y & pip install submodules/langsplat-rasterization/
#
# >> eval edit <<
# Step 4: go to eval/eval.sh change CASE_AC_DIM 



# Global variable
language_feature_dim=15
case_name="ramen"

# Train auto-encoder
cd autoencoder
python train.py --dataset_path ../dataset/lerf_ovs/${case_name} --encoder_dims 256 128 64 32 ${language_feature_dim} --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name ${case_name}

# get the n-dims language feature of the scene
python test.py --dataset_path ../dataset/lerf_ovs/${case_name} --dataset_name $case_name --encoder_dims 256 128 64 32 ${language_feature_dim} --decoder_dims 16 32 64 128 256 256 512

cd ..

# start to train the Langsplat
# cannot use variable here - must replace it with desired case name
for level in 1 2 3
do
    python train.py -s dataset/lerf_ovs/ramen -m dataset/lerf_ovs/ramen/output/ramen --start_checkpoint dataset/lerf_ovs/ramen/output/ramen/chkpnt30000.pth --feature_level ${level} --language_feature_dim $language_feature_dim
done

# render

# remember to replace the ramen here if use other dataset
for level in 1 2 3
do
    # render rgb
    python render.py -m /datadrive/yingwei/LangSplat/dataset/lerf_ovs/ramen/output/ramen_${level} --language_feature_dim $language_feature_dim

    # render language features
    python render.py -m /datadrive/yingwei/LangSplat/dataset/lerf_ovs/ramen/output/ramen_${level} --include_feature  --language_feature_dim $language_feature_dim
done

# eval
cd eval
sh eval.sh
