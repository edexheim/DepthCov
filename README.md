
# DepthCov

Code for the paper "Learning a Depth Covariance Function" including the training pipeline and a real-time dense monocular visual odometry pipeline DepthCov-VO.

## Setup

We have included the ```environment.yml``` file for the exact environment used, but the full instructions for required packages are below.

Create a new conda environment
```
conda create -n depth_cov python=3.8.15
```

Activate the environment via
```
conda activate depth_cov
```

Install the following packages:
```
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
conda install matplotlib h5py
pip install pytorch-lightning==1.8.2 opencv-python-headless open3d pyrealsense2 
```

For the odometry backend, we require the batched SE3 exponential map from Lietorch https://github.com/princeton-vl/lietorch.  Clone the repo and run the setup command while in the new conda environment:
```
git clone --recursive https://github.com/princeton-vl/lietorch.git
cd lietorch
python setup.py install
```

Lastly, in the virtual environment, go back to the base directory and install
``` 
cd ..
pip install -e . 
``` 
Executables in "depth_cov/" including the odometry can then be run.

Please note that the "extra_compile_args" in "setup.py" for both LieTorch and DepthCov may need to be modified to match the desired GPU/CUDA setup.

## Network Training

For ScanNet, we used the repo https://github.com/jczarnowski/scannet_toolkit to setup training and validation splits.  We used the raw depth maps for training.

For the ScanNet dataloader, we require a .txt file where each row is a comma separated list of "RGB_path,depth_path".  

For the NYUv2 dataloader, we require a .txt file where each row is the .h5 file containing both the RGB and depth. 

Use ```depth_cov/train.py``` to jointly train the GP hyperparameters and the UNet parameters.

## Depth Completion Examples

https://github.com/edexheim/DepthCov/assets/47486413/22239c2e-1aaf-4b89-98be-320ffed0eef0


To see how to perform depth completion, see the ```depth_cov/depth_completion_viz.py``` example, which shows the output conditional mean and variance given randomply sampled observations.

We also include an interactive demo to visualize the effect of adding new observations and how a covariance matrix looks in ```depth_cov/active_depth_viz.py```.  The user may click on the RGB image or the standard deviation visualization to improve the depth estimate.

## Bundle Adjustment with Depth Priors

https://github.com/edexheim/DepthCov/assets/47486413/fd49812b-6ea8-4925-95aa-796f15c229c8

To run the bundle adjustment example ```depth_cov/bundle_adjustment.py```, follow the additional installation instructions at https://github.com/edexheim/DepthCovFactors. 

## Real-Time Odometry (DepthCov-VO)


https://github.com/edexheim/DepthCov/assets/47486413/5b7e2658-ade4-48fb-9935-c6e805b466bb


The file ```depth_cov/odom_dataset.py``` can be used to run the odometry on a dataset while ```depth_cov/odom_demo.py``` is set up to run the real-time odometry with an Intel RealSense.  

We have tested the odometry on CUDA 11 with an RTX 3080.

Since we have now focused on the real-time aspect, the odometry code has been overhauled since the original paper version.  There are now multiple threads and some added features.  We include a config file that was originally used for the single threaded odometry with one tracking and one mapping iteration per frame in the paper, but this has not been tuned for the new system and lacks the improved functionality, as we focus on achieving ~30 FPS.  

### TUM

Download sequences from https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download.

### ScanNet

Please refer to the ScanNet Training section on how to download sequences.

### Real-time with Intel RealSense

We use the RGB stream for a RealSense camera via the pyrealsense2 API.  We assume the factory calibration is sufficient, as we load this directly in the dataloader.  The default stream settings we use are 640x480 at 60 FPS, but for USB2 this may not be possible, so adjust ```realsense.yml``` as needed.


# Citation
If you found this code/work to be useful in your own research, please consider citing the following:
```bibtex
@inproceedings{dexheimer2023depthcov,
  title={{Learning a Depth Covariance Function},
  author={Eric Dexheimer and Andrew J. Davison},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).},
  year={2023},
  }
}
```
