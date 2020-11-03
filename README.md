This repository is for reproducing following paper
about the *BCD-NET* approach to image reconstruction: 

Hongki Lim, Il Yong Chun, Yuni Dewaraja, and Jeffrey Fessler:
"Improved low-count quantitative PET reconstruction with a variational neural network."
[IEEE Transactions on Medical Imaging, 39(11):3512-22, Nov. 2020.](http://doi.org/10.1109/TMI.2020.2998480)

[arXiv version of paper.](https://arxiv.org/abs/1906.02327)


## Setting up and Reproducing

To reproduce the paper, please make sure you have the following:
Michigan Image Reconstruction Toolbox (MIRT) installed:
http://web.eecs.umich.edu/~fessler/code/index.html.  

Please download dataset via following link: 
https://drive.google.com/open?id=1VPcpI44LBNhKSYQ9EtMC6k-vTwDwRd_r

Modify paths in `pcodes_init.m` and `train_matlab.py` (l8th line) in `mypcodes` folder.

Then run `main_recon_bcd_net_sca.m`.
