This repository is for reproducing following paper: 

Lim, Hongki, Yuni K. Dewaraja, and Jeffrey A. Fessler. "A PET reconstruction formulation that enforces non-negativity in projection space for bias reduction in Y-90 imaging." Physics in medicine and biology (2018).

[Find the paper here.](https://doi.org/10.1088%2F1361-6560%2Faaa71b)

BibTeX:
```
@article{lim2018pet,
  title={A PET reconstruction formulation that enforces non-negativity in projection space for bias reduction in Y-90 imaging},
  author={Lim, Hongki and Dewaraja, Yuni K and Fessler, Jeffrey A},
  journal={Physics in medicine and biology},
  year={2018},
  publisher={IOP Publishing}
}
```

## Setting up and Reproducing

To reproduce the paper, please make sure you have the following:

Michigan Image Reconstruction Toolbox (MIRT) installed: http://web.eecs.umich.edu/~fessler/code/index.html.  

Then simply run "main.m".

"main.m" includes reconstructions using SPS-Reg, Neg-ML-Reg, ADMM-Reg in the paper. 
There are comments in the code. 
Please contact "hongki@umich.edu" if you have any questions.  

Updated 02/07/2018

