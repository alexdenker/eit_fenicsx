# EIT Complete Electrode Model

This repository contains a FEM solver for the complete electrode model (CEM) used in electrical impedance tomography (EIT). The FEM solver is implemented in FenicsX. The Jacobian computation follows the work by Margotti ([Section 5.2.2.](https://publikationen.bibliothek.kit.edu/1000048606))

This solver was used in the submission to KTC2023 by the team Alexander Denker, Zeljko Kereta, Imraj Singh, Tom Freudenberg, Tobias Kluth, Simon Arridge and Peter Maass from University of Bremen and University College London.

## Installation 

You will need to install FenicsX:

```python
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista
```

FenicsX is compatible with pytorch. The code is tested with pytorch version 2.3.1 and CUDA 12.1. This can be installed via:

```python
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Example 

Examples will be coming soon.


### Citation

If you use this FEM solver in your work, please cite:

```
@article{denker2024data,
  title={Data-driven approaches for electrical impedance tomography image segmentation from partial boundary data},
  author={Denker, Alexander and Kereta, {\v{Z}}eljko and Singh, Imraj and Freudenberg, Tom and Kluth, Tobias and Maass, Peter and Arridge, Simon},
  journal={Applied Mathematics for Modern Challenges},
  pages={0--0},
  year={2024},
  publisher={Applied Mathematics for Modern Challenges}
}
```