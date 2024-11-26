# EIT Complete Electrode Model

This repository contains a FEM solver for the complete electrode model (CEM) used in electrical impedance tomography (EIT). The FEM solver is implemented in FenicsX. The Jacobian computation follows the work by Margotti ([Section 5.2.2.](https://publikationen.bibliothek.kit.edu/1000048606))

This solver was used in the submission to KTC2023 by the team Alexander Denker, Zeljko Kereta, Imraj Singh, Tom Freudenberg, Tobias Kluth, Simon Arridge and Peter Maass from University of Bremen and University College London.

### Background 

Let $\Omega$ be the domain with boundary $\partial \Omega$ and $L$ electrodes $e_l \subset \partial \Omega$ for $l=1,\dots,L$. The electric potential $u$ is governed by the following PDE 

$$ - \nabla \cdot(\sigma \nabla u) = 0 \quad \text{in } \Omega, $$

where $\sigma$ is the conductivity distribution. In EIT we apply a current $I = (I_1, \dots, I_L)$ on the electrodes and measure the voltage $U = (U_1, \dots, U_L)$. In the CEM we model this with boundary conditions

$$ u + z_l \sigma \frac{\partial u}{\partial \nu} = U_l, \quad \text{on } e_l, l=1,\dots,L $$
$$ \sigma \frac{\partial u}{\partial \nu} = 0, \quad \text{on } \partial \Omega \setminus  \cup_l e_l$$
$$ \int_{e_l} \sigma \frac{\partial u}{\partial \nu} ds = I_l, \quad \text{on } e_l, l=1,\dots,L $$

where $z = (z_1, \dots, z_L)$ are contact impedances. Further, we have an additional mean-free condition for the potential

$$ \sum_{l=1}^L U_l = 0. $$

We solve the CEM for a fixed $\sigma$ using a Finite-Element Method. To include the mean-free condition we introduce a Lagrange multiplier $\lambda \in \mathbb{R}$. The weak formulation reads:

$$ \int_\Omega \sigma \nabla u \cdot \nabla v dx + \sum_{l=1}^L \frac{1}{z_l} \int_{e_l} (u - U_l)(v - V_l) ds + \sum_{l=1}^L (\lambda V_l + \nu U_l) = \sum_{l=1}^L I_l V_l $$

for $(v, V, \nu) \in H^1(\Omega) \times \mathbb{R}^L \times \mathbb{R}$. Note that the current pattern $I = (I_1, \dots, I_L)$ only appears on the RHS of the equation. When considering the linear system we can reuse intermediate steps, e.g., the LU factorisation of the system matrix, to compute the solution $(u,U)$ for different current patterns $I$. 

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