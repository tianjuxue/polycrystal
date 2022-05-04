# Phase-field simulation of microstructure evolution in additive manufacturing

This is the repository for our project on microstructure evolution using the phase-field method. We implemented both classic direct numerical simulation (DNS) based on the finite difference method and a new physics-embedded graph network (PEGN) approach. The code runs on both CPU and GPU. More demos and documentation coming soon! 

## Requirements 

We use [JAX](https://github.com/google/jax) for implementation of the computationally intensive part. The graph construction is based on [Jraph](https://github.com/deepmind/jraph). The polycrystal structure is generated with [Neper](https://neper.info/). 

## Descriptions

We describe the typical workflow of using the code.

```
python -m src.single_track
```




<p align="middle">
  <img src="materials/single_track_T_DNS.gif" width="400" />
  <img src="materials/single_track_T_PEGN.gif" width="400" /> 
</p>

<p align="middle">
  <img src="materials/single_track_zeta_DNS.gif" width="400" />
  <img src="materials/single_track_zeta_PEGN.gif" width="400" /> 
</p>

<p align="middle">
  <img src="materials/single_track_eta_DNS.gif" width="400" />
  <img src="materials/single_track_eta_PEGN.gif" width="400" /> 
</p>


<p align="middle">
  <img src="materials/solidification_isotropic.gif" width="400" />
  <img src="materials/solidification_anisotropic.gif" width="400" /> 
</p>


