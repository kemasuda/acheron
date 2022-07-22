ages of Kepler stars with (and without) rotational modulation



## Detectabiltiy of rotational modulation as a function of age

Code for reproducing the analyses in: "Detectability of Rotational Modulation in Kepler Sun-like Stars as a Function of Age," (Masuda 2022b, ApJ submitted). 

- cks_frot: infer the fraction of stars with rotation period detection as a function of mass and age for the CKS stars (Section 3 and 5 of the paper). Depends on [jhbayes](https://github.com/kemasuda/jhbayes) for hierarhical modeling.

  - plots: main results

  - detectability: comparison of modulation amplitudes in Mazeh+15 and McQuillan+14 samples (Section 6.1)

  - detection_model: simple detection model used in Figures 12 and 13

- posterior_samples: posterior samples from isochrone fitting (avaialble from the author) obtained with Hamiltonian Monte Carlo using [jaxstar](https://github.com/kemasuda/jaxstar).
  - posteriors_cks_linage2: isochrone-only results for the CKS sample
  - posteriors_cksjoint_linage2: joint isochrone-gyrochrone results for the CKS sample
  - posteriors_hall_linage: isochrone-only results for the asteroseismic sample (Section 2.3)

- tests_simulation: test results using simulated data (Section 2.2, Section 4.3)
  - input: information of the simulated stars
  - posteriors_simulated_cks3: posterior samples from isochrone fitting (just include one case here; avaialble from the author)
  - hierarchical_recovery: inference of the mass-age distribution


-  tests_astero: comparison with asteroseismic stars in Section 2.3.
-  input: star info



## Rotational modulation amplitude vs rotation period/Rossby number

Notebooks used in: "On the Evolution of Rotational Modulation Amplitude in Solar-mass Main-sequence Stars," (Masuda 2022a, ApJ accepted) https://arxiv.org/abs/2206.01595

- kepler_prot_teff
