ages of Kepler stars with (and without) rotational modulation



### Detectabiltiy of rotational modulation as a function of age

***

Code for reproducing the analyses in: "Detectability of Rotational Modulation in Kepler Sun-like Stars as a Function of Age," (Masuda 2022b, ApJ submitted). 

- cks_frot

Infer the fraction of stars with Prot detection as a function of mass and age for the CKS stars (Section 3,4,5 of the paper). Depends on [jhbayes](https://github.com/kemasuda/jhbayes) for hierarhical modeling and [jaxstar](https://github.com/kemasuda/jaxstar) for obtaining posterior samples from isochrone fitting via Hamiltonian Monte Carlo. The posterior samples are not in this repository but avaialble from the author.

- tests_simulation

Results of the injection-recovery tests in Section 2.2.

-  tests_astero

Comparison with asteroseismic stars in Section 2.3.



### Rotational modulation amplitude vs rotation period/Rossby number

***

Notebooks used in: "On the Evolution of Rotational Modulation Amplitude in Solar-mass Main-Sequence Stars," (Masuda 2022a, ApJ accepted) https://arxiv.org/abs/2206.01595

- kepler_prot_teff
