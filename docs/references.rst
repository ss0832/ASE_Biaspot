References
==========

The following works should be cited when using ASE_Biaspot in academic publications.

AFIR (Artificial Force Induced Reaction)
-----------------------------------------

The AFIR energy function implemented in :mod:`ase_biaspot.afir` is based on the
single-component AFIR method developed by Maeda, Morokuma, and co-workers.
Please cite the following papers when using the AFIR term:

1. Maeda, S.; Morokuma, K.
   *Finding Reaction Pathways of Type A + B → X: Toward Systematic Prediction
   of Reaction Mechanisms.*
   **Chem. Rec.** 2016, *16*, 2232–2248.
   DOI: `10.1002/tcr.201600045 <https://doi.org/10.1002/tcr.201600045>`_

2. Maeda, S.; Harabuchi, Y.; Ono, Y.; Taketsugu, T.; Morokuma, K.
   *Intrinsic Reaction Coordinate: Calculation, Bifurcation, and Automated
   Search.*
   **J. Comput. Chem.** 2018, *39*, 233–251.
   DOI: `10.1002/jcc.25047 <https://doi.org/10.1002/jcc.25047>`_

3. Maeda, S.; Harabuchi, Y.
   *Exploring paths of chemical transformations in molecular and periodic
   systems: An approach utilizing force.*
   **WIREs Comput. Mol. Sci.** 2021, *11*, e1538.
   DOI: `10.1002/wcms.1538 <https://doi.org/10.1002/wcms.1538>`_

ASE (Atomic Simulation Environment)
-------------------------------------

ASE_Biaspot builds on top of the Atomic Simulation Environment (ASE).
Please also cite the ASE paper when using this library:

4. Larsen, A. H.; Mortensen, J. J.; Blomqvist, J.; Castelli, I. E.;
   Christensen, R.; Dułak, M.; Friis, J.; Groves, M. N.; Hammer, B.;
   Hargus, C.; Hermes, E. D.; Jennings, P. C.; Jensen, P. B.;
   Kermode, J.; Kitchin, J. R.; Kolsbjerg, E. L.; Kubal, J.;
   Kaasbjerg, K.; Lysgaard, S.; Maronsson, J. B.; Maxson, T.;
   Olsen, T.; Pastewka, L.; Peterson, A.; Rostgaard, C.; Schiøtz, J.;
   Schütt, O.; Strange, M.; Thygesen, K. S.; Vegge, T.; Vilhelmsen, L.;
   Walter, M.; Zeng, Z.; Jacobsen, K. W.
   *The atomic simulation environment — a Python library for working with atoms.*
   **J. Phys.: Condens. Matter** 2017, *29*, 273002.
   DOI: `10.1088/1361-648X/aa680e <https://doi.org/10.1088/1361-648X/aa680e>`_

NEB (Nudged Elastic Band)
--------------------------

The NEB method used in the quickstart Claisen example (Step 2) was introduced in:

5. Henkelman, G.; Uberuaga, B. P.; Jónsson, H.
   *A climbing image nudged elastic band method for finding saddle points and
   minimum energy paths.*
   **J. Chem. Phys.** 2000, *113*, 9901–9904.
   DOI: `10.1063/1.1329672 <https://doi.org/10.1063/1.1329672>`_

Psi4
-----

When using the Psi4 quantum-chemistry engine as the base calculator (as in the
quickstart Claisen example), please also cite:

6. Smith, D. G. A.; Burns, L. A.; Simmonett, A. C.; Parrish, R. M.;
   Schieber, M. C.; Galvelis, R.; Kraus, P.; Kruse, H.; Di Remigio, R.;
   Alenaizan, A.; James, A. M.; Lehtola, S.; Misiewicz, J. P.;
   Scheurer, M.; Shaw, R. A.; Sherrill, C. D.; et al.
   *PSI4 1.4: Open-source software for high-throughput quantum chemistry.*
   **J. Chem. Phys.** 2020, *152*, 184108.
   DOI: `10.1063/5.0006002 <https://doi.org/10.1063/5.0006002>`_

BibTeX entries
--------------

.. code-block:: bibtex

    @article{maeda2016afir,
      author  = {Maeda, Satoshi and Morokuma, Keiji},
      title   = {Finding Reaction Pathways of Type A + B → X: Toward Systematic
                 Prediction of Reaction Mechanisms},
      journal = {Chem. Rec.},
      year    = {2016},
      volume  = {16},
      pages   = {2232--2248},
      doi     = {10.1002/tcr.201600045},
    }

    @article{maeda2018afir,
      author  = {Maeda, Satoshi and Harabuchi, Yu and Ono, Yuriko and
                 Taketsugu, Tetsuya and Morokuma, Keiji},
      title   = {Intrinsic Reaction Coordinate: Calculation, Bifurcation,
                 and Automated Search},
      journal = {J. Comput. Chem.},
      year    = {2018},
      volume  = {39},
      pages   = {233--251},
      doi     = {10.1002/jcc.25047},
    }

    @article{maeda2021afir,
      author  = {Maeda, Satoshi and Harabuchi, Yu},
      title   = {Exploring paths of chemical transformations in molecular and
                 periodic systems: {A}n approach utilizing force},
      journal = {WIREs Comput. Mol. Sci.},
      year    = {2021},
      volume  = {11},
      pages   = {e1538},
      doi     = {10.1002/wcms.1538},
    }

    @article{larsen2017ase,
      author  = {Larsen, Ask Hjorth and others},
      title   = {The atomic simulation environment --- a {P}ython library for
                 working with atoms},
      journal = {J. Phys.: Condens. Matter},
      year    = {2017},
      volume  = {29},
      pages   = {273002},
      doi     = {10.1088/1361-648X/aa680e},
    }

    @article{henkelman2000neb,
      author  = {Henkelman, Graeme and Uberuaga, Blas P. and J{\'o}nsson, Hannes},
      title   = {A climbing image nudged elastic band method for finding saddle
                 points and minimum energy paths},
      journal = {J. Chem. Phys.},
      year    = {2000},
      volume  = {113},
      pages   = {9901--9904},
      doi     = {10.1063/1.1329672},
    }

    @article{smith2020psi4,
      author  = {Smith, Daniel G. A. and Burns, Lori A. and Simmonett, Andrew C.
                 and others},
      title   = {{PSI4} 1.4: Open-source software for high-throughput quantum
                 chemistry},
      journal = {J. Chem. Phys.},
      year    = {2020},
      volume  = {152},
      pages   = {184108},
      doi     = {10.1063/5.0006002},
    }
