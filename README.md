# GIANO-CT

This is code to perform continuum normalization and telluric correction for GIANO spectra.

## Usage

1. Install `alpha-continuum`
    - Clone https://github.com/MingjieJian/alpha_continuum and pip install.
2. Install other requirement packages
    - numpy, pandas, matplotlib, tqdm, astropy, telfit, scipy, spectres
3. Modify `batch_run.sh` to the folder containing GIANO spectra (for now it is desiged for the SPA data)
4. Run `batch_sun.sh`