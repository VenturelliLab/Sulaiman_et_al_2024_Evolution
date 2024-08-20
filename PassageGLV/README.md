# PassageGLV

Code for fitting a General Lotka-Volterra (GLV) model to long-term passging experiments.

Original Hardware Specifictions:
* OS: Microsoft Windows 11 Pro (v10.0.22631)
* Processor: AMD Ryzen 9 7950X 16-Core Processor
* RAM: 128 GB

Expected Run Time for Model Fitting (9 Processes): 240-300 Minutes

Run time may need to be extended for model fitting as it is based on a time limit.

Cross validation may take up to a day (high throughput computing servers) or up to two weeks depending on setup.

It is expected that the model fitting process can take up to a week or more for computers with worse specifications.

Ideally, each trial run (distributed) should be run separately since the timers (each run is timed) may overlap, which may end prematurely. An easy solution is to rerun many times and filter the ones that failed.

## Installation Guide and Usage with included files

1. Download [Julia](https://julialang.org/downloads/) (v1.10.1)
    * Recommended: Use [Visual Studio Code](https://code.visualstudio.com/docs/languages/julia) (VS Code) for running Julia for this for an interactive experience. Click the link to learn how to get started with Julia in VS Code.
    * Not Recommended: Open Windows terminal and type "julia \<filename\>".
    * Not Recommended: Open the Julia REPL in that directory and type include("\<filename\>") to run the .jl files.

2. Install the required packages using **install_packages.jl**
3. Run **fit_data_NL_opt_dist_passage_constr.jl** to get parameters. If needed, run **fit_LOOCV.jl** for the cross validation (may need to change some lines for different setups). **fit_pair_triple** only uses pairs or triples of species.
    * Change the number of processes "addprocs" according to your hardware
    * Parameters are stored in GLV_pars_passage_constr folder
4. Run **param_analysis_passage_contr.jl** to get parameter CSV file and associated plot or SVG file
5. Run **passage_constr_analysis.jl** to get the predictions CSV file. Similar for **LOOCV_analysis.jl** and **pair_triple_analysis.jl**.
6. Run **sample_simulations_passage_constr.jl** to see some simulations of the model

## Installation Guide and Usage with alternative files

1. Same as steps 1-2 with the original files
2. Format your csv in the same way as the original
3. Change get_data.jl for your data
4. Change the time-related variables in other files according to your experiments
5. Same as steps 4-6 with the original files (change as needed)

## Required Packages

- LinearAlgebra.jl (Included with Julia)
- Random.jl (Included with Julia)
- Statistics.jl (v1.10.0)
- HypothesisTests.jl (v0.11.0)
- CSV.jl (v0.10.14)
- DataFrames.jl (v1.6.1)
- JLD2.jl (v0.4.48)
- PyCall.jl (v1.96.4)
- PyPlot.jl (v2.11.2)
- ColorSchemes.jl (v3.25.0)
- Colors.jl (v0.12.11)
- Plots.jl (v1.40.4)
- Optimization.jl (v3.24.3)
- OptimizationNLopt.jl (v0.2.2)

## Included Files and Folders (This includes the expected output)

### GLV_pars_assage_constr
This is the folder containing all the model parameters.

### GLV_pars_passage_constr_LOOCV
This is the folder containing all the model parameters for leave-one-out cross validation.

### GLV_pars_pair
This is the folder containing all the model parameters for a model trained only on pairwise data (note: not all possible pairwise was used).

### GLV_pars_triple
This is the folder containing all the model parameters for a model trained only on triple species data (note: not all possible pairwise was used).

### EXP0004_35passages_formodelfitting.csv
This is the passage data file.

### EXP0004_CV_analysis.csv
This is the CV data file.

### EXP0004_pair_analysis.cdsv
This is the data file using a model trained only on pairwise communities.

### EXP0004_triple_analysis.csv
This is the data file extracted using only a model trained on triple species communities.

### EXP0004_pair_analysis_params.csv
This is the file containing parameters extracted from pairwise communities.

### EXP0004_triple_analysis_params.csv
This is the file containing parameters extracted from triple species communities.

### EXP0004_P_passage_constr.svg
This is the parameter plot file.

### EXP0004_P_passage_constr.csv
This is the parameter file.

### EXP0004_passage_constr.csv
This is the model fitting results file.

### get_data.jl
This file extracts the passage data from their corresponding CSV files.

### GLV.jl
This file includes the GLV model, passage simulation, and the objective function.

### fit_data_NL_opt_dist_passage_constr.jl
This file fits the GLV model using the data obtained from get_data.jl in a distributed manner.

### fit_LOOCV.jl
This file contains a modified version of the original code to obtain models in parallel.

### fit_pair_triple.jl
This file fits using either only pairwise or triple species.

### passage_constr_analysis.jl
This file computes the prediction results.

### LOOCV_analysis.jl
This file compute the cross-validation prediction results.

### pair_triple_analysis.jl
This file is similar to LOOCV_analysis.jl except we validate pair models on triple species data, and vice versa.

### param_analysis_passage_contr.jl
This file analyzes the parameters of the GLV model for models that were successfully optimized.

### sample_simulations_passage_constr.jl
This file gives example simulations of the GLV model.

## Citations (Package Provided Ones Only)
Julia (LinearAlgebra.jl, Random.jl, Statistics.jl):
- Bezanson, J., Edelman, A., Karpinski, S., & Shah, V. B. (2017). Julia: A fresh approach to numerical computing. SIAM review, 59(1), 65-98.

DataFrames.jl:
- Bouchet-Valat, M., & Kamiński, B. (2023). DataFrames. jl: flexible and fast tabular data in Julia. Journal of Statistical Software, 107, 1-32.

Plots.jl:
- Tom Breloff. (2024). Plots.jl (v1.40.4). Zenodo. https://doi.org/10.5281/zenodo.10959005

Optimization.jl:
- Vaibhav Kumar Dixit, & Christopher Rackauckas. (2023). Optimization.jl: A Unified Optimization Package (v3.12.1). Zenodo. https://doi.org/10.5281/zenodo.7738525

OptimizationNLopt.jl (NLopt)
- Johnson, S. G. (2014). The NLopt nonlinear-optimization package.
