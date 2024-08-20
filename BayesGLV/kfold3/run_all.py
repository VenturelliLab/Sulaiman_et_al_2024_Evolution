import subprocess

# loop over strain
strains = ['CDevo', 'CDanc']
for strain in strains:

    # Loop over k values from 0 to 19
    for k in range(20):
        # Run the script with the current value of k
        subprocess.run(['python', 'Kfold_gLV3_Fold.py', strain, str(k)])
