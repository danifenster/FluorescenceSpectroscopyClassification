# FluorescenceSpectroscopyClassification

### Data Directory/Collection:
https://www.kaggle.com/datasets/edumisvieramartin/multi-wavelength-for-metal-classification?resource=download

### Usage:

1.  Ensure that the `wavelength.csv` file is in the same directory as the `main.py` script.
2.  Run the `main.py` script from the command line:

    ```
    python main.py
    ```

3.  The script will:
    *   Load the data from `wavelength.csv`.
    *   Train an MLP classifier model.
    *   Evaluate the model performance using different combinations of optimizers and activation functions.
    *   Print the best performing optimizer and activation function combination, with its average accuracy.
    *   Prompt the user to enter red, green, and blue channel wavelength values for a material.
    *   Predict the material class based on the input wavelengths.

### Input:

The script requires user input for the red, green, and blue channel wavelength values (at 8mM concentration) to predict the material class. The user will be prompted to enter these values after the model training and evaluation are complete.

### Output:

The script provides the following output:

*   The best combination of optimizer and activation function that yields the highest R-squared score.
*   The predicted material class for the user-provided input wavelengths.
*   The average accuracy across the R-squared Scores

### Author:

Daniella Fenster
