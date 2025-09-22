# rrp4822_Riddhi_ML_Lab2
# Neural Data Analysis: Predicting Monkey Hand Velocity

This project explores the relationship between neural activity in the primary motor cortex (M1) and hand movement. Using a dataset from a monkey performing reaching tasks, we build linear regression models to predict hand velocity from recorded neural spike counts. The lab progresses from a simple "memoryless" model to a more complex time-delayed model (an FIR filter), and finally uses model order selection to find the optimal delay for the best predictive performance.

This lab was adapted from Prof. Christopher Musco's CS-GY 6923 course.



## Core Concepts

* **Linear Regression:** Using the normal equation to find the optimal model coefficients.
* **Time-Series Analysis:** Modeling systems where past inputs influence present outputs.
* **Feature Engineering:** Creating a new feature matrix (`Xdly`) that incorporates time-delayed neural signals.
* **Model Order Selection:** Systematically testing model complexity (in this case, the time delay `d`) to find the one that generalizes best to unseen data.
* **Train-Test Split:** Splitting data to evaluate model performance on unseen samples.

## Methodology

The analysis was performed in three main stages:

### 1. Memoryless Linear Model

First, a simple linear model was created to establish a baseline. This model predicts the hand velocity at time `i` using only the neural spike counts from that same instant, `i`.

The model form is:
$$ \hat{y}_i = \beta_0 + \sum_{j=1}^{p} X_{i,j}\beta_j $$

* **Implementation:** A custom function was written to solve for the `beta` coefficients using the normal equation: $\beta = (X^T X)^{-1} X^T y$.
* **Result:** This model achieved a Mean Squared Error (MSE) of approximately **32-33** on the test set. The performance was limited, as it fails to capture the inherent delay between neural firing and physical movement.

### 2. Linear Model with Time Delays (FIR Filter)

To improve the model, we incorporated past neural activity. The hypothesis is that the velocity at a given time is influenced by the neural signals from several preceding moments. A new feature matrix was engineered where each row contains the concatenated neural data from the current time step `i` back to `i-d`.

The model form is:
$$ \hat{y}_{i+d} = \sum_{m=0}^d \sum_{j=1}^{p} X_{i+m,j}\beta_{j,m} $$

* **Implementation:** A `create_dly_data` function was built to transform the raw time-series data into this new format.
* **Result (with fixed `d=6`):** This model showed a **significant improvement**, with the MSE dropping to approximately **18.8**. This confirmed that incorporating past neural activity is crucial for accurate predictions.

### 3. Optimal Delay Selection

While `d=6` was better, it was an arbitrary choice. The final step was to find the optimal delay `d` by testing a range of values from 0 to 29.

* **Implementation:** To do this efficiently, a single "master" dataset was created with the maximum delay (`dmax=30`). A loop then tested each delay value by training a model on a progressively wider **slice** of the feature matrix. This avoids recreating the dataset in every iteration, saving significant computation time.
* **Result:** The optimal delay was found by identifying the `d` value that resulted in the lowest MSE on the test set. The plot below shows how the model's error decreases as more delays are added, eventually plateauing.


The optimal delay was found to be **d=29**, which achieved a minimum MSE of approximately **19.9**.

## How to Run

1.  **Dependencies:** Ensure you have Python installed with the following libraries:
    * `numpy`
    * `matplotlib`
    * `scikit-learn`

2.  **Data:** The script will automatically download the required data file, `example_data_s1.pickle`, if it's not already present.

3.  **Execution:** The analysis is contained within a single script or Jupyter Notebook. Run the code from top to bottom to reproduce the data loading, model fitting, and final analysis.

## Conclusion

This project successfully demonstrates that a monkey's hand velocity can be predicted from M1 neural signals. It also clearly shows that accounting for the time delay between neural commands and physical action is critical for building an accurate predictive model. The process of model order selection provides a data-driven way to determine the optimal memory capacity for such a system.
