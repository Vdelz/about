import numpy as np
import pandas as pd
from scipy.stats import norm
import dowhy
from dowhy import CausalModel


# Define the causal graph in DOT format

causal_graph = """
digraph {
    z3 -> x; z3 -> y;
    x -> z1; y -> z1;
    z1 -> z2;
}
"""


def generate_confounded_data(n_samples=1000, seed=42):
    """
    Generates a synthetic dataset with confounder (z3), 
    collider (z1), and descendant (z2) relationships.
    """
    np.random.seed(seed)
    
    # 1. z3: Discrete binary confounder
    z3 = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
    
    # 2. x: Vectorized probabilistic mapping based on z3
    # If z3 is 0, p(x=1) is 0.3. If z3 is 1, p(x=1) is 0.8.
    probs_x = np.where(z3 == 0, 0.3, 0.8)
    x = np.random.binomial(n=1, p=probs_x)
    
    # 3. y: Vectorized normal distribution based on z3
    # If z3 is 0, mean is 1.0. If z3 is 1, mean is 3.0.
    means_y = np.where(z3 == 0, 1.0, 3.0)
    y = np.random.normal(loc=means_y, scale=0.5)
    
    # 4. z1: Collider (depends on x and y)
    z1 = 0.5 * x + 0.5 * y + np.random.normal(0, 0.1, size=n_samples)
    
    # 5. z2: Descendant of z1
    z2 = z1 + np.random.normal(0, 0.2, size=n_samples)
    
    # Create and return DataFrame
    return pd.DataFrame({
        "z3": z3,
        "x": x,
        "y": y,
        "z1": z1,
        "z2": z2
    })


def calculate_causal_probabilities(df, x_val, y_val, z1_val):
    """
    Calculates P0, P1, P2, and P3 for specific values of x, y, and z1.
    """
    # 1. Estimate Priors and Conditionals from the data
    p_z3_1 = df['z3'].mean()
    p_z3_0 = 1 - p_z3_1
    
    def get_pdf(target_col, cond_val_z3, value):
        # Filter data by z3 and x (if needed) to get local distribution parameters
        subset = df[(df['z3'] == cond_val_z3) & (df['x'] == x_val)]
        return norm.pdf(value, loc=subset[target_col].mean(), scale=subset[target_col].std())
    
    # Helper for P(z1 | y, x, z3) - requires a slice where y is close to y_val
    def get_conditional_pdf_z1_y(z3_val):
        subset = df[(df['z3'] == z3_val) & (df['x'] == x_val)]
        # Simple linear approximation or small window for y
        window = subset[(subset['y'] > y_val - 0.1) & (subset['y'] < y_val + 0.1)]
        return norm.pdf(z1_val, loc=window['z1'].mean(), scale=window['z1'].std())
    
    # --- P0 Calculation ---
    # P(y, z1 | x, z3) * P(z3)
    # We treat y and z1 as independent given x and z3 (based on your generative code)
    term0_z3_0 = get_pdf('y', 0, y_val) * get_pdf('z1', 0, z1_val) * p_z3_0
    term0_z3_1 = get_pdf('y', 1, y_val) * get_pdf('z1', 1, z1_val) * p_z3_1
    p0 = term0_z3_0 + term0_z3_1
    
    # --- P1 Calculation (Chain Rule expansion of P0) ---
    # P(y | z1, x, z3) * P(z1 | x, z3) * P(z3)
    # In your DAG, y depends on z3, not z1, so P(y | z1, x, z3) = P(y | x, z3)
    p1 = p0 
    
    # --- P2 Calculation ---
    # P(z1 | y, x, z3) * P(y | x, z3) * P(z3)
    term2_z3_0 = get_conditional_pdf_z1_y(0) * get_pdf('y', 0, y_val) * p_z3_0
    term2_z3_1 = get_conditional_pdf_z1_y(1) * get_pdf('y', 1, y_val) * p_z3_1
    p2 = term2_z3_0 + term2_z3_1
    
    # --- P3 Calculation (Observational) ---
    # P(y, z1 | x)
    obs_subset = df[df['x'] == x_val]
    p3 = norm.pdf(y_val, obs_subset['y'].mean(), obs_subset['y'].std()) * \
         norm.pdf(z1_val, obs_subset['z1'].mean(), obs_subset['z1'].std())
    
    print(f"Results for x={x_val}, y={y_val:.2f}, z1={z1_val:.2f}:")
    print(f"P0 (Adjustment): {p0:.5f}")
    print(f"P1 (Decomposed): {p1:.5f}")
    print(f"P2 (Inverse Cond): {p2:.5f}")
    print(f"P3 (Observational): {p3:.5f}")


for n_elements in 10**np.arange(3,7):
    print("_"*80)
    print("\nGenerating toy dataset with",n_elements,"elements")
    df = generate_confounded_data(n_elements)
    print("\nP values calculated with hardcoded formulas in numpy")
    calculate_causal_probabilities(df, x_val=1, y_val=3.0, z1_val=2.0)
    
    print("\nP values calculated using dowhy")
    model = CausalModel(
        data=df,
        treatment='x',
        outcome='y',
        graph=causal_graph
    )
    
    # Step 2: Identify the estimand (This is where it finds the P0/P1 formulas)
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print(identified_estimand)
    
    # Step 3: Estimate the effect using the adjustment formula
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    print(f"Causal Effect Estimate: {estimate.value}")

