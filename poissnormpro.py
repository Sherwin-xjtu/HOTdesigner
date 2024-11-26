import numpy as np
from scipy.stats import poisson
import pandas as pd


def poissnormpro(informative_probabilities, test_points):
    # Example informativity rates for 5 markers
    # informative_probabilities = [0.9, 0.9, 0.9, 0.8, 0.9,0.9, 0.9, 0.9, 0.8, 0.9]
    
    informative_probabilities = informative_probabilities.tolist()
    test_points = test_points.tolist()
    n = len(informative_probabilities)

    # Step 1: Calculate the expected number of informative markers
    # lambda_value = sum(informative_probabilities)

    # Step 2: Calculate the probability of observing at least 3 informative markers
    # Y_N(3) = 1 - F_N(2), where F_N(2) is the cumulative probability of 2 or fewer markers
    # Poisson approximation
    # test_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    f_t_pro= []
    for p in test_points:
        informative_probabilities.append(p) 
        lambda_value = sum(informative_probabilities)
        prob_at_least = 1 - poisson.cdf(n-1, lambda_value)
        f_t_pro.append(prob_at_least)
    return f_t_pro
        # print(f"Probability of having at least {3} informative markers: {p}: {prob_at_least}")


def simulate_data(n1=10,n=1000):

    hot_reginos_probabilities = np.random.rand(n)

    surrounding_hot_reginos_probabilities = np.random.choice(hot_reginos_probabilities, n1, replace=False)

    candidate_genes_probabilities = np.random.rand(10)

    return surrounding_hot_reginos_probabilities, candidate_genes_probabilities, hot_reginos_probabilities


def pissnormfit(chip_file=None):

    if chip_file is None:
    
        f1, t1, h1 = simulate_data()
        f2, t2, h2 = simulate_data()
        f3, t3, h3 = simulate_data()
        f4, t4, h4 = simulate_data()
        f5, t5, h5 = simulate_data()

    else:
        chip_df = pd.read_csv(chip_file)
        f1 = chip_df['f1'].tolist()
        f2 = chip_df['f2'].tolist()
        f3 = chip_df['f3'].tolist()
        f4 = chip_df['f4'].tolist()
        f5 = chip_df['f5'].tolist()
        t1 = chip_df['t1'].tolist()
        t2 = chip_df['t2'].tolist()
        t3 = chip_df['t3'].tolist()
        t4 = chip_df['t4'].tolist()
        t5 = chip_df['t5'].tolist()
        h1 = chip_df['h1'].tolist()
        h2 = chip_df['h2'].tolist()
        h3 = chip_df['h3'].tolist()
        h4 = chip_df['h4'].tolist()
        h5 = chip_df['h5'].tolist()

    pissnorm_li = []

    h_np = np.array([h1, h2, h3, h4, h5]).T

    for i, (f, t, h) in enumerate(zip([f1, f2, f3, f4, f5], [t1, t2, t3, t4, t5], [h1, h2, h3, h4, h5]), start=1):
        f_t_pro = poissnormpro(f, t)
        pissnorm_li.append(f_t_pro)

    pissnorm_test = np.array(pissnorm_li).T
    return pissnorm_test, h_np

if __name__ == '__main__':
    pissnormfit()