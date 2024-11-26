import numpy as np
from scipy.stats import multivariate_normal, chi2
from poissnormpro import pissnormfit

def test_mahalanobis_distance(pissnorm_test, h_np, alpha = 0.05):
    
    
    data = h_np
    # Step 2: Compute mean vector and covariance matrix
    mean_vec = np.mean(data, axis=0)  # Mean vector μ
    cov_matrix = np.cov(data, rowvar=False)  # Covariance matrix Σ

    # To ensure the covariance matrix is positive definite, add a small regularization term
    cov_matrix += 1e-6 * np.eye(cov_matrix.shape[0])
    
    # Step 3: Calculate the threshold for Mahalanobis distance
      # Significance level
    dof = data.shape[1]  # Degrees of freedom equal to the number of features
    chi2_threshold = chi2.ppf(1 - alpha, df=dof)  # Critical value of chi-squared distribution

    # Step 4: Compute Mahalanobis distance for a new point

    for pt in pissnorm_test:

        new_point = pt
        mahalanobis_dist = (new_point - mean_vec).T @ np.linalg.inv(cov_matrix) @ (new_point - mean_vec)

        # Step 5: Determine if the new point belongs to the known data distribution
        if mahalanobis_dist <= chi2_threshold:
            print(f"The new point belongs to the known distribution (Mahalanobis distance squared: {mahalanobis_dist:.4f})")
        else:
            print(f"The new point does not belong to the known distribution (Mahalanobis distance squared: {mahalanobis_dist:.4f})")

        print(f"Threshold (Chi-squared critical value, significance level {alpha}): {chi2_threshold:.4f}")


def MultivariateGaussianModel(pissnorm_test, h_np):

    test_mahalanobis_distance(pissnorm_test, h_np)

def main():

    pissnorm_test, h_np = pissnormfit() 
    MultivariateGaussianModel(pissnorm_test, h_np)



if __name__ == '__main__':
    main()
