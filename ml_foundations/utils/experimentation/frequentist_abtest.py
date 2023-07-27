import scipy.stats as stats

def calculate_ctr(clicks, impressions):
    return clicks / impressions

def perform_ab_test(clicks_A, impressions_A, clicks_B, impressions_B):
    ctr_A = calculate_ctr(clicks_A, impressions_A) # ctr prob a
    ctr_B = calculate_ctr(clicks_B, impressions_B) # ctr prob b

    # Assuming clicks and impressions follow a binomial distribution
    # Calculate the standard error for each group
    se_A = (ctr_A * (1 - ctr_A) / impressions_A) ** 0.5
    se_B = (ctr_B * (1 - ctr_B) / impressions_B) ** 0.5

    # Calculate the z-score
    z_score = (ctr_B - ctr_A) / ((se_A ** 2 + se_B ** 2) ** 0.5)

    # Define the significance level (alpha) for the test
    alpha = 0.05

    # Calculate the critical z-value for a two-tailed test
    critical_z = stats.norm.ppf(1 - alpha / 2)

    # Check if the z-score is greater than the critical z-value
    if abs(z_score) > critical_z:
        print("There is a statistically significant difference between versions A and B.")
    else:
        print("There is no statistically significant difference between versions A and B.")

if __name__ == "__main__":
    # Sample data for version A
    clicks_A = 100
    impressions_A = 1000

    # Sample data for version B
    clicks_B = 120
    impressions_B = 1000

    # Perform AB test
    perform_ab_test(clicks_A, impressions_A, clicks_B, impressions_B)