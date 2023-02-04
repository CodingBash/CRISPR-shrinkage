from typing import List, Union # TODO: Add this to Poetry depency
from scipy.stats import beta, chi
from matplotlib import pyplot as plt # TODO: Add this to Poetry dependency
import numpy as np


class Guide:
    def __init__(self, identifier, position: Union[int, None], pop1_raw_count_reps: List[int], pop2_raw_count_reps: List[int]):
        self.identifier = identifier
        self.position = position
        self.pop1_raw_count_reps = np.asarray(pop1_raw_count_reps)
        self.pop2_raw_count_reps = np.asarray(pop2_raw_count_reps)

class StatisticalHelperMethods:
    @staticmethod
    def get_ols_estimators(X, Y):
        X_np = np.asarray(X)
        Y_np = np.asarray(Y)

        X_mean = np.mean(X_np)
        Y_mean = np.mean(Y_np)

        beta_coefficient_ols = np.sum((X_np-X_mean)*(Y_np-Y_mean))/(np.sum((X_np-X_mean)**2))
        beta_intercept_ols = Y_mean - (beta_coefficient_ols*X_mean)
        return beta_intercept_ols, beta_coefficient_ols

    @staticmethod
    def calculate_Y_hat(X, beta_intercept, beta_coefficient):
        X_np = np.asarray(X)

        Y_hat = beta_intercept + (X_np*beta_coefficient)
        return Y_hat

    @staticmethod
    def calculate_squared_residuals(Y, Y_hat):
        Y_np = np.asarray(Y)
        Y_hat_np = np.asarray(Y_hat)

        return (Y_np-Y_hat_np)**2

    @staticmethod
    def calculate_r_squared(Y, Y_hat):
        Y_np = np.asarray(Y)
        Y_hat_np = np.asarray(Y_hat)

        r_squared = 1-((np.sum((Y_np-Y_hat_np)**2))/(np.sum((Y_np-np.mean(Y_hat))**2)))

        return r_squared
    
    @staticmethod
    def gaussian_kernel(range, point, bandwidth): 
        return np.exp(-(range-point)**2/(2*bandwidth**2))/(bandwidth*np.sqrt(2*np.pi))

    @staticmethod
    def KL_beta(alpha_f, beta_f, alpha_g, beta_g):
        return np.log(scipy.special.beta(alpha_g, beta_g)/(scipy.special.beta(alpha_f, beta_f))) + ((alpha_f - alpha_g)*(scipy.special.digamma(alpha_f) - scipy.special.digamma(alpha_f+beta_f))) + ((beta_f - beta_g)*(scipy.special.digamma(beta_f) - scipy.special.digamma(alpha_f+beta_f)))

def optimize_score_shrinkage_prior_strength():
    pass # TODO: Add here


def optimize_spatial_imputation_prior_strength(all_guides: List[Guide], negative_control_guide_pop1_total_normalized_counts_reps: List[float], negative_control_guide_pop2_total_normalized_counts_reps: List[float], num_replicates: int, spatial_bandwidth: float) -> List[float]:
    max_spatial_imputation_prior_strength_tested = 100 # This should be passed as argument or determined automatically without input
    spatial_imputation_prior_tuning_attempts = 100 # This should be passed as argument or determined automatically based on max prior tested
    spatial_imputation_prior_strength_test_list = np.linspace(0, max_spatial_imputation_prior_strength_tested, spatial_imputation_prior_tuning_attempts)
    
    spatial_imputation_prior_strength: List[float] = []
    for rep_i in range(num_replicates):
        KL_guide_imputation_score_total_list : List[float] = []
        for spatial_imputation_prior_strength_test in spatial_imputation_prior_strength_test_list:
            KL_guide_imputation_score_total: float  = 0.
            for each_guide in all_guides:
                if each_guide.position is not None:
                    each_guide_pop1_spatial_contribution: float = 0.
                    each_guide_pop2_spatial_contribution: float = 0.
                    for neighboring_guide in all_guides: 
                        if (neighboring_guide.identifier != each_guide.identifier) and (neighboring_guide.position is not None):
                            neighboring_guide_spatial_contribution = StatisticalHelperMethods.gaussian_kernel(neighboring_guide.position, each_guide.position, spatial_bandwidth)

                            each_guide_pop1_spatial_contribution = each_guide_pop1_spatial_contribution + (neighboring_guide_spatial_contribution*neighboring_guide.pop1_normalized_count_reps[rep_i])

                            each_guide_pop2_spatial_contribution = each_guide_pop2_spatial_contribution + (neighboring_guide_spatial_contribution*neighboring_guide.pop2_normalized_count_reps[rep_i])

                    pop1_spatial_posterior_alpha = (spatial_imputation_prior_strength_test*negative_control_guide_pop1_total_normalized_counts_reps[rep_i]) + each_guide_pop1_spatial_contribution
                    
                    pop2_spatial_posterior_beta = (spatial_imputation_prior_strength_test*negative_control_guide_pop2_total_normalized_counts_reps[rep_i]) + each_guide_pop2_spatial_contribution

                    true_alpha = each_guide.pop1_normalized_count_reps[rep_i]
                    true_beta = each_guide.pop2_normalized_count_reps[rep_i]

                    KL_guide_imputation_score = StatisticalHelperMethods.KL_beta(true_alpha, true_beta, pop1_spatial_posterior_alpha, pop2_spatial_posterior_beta)
                    KL_guide_imputation_score_total = KL_guide_imputation_score_total + KL_guide_imputation_score 
                    # TODO: I feel that this will implicitly place weight on null guides (since the majority will be null), so I wonder if there is a way to have the score prioritize guides that deviate from the null - think about how "deviation" from the null is defined quantitatively, and how this quantitation would be integrated into the heuristic.
            KL_guide_imputation_score_total_list.append(KL_guide_imputation_score_total)
        spatial_imputation_prior_strength_selected = spatial_imputation_prior_strength_test_list[np.argmin(KL_guide_imputation_score_total_list)]
        spatial_imputation_prior_strength.append(spatial_imputation_prior_strength_selected)
    spatial_imputation_prior_strength = np.asarray(spatial_imputation_prior_strength)
    return spatial_imputation_prior_strength

def perform_shrinkage(
    negative_control_guides: List[Guide],
    positive_control_guides: List[Guide],
    observation_guides: List[Guide],
    num_replicates: int,
    include_observational_guides_in_fit: bool = True,
    include_positive_control_guides_in_fit: bool = False,
    pop1_amplification_factors: List[float] = None,
    pop2_amplification_factors: List[float] = None,
    monte_carlo_trials: int = 1000,
    enable_spatial_prior: bool = False,
    spatial_bandwidth: int = 1,
    spatial_imputation_prior_strength: Union[List[float], None] = None, # This could be optimized by maximizing correlation of guide with neighborhood (perhaps in binomial GLM fashion?)
    shrinkage_prior_strength: Union[List[float], None] = None, 
    random_seed: Union[int, None] = None
    ):
    
    # Validation
    assert monte_carlo_trials>0, "Monte-Carlo trial amout must be greater than 0"
    assert num_replicates>0, "Number of replicates specified must be greater than 0"
    assert spatial_prior_bandwidth>0, "Spatial prior bandwidth used for Gaussian kernel must be greater than 0"

    for guide in negative_control_guides:
        assert num_replicates == len(guide.pop1_raw_count_reps) == len(guide.pop2_raw_count_reps), "Guide {} number of counts does not equal replicates"
    for guide in observation_guides:
        assert num_replicates == len(guide.pop1_raw_count_reps) == len(guide.pop2_raw_count_reps), "Guide {} number of counts does not equal replicates"
    for guide in observation_guides:
        assert num_replicates == len(guide.pop1_raw_count_reps) == len(guide.pop2_raw_count_reps), "Guide {} number of counts does not equal replicates"


    # Set the amplification factors
    pop1_amplification_factors = np.repeat(1.,num_replicates) if pop1_amplification_factors == None else np.asarray(pop1_amplification_factors)
    pop2_amplification_factors = np.repeat(1.,num_replicates) if pop2_amplification_factors == None else np.asarray(pop2_amplification_factors)
    
    assert len(pop1_amplification_factors) == num_replicates, "Number of population 1 amplification factors does not equal replicates, instead is {}".format(len(pop1_amplification_factors))
    assert len(pop2_amplification_factors) == num_replicates, "Number of population 2 amplification factors does not equal replicates, instead is {}".format(len(pop2_amplification_factors))

    # Normalize the guide count
    def normalize_guide_counts(guide_list: List[Guide], pop1_amplification_factors, pop2_amplification_factors):
        for guide in guide_list:
            guide.pop1_normalized_count_reps = guide.pop1_raw_count_reps/pop1_amplification_factors
            guide.pop2_normalized_count_reps = guide.pop2_raw_count_reps/pop2_amplification_factors

    normalize_guide_counts(negative_control_guides, pop1_amplification_factors, pop2_amplification_factors)
    normalize_guide_counts(positive_control_guides, pop1_amplification_factors, pop2_amplification_factors)
    normalize_guide_counts(observation_guides, pop1_amplification_factors, pop2_amplification_factors)
    

    if spatial_imputation_prior_strength != None:
        spatial_imputation_prior_strength = np.asarray(spatial_imputation_prior_strength)
        assert len(spatial_imputation_prior_strength) == num_replicates, "Number of spatial imputation prior strength values in list must equal number of replicates"
    if shrinkage_prior_strength != None:
        shrinkage_prior_strength = np.asarray(shrinkage_prior_strength)
        assert len(shrinkage_prior_strength) == num_replicates, "Number of shrinkage prior strength values in list must equal number of replicates"

    # Set the guides used for shrinkage model fitting
    guides_for_fit: List[Guide] = negative_control_guides
    guides_for_fit = np.concatenate([guides_for_fit, observation_guides]) if include_observational_guides_in_fit else guides_for_fit
    guides_for_fit = np.concatenate([guides_for_fit, positive_control_guides]) if include_positive_control_guides_in_fit else guides_for_fit
    assert len(guides_for_fit) > 0, "Total guides used for shrinkage model fitting is zero"

    # Create all guides set used for informing neighborhood prior, performing final shrinkage, and visualization    
    all_guides: List[Guide] = np.concatenate([negative_control_guides, observation_guides, positive_control_guides])


    # Get total normalized counts of negative controls in both populations to be used as initial prior
    negative_control_guide_pop1_total_normalized_counts_reps: List[int] = np.repeat(0., num_replicates)
    negative_control_guide_pop2_total_normalized_counts_reps: List[int] = np.repeat(0., num_replicates)
    negative_control_guide: Guide
    for negative_control_guide in negative_control_guides:
            negative_control_guide_pop1_total_normalized_counts_reps = negative_control_guide_pop1_total_normalized_counts_reps + negative_control_guide.pop1_normalized_count_reps
            negative_control_guide_pop2_total_normalized_counts_reps = negative_control_guide_pop2_total_normalized_counts_reps + negative_control_guide.pop2_normalized_count_reps

    #
    # Identification of optimal spatial_imputation_prior_strength 
    #
    ## TODO: Here we identify the optimal spatial_imputation_prior weights - this can be in its own function for readibility
    if spatial_imputation_prior_strength is None and enable_spatial_prior:
            # No spatial_control_prior_strength was provided. We will need to optimize by assessing the correlation of the posterior to the actual guide. 
            # Notes: We will measure correlation somehow via a heuristic, starting with calculating the KL divergence between the beta distribution of the prior and from the counts. But how will we weight KL scores from guides with high counts compared to low counts? This is why I liked implementing the binomial likelihood somehow.
            
            spatial_imputation_prior_strength = optimize_spatial_imputation_prior_strength(all_guides, negative_control_guide_pop1_total_normalized_counts_reps, negative_control_guide_pop2_total_normalized_counts_reps, num_replicates, spatial_bandwidth)



    #
    # Identification of optimal shrinkage_prior_strength
    #
    if shrinkage_prior_strength is None:
        for each_guide in all_guides:

            # TODO: The code for calculating the posterior inputs for the spatial_imputation model could be modularized so that there are not any repetitive code

            prior_alpha = negative_control_guide_pop1_total_normalized_counts_reps
            prior_beta = negative_control_guide_pop2_total_normalized_counts_reps
            # Perform shrinkage utilizing spatial information
            if (each_guide.position != None) and enable_spatial_prior: # TODO: Consider the non enable_spatial_prior case. 
                
                # Get Spatial Prior "Likelihood" Counts 
                each_guide_pop1_spatial_contribution_reps: List[int] = np.repeat(0., num_replicates)
                each_guide_pop2_spatial_contribution_reps: List[int] = np.repeat(0., num_replicates)
                for neighboring_guide in all_guides:
                    if (neighboring_guide.identifier != each_guide.identifier) and (neighboring_guide.position is not None):
                        neighboring_guide_spatial_contribution = StatisticalHelperMethods.gaussian_kernel(neighboring_guide.position, each_guide.position, spatial_bandwidth)

                        each_guide_pop1_spatial_contribution_reps = each_guide_pop1_spatial_contribution_reps + (neighboring_guide_spatial_contribution*neighboring_guide.pop1_normalized_count_reps)
                        each_guide_pop2_spatial_contribution_reps = each_guide_pop2_spatial_contribution_reps + (neighboring_guide_spatial_contribution*neighboring_guide.pop2_normalized_count_reps)

                spatially_informed_posterior_samples_reps: Union[List[List[float]], None] = None 
                

                pop1_spatial_posterior_alpha = (spatial_imputation_prior_strength_test*negative_control_guide_pop1_total_normalized_counts_reps) + each_guide_pop1_spatial_contribution
                prior_alpha = pop1_spatial_posterior_alpha

                pop2_spatial_posterior_beta = (spatial_imputation_prior_strength_test*negative_control_guide_pop2_total_normalized_counts_reps) + each_guide_pop2_spatial_contribution
                prior_beta = pop2_spatial_posterior_beta
                
            shrinkage_prior_alpha = shrinkage_prior_strength * prior_alpha
            shrinkage_prior_beta = shrinkage_prior_strength * prior_beta

            # TODO: LEFTOFF OFF HERE - Just finished code for optimizing imputational stage, now starting shrinkage stage, which is just translating the code from the notebook to here. The two code lines above is the priors, now continue (by seeing shrink_LFC_from_counts)


                pass
            else:
                pass


        # Perform final fit

