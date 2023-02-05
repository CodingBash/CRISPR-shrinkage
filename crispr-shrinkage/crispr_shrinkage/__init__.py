from typing import List, Union, Tuple # TODO: Add this to Poetry depency
from scipy.stats import beta, chi
from matplotlib import pyplot as plt # TODO: Add this to Poetry dependency
import numpy as np
import scipy.stats
import scipy.special



class Guide:
    def __init__(self, identifier, position: Union[int, None], pop1_raw_count_reps: List[int], pop2_raw_count_reps: List[int]):
        self.identifier = identifier
        self.position = position
        self.pop1_raw_count_reps = np.asarray(pop1_raw_count_reps)
        self.pop2_raw_count_reps = np.asarray(pop2_raw_count_reps)

class ShrinkageResult:
    def __init__(self,  guide_count_beta_samples_normalized_list: List[List[float]],
            guide_count_LFC_samples_normalized_list: List[List[float]],
            guide_count_posterior_beta_samples_normalized_list: List[List[float]],
            guide_count_posterior_LFC_samples_normalized_list: List[List[float]]):
            self.guide_count_beta_samples_normalized_list=guide_count_beta_samples_normalized_list
            self.guide_count_LFC_samples_normalized_list=guide_count_LFC_samples_normalized_list
            self.guide_count_posterior_beta_samples_normalized_list=guide_count_posterior_beta_samples_normalized_list
            self.guide_count_posterior_LFC_samples_normalized_list=guide_count_posterior_LFC_samples_normalized_list

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
    
    @staticmethod
    def normalize_beta_distribution(posterior_beta_samples, control_beta_samples, baseline=0.5):
        return np.asarray([posterior_beta_samples[i]*(baseline/control_beta_samples[i]) if posterior_beta_samples[i] <= control_beta_samples[i] else 1- ((1-baseline)/(1-control_beta_samples[i]))*(1-posterior_beta_samples[i]) for i, _ in enumerate(list(posterior_beta_samples))])

    @staticmethod
    def calculate_map(posterior_MC_samples: List[float]):
        posterior_MC_samples = np.asarray(posterior_MC_samples)
        n, bins = np.histogram(posterior_MC_samples, bins='sturges')
        bin_idx = np.argmax(n)
        bin_width = bins[1] - bins[0]
        map_estimate = bins[bin_idx] + bin_width / 2
        return map_estimate

def perform_score_shrinkage(each_guide: Guide, negative_control_guide_pop1_total_normalized_counts_reps: List[float], negative_control_guide_pop2_total_normalized_counts_reps: List[float], shrinkage_prior_strength: List[float], unweighted_prior_alpha: List[float], unweighted_prior_beta: List[float], baseline_proportion: float,  monte_carlo_trials: int, random_seed: int, num_replicates: int) -> ShrinkageResult:
    shrinkage_prior_alpha = shrinkage_prior_strength * unweighted_prior_alpha
    shrinkage_prior_beta = shrinkage_prior_strength * unweighted_prior_beta

    #
    # Monte-Carlo sampling of beta distributions (i.e. conjugate priors and posterior distributions)
    #

    # This is for visualization of the prior
    shrinkage_prior_beta_samples_list: List[List[float]] = np.asarray([beta.rvs(shrinkage_prior_alpha[rep_i], shrinkage_prior_beta[rep_i], size=monte_carlo_trials, random_state=random_seed) for rep_i in range(num_replicates)])

    # This is for visualization of the non-influenced data beta distribution
    guide_count_beta_samples_list: List[List[float]] = np.asarray([beta.rvs(each_guide.pop1_normalized_count_reps, each_guide.pop2_normalized_count_reps, size=monte_carlo_trials, random_state=random_seed) for rep_i in range(num_replicates)])

    # This is used for normalization of the non-influced data beta distribution
    control_count_beta_samples_list: List[List[float]] = np.asarray([beta.rvs(negative_control_guide_pop1_total_normalized_counts_reps[rep_i], negative_control_guide_pop2_total_normalized_counts_reps[rep_i], size=monte_carlo_trials, random_state=random_seed) for rep_i in range(num_replicates)])

    # This is the final shrunk posterior
    guide_count_posterior_beta_samples_list: List[List[float]] = np.asarray([beta.rvs(shrinkage_prior_alpha + each_guide.pop1_normalized_count_reps, shrinkage_prior_beta + each_guide.pop2_normalized_count_reps, size=monte_carlo_trials, random_state=random_seed) for rep_i in range(num_replicates)])

    # This is used for normalization
    control_count_posterior_beta_samples_list: List[List[float]] = np.asarray([beta.rvs(shrinkage_prior_alpha + negative_control_guide_pop1_total_normalized_counts_reps[rep_i], shrinkage_prior_beta + negative_control_guide_pop2_total_normalized_counts_reps[rep_i], size=monte_carlo_trials, random_state=random_seed) for rep_i in range(num_replicates)])

    #
    # Normalization of posterior distributions
    #

    # Normalize non-influenced beta samples
    guide_count_beta_samples_normalized_list: List[List[float]] = np.asarray([StatisticalHelperMethods.normalize_beta_distribution(guide_count_beta_samples_list[rep_i], control_count_beta_samples_list[rep_i], baseline_proportion) for rep_i in range(num_replicates)])

    # Normalize the posterior
    guide_count_posterior_beta_samples_normalized_list: List[List[float]] = np.asarray([StatisticalHelperMethods.normalize_beta_distribution(guide_count_posterior_beta_samples_list[rep_i], control_count_posterior_beta_samples_list[rep_i], baseline_proportion) for rep_i in range(num_replicates)])

    guide_count_LFC_samples_normalized_list = np.log(guide_count_beta_samples_normalized_list/baseline_proportion)
    guide_count_posterior_LFC_samples_normalized_list = np.log(guide_count_beta_samples_normalized_list/baseline_proportion)

    # NOTE: When needed, I can add more to this object
    shrinkage_result = ShrinkageResult(guide_count_beta_samples_normalized_list=guide_count_beta_samples_normalized_list,
    guide_count_LFC_samples_normalized_list=guide_count_LFC_samples_normalized_list,
    guide_count_posterior_beta_samples_normalized_list=guide_count_posterior_beta_samples_normalized_list,
    guide_count_posterior_LFC_samples_normalized_list=guide_count_posterior_LFC_samples_normalized_list)

    return shrinkage_result

def optimize_score_shrinkage_prior_strength(guides_for_fit: List[Guide], all_guides: List[Guide], num_replicates: int, negative_control_guide_pop1_total_normalized_counts_reps: List[float], negative_control_guide_pop2_total_normalized_counts_reps: List[float], enable_spatial_prior: bool, spatial_imputation_prior_strength: Union[List[float], None], baseline_proportion: float, spatial_bandwidth: float, monte_carlo_trials: int, random_seed: Union[int, None], max_shrinkage_prior_strength_tested: int = 100, shrinkage_prior_tuning_attempts: int = 100) -> List[float]:
    total_normalized_counts_per_guide_reps : List[List[float]] = [each_guide.pop1_normalized_count_reps + each_guide.pop2_normalized_count_reps for each_guide in guides_for_fit]

    # Get list of prior weight to test
    shrinkage_prior_strength_test_list = np.linspace(0, max_shrinkage_prior_strength_tested, shrinkage_prior_tuning_attempts).repeat(num_replicates).reshape(shrinkage_prior_tuning_attempts, num_replicates)

    BP_statistic_reps_per_test: List[List[float]] = []
    BP_pval_reps_per_test: List[List[float]] = []
    for shrinkage_prior_strength_test in shrinkage_prior_strength_test_list:

        # NOTE: The first list corresponds to each guide, the second list corresponds to number of replicates, the value is the mean LFC
        guide_count_posterior_LFC_normalized_mean_list_per_guide : List[List[float]] = []
        for each_guide in guides_for_fit:

            # TODO: The code for calculating the posterior inputs for the spatial_imputation model could be modularized so that there are not any repetitive code

            # By default, set the unweighted prior as the negative control normalized counts
            unweighted_prior_alpha = negative_control_guide_pop1_total_normalized_counts_reps
            unweighted_prior_beta = negative_control_guide_pop2_total_normalized_counts_reps

            # If able to use spatial information, replace the unweighted priors with the spatial imputational posterior
            if (each_guide.position != None) and enable_spatial_prior: 
                imputation_posterior_alpha, imputation_posterior_beta = perform_score_imputation(each_guide, all_guides, negative_control_guide_pop1_total_normalized_counts_reps, negative_control_guide_pop2_total_normalized_counts_reps, spatial_imputation_prior_strength, num_replicates, spatial_bandwidth)

                # Propogate the imputation posterior to the shrinkage prior
                unweighted_prior_alpha = imputation_posterior_alpha
                unweighted_prior_beta = imputation_posterior_beta


        

            shrinkage_result: ShrinkageResult = perform_score_shrinkage(each_guide, negative_control_guide_pop1_total_normalized_counts_reps, negative_control_guide_pop2_total_normalized_counts_reps, shrinkage_prior_strength_test, unweighted_prior_alpha, unweighted_prior_beta, baseline_proportion,  monte_carlo_trials, random_seed, num_replicates)


            #shrinkage_result.guide_count_beta_samples_normalized_list
            #shrinkage_result.guide_count_LFC_samples_normalized_list
            #shrinkage_result.guide_count_posterior_beta_samples_normalized_list

            # NOTE: List[List[float]], first list is each replicate, second list is the monte-carlo samples. We want the mean of the monte-carlo samples next
            guide_count_posterior_LFC_samples_normalized_list: List[List[float]] = shrinkage_result.guide_count_posterior_LFC_samples_normalized_list 
            
            # This corresponds to the guide posterior mean LFC for each replicate separately for shrinkage prior weight optimization. After optimization of the shrinkage weight, the mean LFC of the averaged posterior of the replicates will be used.
            guide_count_posterior_LFC_normalized_mean_list: List[float] = np.mean(guide_count_posterior_LFC_samples_normalized_list, axis=0) 
            guide_count_posterior_LFC_normalized_mean_list_per_guide.append(guide_count_posterior_LFC_normalized_mean_list)


        # NOTE: Instead of relying on numpy matrix operations to calculate heteroscedasticity, will just iterate through each replicate individually since it may be challenging to do otherwise. Could be a potential optimization in the future to use np matrix operations, but it may not be as readable anyways. (2/4/2023)

        BP_statistic_reps: List[float] = []
        BP_pval_reps: List[float] = []
        for rep_i in range(num_replicates):
            #
            # Calculate the Breusch-Pagan statistic
            #

            # Prepare X - which is the normalized count, since the objective is to reduce hederoscedasticity across count
            total_normalized_count_per_guide_X = [total_normalized_counts_reps[rep_i] for total_normalized_counts_reps in total_normalized_counts_per_guide_reps]

            # Prepare Y - which is the LFC score, since we want to reduce heteroscedasticity of the LFC
            LFC_posterior_mean_per_guide_Y = [guide_count_posterior_LFC_normalized_mean_list[rep_i] for guide_count_posterior_LFC_normalized_mean_list in guide_count_posterior_LFC_normalized_mean_list_per_guide]

            # Regress Y over X - get the intercept and coefficient via OLS
            beta_intercept_ols, beta_coefficient_ols = StatisticalHelperMethods.get_ols_estimators(total_normalized_count_per_guide_X, LFC_posterior_mean_per_guide_Y)

            # Based on the regression estimates, calculate Y_hat
            LFC_posterior_mean_per_guide_Y_hat = StatisticalHelperMethods.calculate_Y_hat(total_normalized_count_per_guide_X, beta_intercept_ols, beta_coefficient_ols)

            # Calculate the squared residuals between Y_hat and Y
            LFC_posterior_mean_per_guide_M_squared_residuals = StatisticalHelperMethods.calculate_squared_residuals(LFC_posterior_mean_per_guide_Y, LFC_posterior_mean_per_guide_Y_hat)

            # Perform a second round of regression of the squared residuals over X.
            beta_intercept_ols_squared_residuals, beta_coefficient_ols_squared_residuals = StatisticalHelperMethods.get_ols_estimators(total_normalized_count_per_guide_X, LFC_posterior_mean_per_guide_M_squared_residuals)

            # Based on the residual regression estimates, calculate residual Y_hat
            LFC_posterior_mean_per_guide_M_squared_residuals_Y_hat = StatisticalHelperMethods.calculate_Y_hat(total_normalized_count_per_guide_X, beta_intercept_ols_squared_residuals, beta_coefficient_ols_squared_residuals)

            # Calculate the model fit R2 coefficient of determination from the residual regression model
            LFC_posterior_mean_per_guide_M_squared_residuals_r_squared = StatisticalHelperMethods.calculate_r_squared(LFC_posterior_mean_per_guide_M_squared_residuals, LFC_posterior_mean_per_guide_M_squared_residuals_Y_hat)

            # Calculate the final Breusch-Pagan chi-squared stastic: BP = n*R2
            LFC_posterior_mean_per_guide_M_BP_statistic = len(total_normalized_count_per_guide_X) * LFC_posterior_mean_per_guide_M_squared_residuals_r_squared

            LFC_posterior_mean_per_guide_M_BP_pval = 1-chi.cdf(LFC_posterior_mean_per_guide_M_BP_statistic, 1) # TODO: Double check if the degree of freedom is correct

            BP_statistic_reps.append(LFC_posterior_mean_per_guide_M_BP_statistic)
            BP_pval_reps.append(LFC_posterior_mean_per_guide_M_BP_pval)
        BP_statistic_reps = np.asarray(BP_statistic_reps)
        BP_pval_reps = np.asarray(BP_pval_reps)

        BP_statistic_reps_per_test.append(BP_statistic_reps)
        BP_pval_reps_per_test.append(BP_pval_reps)
    BP_statistic_reps_per_test = np.asarray(BP_statistic_reps_per_test)
    BP_pval_reps_per_test = np.asarray(BP_pval_reps_per_test)


    shrinkage_prior_strength_selected : List[float] = shrinkage_prior_strength_test_list[np.argmin(BP_statistic_reps_per_test, axis=0)]

    return shrinkage_prior_strength_selected

def perform_score_imputation(each_guide: Guide, all_guides: List[Guide], negative_control_guide_pop1_total_normalized_counts_reps: List[float], negative_control_guide_pop2_total_normalized_counts_reps: List[float], spatial_imputation_prior_strength_test: List[float], num_replicates: int, spatial_bandwidth: float) -> Tuple[List[float], List[float]]:
    # Get Spatial Prior "Likelihood" Counts 


    each_guide_pop1_spatial_contribution_reps: List[float] = np.repeat(0., num_replicates)
    each_guide_pop2_spatial_contribution_reps: List[float] = np.repeat(0., num_replicates)
    
    # Iterate through all neighboring guides
    for neighboring_guide in all_guides:
        if (neighboring_guide.identifier != each_guide.identifier) and (neighboring_guide.position is not None):

            # Along with the weight, is the spatial bandwidth also something that we should optimize as well?
            neighboring_guide_spatial_contribution = StatisticalHelperMethods.gaussian_kernel(neighboring_guide.position, each_guide.position, spatial_bandwidth)

            each_guide_pop1_spatial_contribution_reps = each_guide_pop1_spatial_contribution_reps + (neighboring_guide_spatial_contribution*neighboring_guide.pop1_normalized_count_reps)
            each_guide_pop2_spatial_contribution_reps = each_guide_pop2_spatial_contribution_reps + (neighboring_guide_spatial_contribution*neighboring_guide.pop2_normalized_count_reps)

    pop1_spatial_posterior_alpha = (spatial_imputation_prior_strength_test*negative_control_guide_pop1_total_normalized_counts_reps) + each_guide_pop1_spatial_contribution_reps
    imputation_posterior_alpha = pop1_spatial_posterior_alpha

    pop2_spatial_posterior_beta = (spatial_imputation_prior_strength_test*negative_control_guide_pop2_total_normalized_counts_reps) + each_guide_pop2_spatial_contribution_reps
    imputation_posterior_beta = pop2_spatial_posterior_beta

    return imputation_posterior_alpha, imputation_posterior_beta


                

def optimize_spatial_imputation_prior_strength(all_guides: List[Guide], negative_control_guide_pop1_total_normalized_counts_reps: List[float], negative_control_guide_pop2_total_normalized_counts_reps: List[float], num_replicates: int, spatial_bandwidth: float) -> List[float]:
    #
    #  Set paramaters for what prior weight to test
    #
    max_spatial_imputation_prior_strength_tested = 100 # This should be passed as argument or determined automatically without input
    spatial_imputation_prior_tuning_attempts = 100 # This should be passed as argument or determined automatically based on max prior tested
    
    # Get list of prior weight to test
    spatial_imputation_prior_strength_test_list = np.linspace(0, max_spatial_imputation_prior_strength_tested, spatial_imputation_prior_tuning_attempts).repeat(num_replicates).reshape(spatial_imputation_prior_tuning_attempts, num_replicates)
    
    spatial_imputation_prior_strength_selected: List[float] = []
    KL_guide_imputation_score_total_list : List[List[float]] = []
    
    # Iterate through each prior weight to test
    for spatial_imputation_prior_strength_test in spatial_imputation_prior_strength_test_list:

        # Placeholder variable to hold total sum KL score for each replicate separately
        KL_guide_imputation_score_total: List[float]  = np.repeat(0., num_replicates)

        # Iterate through each guide to test prior with tested weight
        for each_guide in all_guides:

            # Ensure that the guide contains a position
            if each_guide.position is not None:

                # Get the posterior
                imputation_posterior_alpha, imputation_posterior_beta = perform_score_imputation(each_guide, all_guides, negative_control_guide_pop1_total_normalized_counts_reps, negative_control_guide_pop2_total_normalized_counts_reps, spatial_imputation_prior_strength_test, num_replicates, spatial_bandwidth)

                true_alpha = each_guide.pop1_normalized_count_reps
                true_beta = each_guide.pop2_normalized_count_reps

                # Calculate KL divergence between the posterior and the likelihood
                KL_guide_imputation_score: List[float] = StatisticalHelperMethods.KL_beta(true_alpha, true_beta, imputation_posterior_alpha, imputation_posterior_beta) 

                # Add score to the main placeholder to get the final sum
                KL_guide_imputation_score_total = KL_guide_imputation_score_total + KL_guide_imputation_score 
                # TODO: I feel that this will implicitly place weight on null guides (since the majority will be null), so I wonder if there is a way to have the score prioritize guides that deviate from the null - think about how "deviation" from the null is defined quantitatively, and how this quantitation would be integrated into the heuristic.

        KL_guide_imputation_score_total_list.append(KL_guide_imputation_score_total)
    KL_guide_imputation_score_total_list = np.asarray(KL_guide_imputation_score_total_list)
    spatial_imputation_prior_strength_selected = spatial_imputation_prior_strength_test_list[np.argmin(KL_guide_imputation_score_total_list, axis=0)]
    return spatial_imputation_prior_strength_selected




def perform_adjustment(
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
    spatial_imputation_prior_strength: Union[List[float], None] = None, # This could be optimized by maximizing correlation of guide with neighborhood (perhaps in binomial GLM fashion?),
    baseline_proportion: float = 0.5, # TODO: Perform validation between (0,1), also accept None value for perfrming no normalization (or have that be another argument)
    shrinkage_prior_strength: Union[List[float], None] = None, 
    posterior_estimator: str = "mean",
    random_seed: Union[int, None] = None
    ):
    
    # Validation
    assert posterior_estimator.upper() in ["MEAN", "MODE"], "Posterior estimator must be of value 'mean' or 'mode'"
    assert monte_carlo_trials>0, "Monte-Carlo trial amout must be greater than 0"
    assert num_replicates>0, "Number of replicates specified must be greater than 0"
    assert spatial_bandwidth>0, "Spatial prior bandwidth used for Gaussian kernel must be greater than 0"

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
    # TODO: I need to find a way to ensure that the fitted prior will work for guides that don't have position, i.e. I assume those without position will not have a prior based on the spatial information, but will the shrinkage weight of the imputation posterior be difference? 
    if shrinkage_prior_strength is None:
        # NOTE: Here, we will be using selected guides for fit since the heteroscedasticity statistic may be biased towards positive effect guides.

        shrinkage_prior_strength = optimize_score_shrinkage_prior_strength(guides_for_fit, all_guides, num_replicates, negative_control_guide_pop1_total_normalized_counts_reps, negative_control_guide_pop2_total_normalized_counts_reps, enable_spatial_prior, spatial_imputation_prior_strength, baseline_proportion, spatial_bandwidth, monte_carlo_trials, random_seed, max_shrinkage_prior_strength_tested = 100, shrinkage_prior_tuning_attempts = 100)

        # LEFTOFF: Just finished draft of selecting shrinkage prior weight. Next, move to function and proceed with calling the selection to run final model to get final posterior, then get the averaged posterior, calculating the MAP (or mean), and return. id probably say the mean is best if there is equal trust in all replicates, if there is not, then mode since any outlier density in the averaged posterior (due to the outlier replicate) will not influence the mode as much. It could be a nice quality control metric to show the concordance of each replicates posterior distribution.l

    # Perform final model inference:
    for each_guide in all_guides:
        # TODO: The code for calculating the posterior inputs for the spatial_imputation model could be modularized so that there are not any repetitive code

        # By default, set the unweighted prior as the negative control normalized counts
        unweighted_prior_alpha = negative_control_guide_pop1_total_normalized_counts_reps
        unweighted_prior_beta = negative_control_guide_pop2_total_normalized_counts_reps

        # If able to use spatial information, replace the unweighted priors with the spatial imputational posterior
        if (each_guide.position != None) and enable_spatial_prior: 
            imputation_posterior_alpha, imputation_posterior_beta = perform_score_imputation(each_guide, all_guides, negative_control_guide_pop1_total_normalized_counts_reps, negative_control_guide_pop2_total_normalized_counts_reps, spatial_imputation_prior_strength, num_replicates, spatial_bandwidth)

            # Propogate the imputation posterior to the shrinkage prior
            unweighted_prior_alpha = imputation_posterior_alpha
            unweighted_prior_beta = imputation_posterior_beta


        shrinkage_result: ShrinkageResult = perform_score_shrinkage(each_guide, negative_control_guide_pop1_total_normalized_counts_reps, negative_control_guide_pop2_total_normalized_counts_reps, shrinkage_prior_strength, unweighted_prior_alpha, unweighted_prior_beta, baseline_proportion,  monte_carlo_trials, random_seed, num_replicates)


        # NOTE: List[List[float]], first list is each replicate, second list is the monte-carlo samples. We want the mean of the monte-carlo samples next
        guide_count_posterior_LFC_samples_normalized_list: List[List[float]] = shrinkage_result.guide_count_posterior_LFC_samples_normalized_list 
        
        guide_count_posterior_LFC_samples_normalized_average = np.mean(guide_count_posterior_LFC_samples_normalized_list, axis=0)

        final_LFC_estimate = None
        if posterior_estimator.upper() == "MEAN":
            final_LFC_estimate = np.mean(guide_count_posterior_LFC_samples_normalized_average)
        elif posterior_estimator.upper() == "MODE":
            final_LFC_estimate = StatisticalHelperMethods.calculate_map(guide_count_posterior_LFC_samples_normalized_average)

        each_guide.final_LFC_estimate = final_LFC_estimate
    
    # TODO: Produce better way to return results, i.e. new results object that has the negative, observational, and positive controls in different values
    return all_guides




















