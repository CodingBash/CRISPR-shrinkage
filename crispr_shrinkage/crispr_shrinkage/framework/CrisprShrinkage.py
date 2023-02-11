#!/usr/bin/env python
from typing import List, Union, Tuple
from scipy.stats import beta, chi
from matplotlib import pyplot as plt # TODO: Add this to Poetry dependency
import numpy as np
import scipy.stats
import scipy.special as sc
import scipy.optimize as so
from decimal import *
import math
import functools
import copy
import logging
import sys


class Guide:
    def __init__(self, identifier, position: Union[int, None], pop1_raw_count_reps: List[int], pop2_raw_count_reps: List[int]):
        assert len(pop1_raw_count_reps) == len(pop2_raw_count_reps), "Counts for two populations must be same length"
        self.identifier = identifier
        self.position = position
        self.pop1_raw_count_reps = np.asarray(pop1_raw_count_reps)
        self.pop2_raw_count_reps = np.asarray(pop2_raw_count_reps)

class ExperimentGuideSets:
    def __init__(self, negative_control_guides: List[Guide], positive_control_guides: List[Guide], observation_guides: List[Guide]):
        self.negative_control_guides = negative_control_guides
        self.positive_control_guides = positive_control_guides
        self.observation_guides = observation_guides

class CrisprShrinkageResult:
    def __init__(self, adjusted_negative_control_guides: List[Guide],
            adjusted_observation_guides:List[Guide],
            adjusted_positive_control_guides:List[Guide],
            shrinkage_prior_strength:Union[List[float],None],
            spatial_imputation_model_weights:Tuple[Union[List[float],None],Union[List[float],None]],
            raw_negative_control_guides: List[Guide],
            raw_positive_control_guides: List[Guide],
            raw_observation_guides: List[Guide],
            num_replicates:int,
            include_observational_guides_in_fit: bool,
            include_positive_control_guides_in_fit:bool,
            pop1_amplification_factors: List[float],
            pop2_amplification_factors: List[float],
            monte_carlo_trials: int,
            enable_spatial_prior: bool,
            spatial_bandwidth: int,
            baseline_proportion: float, # TODO: Perform validation between (0,1), also accept None value for perfrming no normalization (or have that be another argument)
            posterior_estimator: str,
            random_seed: Union[int, None]):
        self.adjusted_negative_control_guides=adjusted_negative_control_guides
        self.adjusted_observation_guides=adjusted_observation_guides
        self.adjusted_positive_control_guides=adjusted_positive_control_guides
        self.shrinkage_prior_strength=shrinkage_prior_strength
        self.spatial_imputation_model_weights=spatial_imputation_model_weights
        self.raw_negative_control_guides=raw_negative_control_guides
        self.raw_positive_control_guides=raw_positive_control_guides
        self.raw_observation_guides=raw_observation_guides
        self.num_replicates=num_replicates
        self.include_observational_guides_in_fit=include_observational_guides_in_fit
        self.include_positive_control_guides_in_fit=include_positive_control_guides_in_fit
        self.pop1_amplification_factors=pop1_amplification_factors
        self.pop2_amplification_factors=pop2_amplification_factors
        self.monte_carlo_trials=monte_carlo_trials
        self.enable_spatial_prior=enable_spatial_prior
        self.spatial_bandwidth=spatial_bandwidth
        self.baseline_proportion=baseline_proportion
        self.posterior_estimator=posterior_estimator
        self.random_seed=random_seed

class ShrinkageResult:
    def __init__(self, guide_count_LFC_samples_normalized_list: List[List[float]],
            guide_count_posterior_LFC_samples_normalized_list: List[List[float]]):
            self.guide_count_LFC_samples_normalized_list=guide_count_LFC_samples_normalized_list
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
    def precise_gamma(num: float) -> Decimal:
        return np.exp(Decimal(sc.gammaln(num)) )
    
    @staticmethod
    def precise_beta(a: float, b: float) -> Decimal:
        return (StatisticalHelperMethods.precise_gamma(a)*StatisticalHelperMethods.precise_gamma(b))/(StatisticalHelperMethods.precise_gamma(a+b))

    @staticmethod
    def KL_beta(alpha_f: float, beta_f: float, alpha_g: float, beta_g: float):
        # NOTE: Because the beta function can output extremely small values, using Decimal for higher precision
        return float(Decimal.ln(StatisticalHelperMethods.precise_beta(alpha_g, beta_g)/(StatisticalHelperMethods.precise_beta(alpha_f, beta_f)))) + ((alpha_f - alpha_g)*(sc.digamma(alpha_f) - sc.digamma(alpha_f+beta_f))) + ((beta_f - beta_g)*(sc.digamma(beta_f) - sc.digamma(alpha_f+beta_f)))


    
    # Deprecated - this causes a non-differential point at the baseline due to the piecewise transformation... 
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


def perform_score_imputation(each_guide: Guide, experiment_guide_sets: ExperimentGuideSets, negative_control_guide_pop1_total_normalized_counts_reps: List[float], negative_control_guide_pop2_total_normalized_counts_reps: List[float], spatial_imputation_prior_strength: List[float], spatial_imputation_likelihood_strength: List[float], replicate_indices: List[int], spatial_bandwidth: float) -> Tuple[List[float], List[float]]:
    # Get Spatial Prior "Likelihood" Counts 
    each_guide_pop1_spatial_contribution_reps: List[float] = np.repeat(0., len(replicate_indices))
    each_guide_pop2_spatial_contribution_reps: List[float] = np.repeat(0., len(replicate_indices))
    
    # Iterate through all neighboring guides
    if each_guide.position is not None:
        for neighboring_guide in np.concatenate([experiment_guide_sets.negative_control_guides, experiment_guide_sets.positive_control_guides, experiment_guide_sets.observation_guides]):
            if (neighboring_guide.identifier != each_guide.identifier) and (neighboring_guide.position is not None):

                # Along with the weight, is the spatial bandwidth also something that we should optimize as well?
                neighboring_guide_spatial_contribution = StatisticalHelperMethods.gaussian_kernel(neighboring_guide.position, each_guide.position, spatial_bandwidth)


                each_guide_pop1_spatial_contribution_reps = each_guide_pop1_spatial_contribution_reps + (neighboring_guide_spatial_contribution*neighboring_guide.pop1_normalized_count_reps[replicate_indices])
                each_guide_pop2_spatial_contribution_reps = each_guide_pop2_spatial_contribution_reps + (neighboring_guide_spatial_contribution*neighboring_guide.pop2_normalized_count_reps[replicate_indices])

    pop1_spatial_posterior_alpha = (spatial_imputation_prior_strength*negative_control_guide_pop1_total_normalized_counts_reps[replicate_indices]) + (spatial_imputation_likelihood_strength * each_guide_pop1_spatial_contribution_reps)
    imputation_posterior_alpha = pop1_spatial_posterior_alpha

    pop2_spatial_posterior_beta = (spatial_imputation_prior_strength*negative_control_guide_pop2_total_normalized_counts_reps[replicate_indices]) + (spatial_imputation_likelihood_strength * each_guide_pop2_spatial_contribution_reps)
    imputation_posterior_beta = pop2_spatial_posterior_beta

    return imputation_posterior_alpha, imputation_posterior_beta

def optimize_spatial_imputation_prior_strength(experiment_guide_sets: ExperimentGuideSets, negative_control_guide_pop1_total_normalized_counts_reps: List[float], negative_control_guide_pop2_total_normalized_counts_reps: List[float], replicate_indices: List[int], spatial_bandwidth: float) -> Tuple[List[float], List[float]]:

    # Get list of prior weight to test
    spatial_imputation_prior_strength_selected: List[float] = []
    


    def optimize_imputation_model_weights(rep_i, params):
        spatial_imputation_prior_strength_test,spatial_imputation_likelihood_strength_test = params

        # Placeholder variable to hold total sum KL score for each replicate separately
        KL_guide_imputation_score_total_positive: float = 0
        KL_guide_imputation_score_total_negative: float = 0
        KL_guide_imputation_score_total_observation: float = 0
        
        # Iterate through each guide to test prior with tested weight
        def retrieve_objective_of_guide_set(guide_set: List[Guide]):
            KL_guide_imputation_score_total: float = 0

            for each_guide in guide_set:

                # Ensure that the guide contains a position

                # Get the posterior


                imputation_posterior_alpha, imputation_posterior_beta = perform_score_imputation(each_guide, experiment_guide_sets, negative_control_guide_pop1_total_normalized_counts_reps, negative_control_guide_pop2_total_normalized_counts_reps, spatial_imputation_prior_strength_test, spatial_imputation_likelihood_strength_test, [rep_i], spatial_bandwidth)

                imputation_posterior_alpha = imputation_posterior_alpha[0]
                imputation_posterior_beta = imputation_posterior_beta[0]

                true_alpha = each_guide.pop1_normalized_count_reps[rep_i]
                true_beta = each_guide.pop2_normalized_count_reps[rep_i]

                # Calculate KL divergence between the posterior and the likelihood
                KL_guide_imputation_score: float = StatisticalHelperMethods.KL_beta(true_alpha, true_beta, imputation_posterior_alpha, imputation_posterior_beta)


                # Add score to the main placeholder to get the final sum
                KL_guide_imputation_score_total = KL_guide_imputation_score_total + KL_guide_imputation_score 
                
                return KL_guide_imputation_score_total, len(guide_set)

                # TODO: I feel that this will implicitly place weight on null guides (since the majority will be null), so I wonder if there is a way to have the score prioritize guides that deviate from the null - think about how "deviation" from the null is defined quantitatively, and how this quantitation would be integrated into the heuristic. (i.e. (1+coef*KL(nc_beta, guide_i_beta)*KL(guide_i_beta, posterior_beta))) Instead of KL(nc_beta, guide_i_beta), could also just use P(guide_i_beta_normalized=baseline_proportion)?? I believe that should be equivalent to P(guide_i_beta=E[nc_beta]), though by using KL we are considering the variance of the nc_beta.

    
        KL_guide_imputation_score_total_negative, negative_n = retrieve_objective_of_guide_set(experiment_guide_sets.negative_control_guides)
        KL_guide_imputation_score_total_positive, positive_n = retrieve_objective_of_guide_set(experiment_guide_sets.positive_control_guides)
        KL_guide_imputation_score_total_observation, observation_n = retrieve_objective_of_guide_set(experiment_guide_sets.observation_guides)

        KL_guide_imputation_score_total_combined_avg = (KL_guide_imputation_score_total_negative+KL_guide_imputation_score_total_positive+KL_guide_imputation_score_total_observation)/(negative_n+positive_n+observation_n)

        return KL_guide_imputation_score_total_combined_avg, KL_guide_imputation_score_total_negative/negative_n, KL_guide_imputation_score_total_positive/positive_n, KL_guide_imputation_score_total_observation/observation_n


    def optimize_imputation_model_weights_wrapper(rep_i, guide_set_weights: Union[List[float], None], params):
        KL_guide_imputation_score_total_combined_avg, KL_guide_imputation_score_total_negative_avg, KL_guide_imputation_score_total_positive_avg, KL_guide_imputation_score_total_observation_avg = optimize_imputation_model_weights(rep_i, params)

        if guide_set_weights is None:
            return KL_guide_imputation_score_total_combined_avg
        else:
            return (guide_set_weights[0]*KL_guide_imputation_score_total_negative_avg) + (guide_set_weights[1]*KL_guide_imputation_score_total_positive_avg) + (guide_set_weights[2]*KL_guide_imputation_score_total_observation_avg)



    spatial_imputation_prior_strength_selected: List[float] = []
    spatial_imputation_likelihood_strength_selected: List[float] = []
    for rep_i in replicate_indices:
        param_vals=[]
        loss_vals=[]
        def store_values(x, convergence):
            f = optimize_imputation_model_weights_p(x)
            print("X: {}, f: {}".format(x, f))
            param_vals.append(x)
            loss_vals.append(f)


        optimize_imputation_model_weights_p = functools.partial(optimize_imputation_model_weights_wrapper, rep_i, None)

        res = scipy.optimize.differential_evolution(optimize_imputation_model_weights_p, bounds=[(0.000001, 10),(0.000001, 10)], callback=store_values) # TODO: Set bounds as just positive - ask chatgpt how...
        

        X=[param[0] for param in param_vals]
        Y=[param[1] for param in param_vals]
        plt.scatter(X,Y , c=loss_vals)
        #plt.plot([param[0] for param in param_vals], [param[1] for param in param_vals], color="black")
        for i in range(len(X) - 1):
            x1, y1 = X[i], Y[i]
            x2, y2 = X[i + 1], Y[i + 1]
            plt.annotate("", xy=(x2, y2), xycoords='data', xytext=(x1, y1), textcoords='data',
                        arrowprops=dict(arrowstyle="->"))
                        
        plt.xlabel("Prior Strength")
        plt.ylabel("Likelihood Strength")
        plt.title("Rep: {}".format(rep_i))
        plt.colorbar(label="loss")
        plt.show()
        if res.success is True:
            spatial_imputation_prior_strength, spatial_imputation_likelihood_strength = res.x
            
            KL_guide_imputation_score_total_combined_avg, KL_guide_imputation_score_total_negative_avg, KL_guide_imputation_score_total_positive_avg, KL_guide_imputation_score_total_observation_avg = optimize_imputation_model_weights(rep_i, res.x)

            print("KL Negative Set Average: {}".format(KL_guide_imputation_score_total_negative_avg))
            print("KL Positive Set Average: {}".format(KL_guide_imputation_score_total_positive_avg))
            print("KL Observation Set Average: {}".format(KL_guide_imputation_score_total_observation_avg))
            print("KL Combined Set Average: {}".format(KL_guide_imputation_score_total_combined_avg))



            spatial_imputation_prior_strength_selected.append(spatial_imputation_prior_strength)
            spatial_imputation_likelihood_strength_selected.append(spatial_imputation_likelihood_strength)
        else:
            raise Exception("Neighborhood imputation optimization failure: {}".format(res.message)) # TODO: Put a more detailed message on optimization failure, such as the message from the result object res.message

    spatial_imputation_prior_strength_selected = np.asarray(spatial_imputation_prior_strength_selected)
    spatial_imputation_likelihood_strength_selected = np.asarray(spatial_imputation_likelihood_strength_selected)

    return spatial_imputation_prior_strength_selected, spatial_imputation_likelihood_strength_selected

def perform_score_shrinkage(each_guide: Guide, negative_control_guide_pop1_total_normalized_counts_reps: List[float], negative_control_guide_pop2_total_normalized_counts_reps: List[float], shrinkage_prior_strength: List[float], unweighted_prior_alpha: List[float], unweighted_prior_beta: List[float], baseline_proportion: float,  monte_carlo_trials: int, random_seed: int, replicate_indices: List[int]) -> ShrinkageResult:

    # NOTE: When indexing the guides that contain all replicates, use "replicates" array, when indexing a subset of the replicates from upstream runs, using "replicate_order" array. Not the best system since prone to error due to mis-indexing. TODO: Figure out better design
    replicate_order = np.arange(len(replicate_indices))
    shrinkage_prior_alpha = shrinkage_prior_strength * unweighted_prior_alpha
    shrinkage_prior_beta = shrinkage_prior_strength * unweighted_prior_beta

    #
    # Monte-Carlo sampling of beta distributions (i.e. conjugate priors and posterior distributions)
    #

    # This is for visualization of the prior
    shrinkage_prior_beta_samples_list: List[List[float]] = np.asarray([[beta.rvs(shrinkage_prior_alpha[rep_order], shrinkage_prior_beta[rep_order], size=monte_carlo_trials, random_state=random_seed)] for rep_order in replicate_order])

    # This is for visualization of the non-influenced data beta distribution
    guide_count_beta_samples_list: List[List[float]] = np.asarray([beta.rvs(each_guide.pop1_normalized_count_reps[rep_i], each_guide.pop2_normalized_count_reps[rep_i], size=monte_carlo_trials, random_state=random_seed) for rep_i in replicate_indices])

    # This is used for normalization of the non-influced data beta distribution
    control_count_beta_samples_list: List[List[float]] = np.asarray([beta.rvs(negative_control_guide_pop1_total_normalized_counts_reps[rep_i], negative_control_guide_pop2_total_normalized_counts_reps[rep_i], size=monte_carlo_trials, random_state=random_seed) for rep_i in replicate_indices])

    # This is the final shrunk posterior
    guide_count_posterior_beta_samples_list: List[List[float]] = np.asarray([beta.rvs(shrinkage_prior_alpha[rep_order] + each_guide.pop1_normalized_count_reps[rep_i], shrinkage_prior_beta[rep_order] + each_guide.pop2_normalized_count_reps[rep_i], size=monte_carlo_trials, random_state=random_seed) for rep_order, rep_i in enumerate(replicate_indices)])

    # This is used for normalization
    control_count_posterior_beta_samples_list: List[List[float]] = np.asarray([beta.rvs(shrinkage_prior_alpha[rep_order] + negative_control_guide_pop1_total_normalized_counts_reps[rep_i], shrinkage_prior_beta[rep_order] + negative_control_guide_pop2_total_normalized_counts_reps[rep_i], size=monte_carlo_trials, random_state=random_seed) for rep_order, rep_i in enumerate(replicate_indices)])


    guide_count_LFC_samples_normalized_list = np.asarray([np.log(guide_count_beta_samples_list[rep_order]/control_count_beta_samples_list[rep_order]) for rep_order in replicate_order])
    
    guide_count_posterior_LFC_samples_normalized_list = np.asarray([np.log(guide_count_posterior_beta_samples_list[rep_order]/ control_count_posterior_beta_samples_list[rep_order]) for rep_order in replicate_order])
    
    # NOTE: When needed, I can add more to this object
    shrinkage_result = ShrinkageResult(guide_count_LFC_samples_normalized_list=guide_count_LFC_samples_normalized_list,
    guide_count_posterior_LFC_samples_normalized_list=guide_count_posterior_LFC_samples_normalized_list)

    return shrinkage_result


def optimize_score_shrinkage_prior_strength(guides_for_fit: List[Guide], experiment_guide_sets: ExperimentGuideSets, replicate_indices: List[int], negative_control_guide_pop1_total_normalized_counts_reps: List[float], negative_control_guide_pop2_total_normalized_counts_reps: List[float], enable_spatial_prior: bool, spatial_imputation_model_weights: Tuple[List[float], List[float]], baseline_proportion: float, spatial_bandwidth: float, monte_carlo_trials: int, random_seed: Union[int, None]) -> List[float]:
    

    # TODO: Create function for optimization. Also see if there is a good way to plot the optimization performance for both optimizations done in the package. Continue to debug and test to fix issues.
    def optimize_shrinkage_model_weights(rep_i, params):
        # TODO: There is a major inefficiency for this optimization function and the other optimization function, is that even though optimization is done for each rep_i, the posterior is calculated for all reps then indexed for the rep_i argument... I think being able to index the arguments to the perform_score_imputation should work if the typehint is correct. 
        shrinkage_prior_strength_test=np.asarray([params[0]])
        
        total_normalized_counts_per_guide : List[float] = [each_guide.pop1_normalized_count_reps[rep_i] + each_guide.pop2_normalized_count_reps[rep_i] for each_guide in guides_for_fit]

        # NOTE: The first list corresponds to each guide, the second list corresponds to number of replicates, the value is the mean LFC
        guide_count_posterior_LFC_normalized_mean_list_per_guide : List[float] = []

        for each_guide in guides_for_fit:

            # TODO: The code for calculating the posterior inputs for the spatial_imputation model could be modularized so that there are not any repetitive code

            # By default, set the unweighted prior as the negative control normalized counts
            unweighted_prior_alpha: float = np.asarray([negative_control_guide_pop1_total_normalized_counts_reps[rep_i]])
            unweighted_prior_beta: float = np.asarray([negative_control_guide_pop2_total_normalized_counts_reps[rep_i]])

            # If able to use spatial information, replace the unweighted priors with the spatial imputational posterior
            spatial_imputation_prior_strength, spatial_imputation_likelihood_strength = spatial_imputation_model_weights

            imputation_posterior_alpha, imputation_posterior_beta = perform_score_imputation(each_guide, experiment_guide_sets, negative_control_guide_pop1_total_normalized_counts_reps, negative_control_guide_pop2_total_normalized_counts_reps, spatial_imputation_prior_strength, spatial_imputation_likelihood_strength, [rep_i], spatial_bandwidth)

            # Propogate the imputation posterior to the shrinkage prior
            unweighted_prior_alpha = imputation_posterior_alpha
            unweighted_prior_beta = imputation_posterior_beta

            shrinkage_result: ShrinkageResult = perform_score_shrinkage(each_guide, negative_control_guide_pop1_total_normalized_counts_reps, negative_control_guide_pop2_total_normalized_counts_reps, shrinkage_prior_strength_test, unweighted_prior_alpha, unweighted_prior_beta, baseline_proportion, monte_carlo_trials, random_seed, [rep_i])


            #shrinkage_result.guide_count_beta_samples_normalized_list
            #shrinkage_result.guide_count_LFC_samples_normalized_list
            #shrinkage_result.guide_count_posterior_beta_samples_normalized_list

            # NOTE: List[List[float]], first list is each replicate, second list is the monte-carlo samples. We want the mean of the monte-carlo samples next
            guide_count_posterior_LFC_samples_normalized_list: List[List[float]] = shrinkage_result.guide_count_posterior_LFC_samples_normalized_list
            
            # This corresponds to the guide posterior mean LFC for each replicate separately for shrinkage prior weight optimization. After optimization of the shrinkage weight, the mean LFC of the averaged posterior of the replicates will be used.
            guide_count_posterior_LFC_normalized_mean_list: List[float] = np.mean(guide_count_posterior_LFC_samples_normalized_list, axis=0) 

            guide_count_posterior_LFC_normalized_mean_list_per_guide.append(guide_count_posterior_LFC_normalized_mean_list[rep_i])


        # NOTE: Instead of relying on numpy matrix operations to calculate heteroscedasticity, will just iterate through each replicate individually since it may be challenging to do otherwise. Could be a potential optimization in the future to use np matrix operations, but it may not be as readable anyways. (2/4/2023)

        #
        # Calculate the Breusch-Pagan statistic
        #
        # Prepare X - which is the normalized count, since the objective is to reduce hederoscedasticity across count
        total_normalized_count_per_guide_X: List[float] = total_normalized_counts_per_guide

        # Prepare Y - which is the LFC score, since we want to reduce heteroscedasticity of the LFC
        LFC_posterior_mean_per_guide_Y: List[float] = guide_count_posterior_LFC_normalized_mean_list_per_guide

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

        return LFC_posterior_mean_per_guide_M_BP_statistic


    shrinkage_prior_strength_selected: List[float] = []

    for rep_i in replicate_indices:
        optimize_shrinkage_model_weights_p = functools.partial(optimize_shrinkage_model_weights, rep_i)

        param_vals=[]
        loss_vals=[]
        def store_values(x, *args):
            f = optimize_shrinkage_model_weights_p(x)
            print("X: {}, f: {}".format(x, f))
            param_vals.append(x)
            loss_vals.append(f)


        res = scipy.optimize.minimize(optimize_shrinkage_model_weights_p, [1], bounds=[(0.000001, 500)], method="L-BFGS-B", callback=store_values, options={"factr": 1e3}) # TODO: Set bounds as just positive - ask chatgpt how...
        
        plt.scatter([param[0] for param in param_vals], loss_vals)
        plt.xlabel("Prior Strength")
        plt.ylabel("Loss")
        plt.title("Rep: {}".format(rep_i))
        plt.show()

        if res.success is True:
            shrinkage_prior_strength = res.x
            shrinkage_prior_strength_selected.append(shrinkage_prior_strength)
        else:

            raise Exception("Optimization failure") # TODO: Put a more detailed message on optimization failure, such as the message from the result object res.message

    shrinkage_prior_strength_selected = np.asarray(shrinkage_prior_strength_selected).flatten() 
    return shrinkage_prior_strength_selected






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
    spatial_imputation_model_weights: Union[Tuple[List[float], List[Tuple]], None] = None, # This could be optimized by maximizing correlation of guide with neighborhood (perhaps in binomial GLM fashion?),
    baseline_proportion: float = 0.5, # TODO: Perform validation between (0,1), also accept None value for perfrming no normalization (or have that be another argument)
    shrinkage_prior_strength: Union[List[float], None] = None, 
    posterior_estimator: str = "mean",
    random_seed: Union[int, None] = None
    ):

    raw_negative_control_guides = copy.deepcopy(negative_control_guides)
    raw_positive_control_guides = copy.deepcopy(positive_control_guides)
    raw_observation_guides = copy.deepcopy(observation_guides)

    # Validation
    assert posterior_estimator.upper() in ["MEAN", "MODE"], "Posterior estimator must be of value 'mean' or 'mode'"
    assert monte_carlo_trials>0, "Monte-Carlo trial amout must be greater than 0"
    assert num_replicates>0, "Number of replicates specified must be greater than 0"
    assert spatial_bandwidth>0, "Spatial prior bandwidth used for Gaussian kernel must be greater than 0"


    replicate_indices = np.asarray(range(num_replicates))

    for guide in negative_control_guides:
        assert num_replicates == len(guide.pop1_raw_count_reps) == len(guide.pop2_raw_count_reps), "Guide {} number of counts does not equal replicates"
    for guide in observation_guides:
        assert num_replicates == len(guide.pop1_raw_count_reps) == len(guide.pop2_raw_count_reps), "Guide {} number of counts does not equal replicates"
    for guide in observation_guides:
        assert num_replicates == len(guide.pop1_raw_count_reps) == len(guide.pop2_raw_count_reps), "Guide {} number of counts does not equal replicates"


    # Set the amplification factors
    pop1_amplification_factors = np.repeat(1.,num_replicates) if pop1_amplification_factors is None else np.asarray(pop1_amplification_factors)
    pop2_amplification_factors = np.repeat(1.,num_replicates) if pop2_amplification_factors is None else np.asarray(pop2_amplification_factors)
    
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
    
    
    if spatial_imputation_model_weights is not None:
        spatial_imputation_model_weights = (np.asarray(spatial_imputation_model_weights[0]), np.asarray(spatial_imputation_model_weights[1]))
        assert len(spatial_imputation_model_weights[0]) == num_replicates, "Number of spatial imputation prior strength values in list must equal number of replicates"
        assert len(spatial_imputation_model_weights[1]) == num_replicates, "Number of spatial imputation prior strength values in list must equal number of replicates"

    if shrinkage_prior_strength is not None:
        shrinkage_prior_strength = np.asarray(shrinkage_prior_strength)
        assert len(shrinkage_prior_strength) == num_replicates, "Number of shrinkage prior strength values in list must equal number of replicates"

    # Set the guides used for shrinkage model fitting
    guides_for_fit: List[Guide] = negative_control_guides
    guides_for_fit = np.concatenate([guides_for_fit, observation_guides]) if include_observational_guides_in_fit else guides_for_fit
    guides_for_fit = np.concatenate([guides_for_fit, positive_control_guides]) if include_positive_control_guides_in_fit else guides_for_fit
    assert len(guides_for_fit) > 0, "Total guides used for shrinkage model fitting is zero"

    # Create all guides set used for informing neighborhood prior, performing final shrinkage, and visualization    
    experiment_guide_sets: ExperimentGuideSets = ExperimentGuideSets(negative_control_guides, positive_control_guides, observation_guides)

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
    if spatial_imputation_model_weights is None and enable_spatial_prior:
            # No spatial_control_prior_strength was provided. We will need to optimize by assessing the correlation of the posterior to the actual guide. 
            # Notes: We will measure correlation somehow via a heuristic, starting with calculating the KL divergence between the beta distribution of the prior and from the counts. But how will we weight KL scores from guides with high counts compared to low counts? This is why I liked implementing the binomial likelihood somehow.
            print("Optimizing imputation weights")
            spatial_imputation_model_weights = optimize_spatial_imputation_prior_strength(experiment_guide_sets, negative_control_guide_pop1_total_normalized_counts_reps, negative_control_guide_pop2_total_normalized_counts_reps, replicate_indices, spatial_bandwidth)
            print("Selected imputation weights: {}".format(spatial_imputation_model_weights))




    #
    # Identification of optimal shrinkage_prior_strength
    #
    # TODO: I need to find a way to ensure that the fitted prior will work for guides that don't have position, i.e. I assume those without position will not have a prior based on the spatial information, but will the shrinkage weight of the imputation posterior be difference? 
    if shrinkage_prior_strength is None:
        # NOTE: Here, we will be using selected guides for fit since the heteroscedasticity statistic may be biased towards positive effect guides.

        print("Optimizing shrinkage prior weights")
        shrinkage_prior_strength = optimize_score_shrinkage_prior_strength(guides_for_fit, experiment_guide_sets, replicate_indices, negative_control_guide_pop1_total_normalized_counts_reps, negative_control_guide_pop2_total_normalized_counts_reps, enable_spatial_prior, spatial_imputation_model_weights, baseline_proportion, spatial_bandwidth, monte_carlo_trials, random_seed)

        print("Selected shrinkage weights: {}".format(shrinkage_prior_strength))
        # LEFTOFF: Just finished draft of selecting shrinkage prior weight. Next, move to function and proceed with calling the selection to run final model to get final posterior, then get the averaged posterior, calculating the MAP (or mean), and return. id probably say the mean is best if there is equal trust in all replicates, if there is not, then mode since any outlier density in the averaged posterior (due to the outlier replicate) will not influence the mode as much. It could be a nice quality control metric to show the concordance of each replicates posterior distribution.l

    # Perform final model inference:
    def inference_guide_set(guide_set: List[Guide], experiment_guide_sets: ExperimentGuideSets):
        for each_guide in guide_set:

        
            # TODO: The code for calculating the posterior inputs for the spatial_imputation model could be modularized so that there are not any repetitive code

            # By default, set the unweighted prior as the negative control normalized counts
            unweighted_prior_alpha = negative_control_guide_pop1_total_normalized_counts_reps
            unweighted_prior_beta = negative_control_guide_pop2_total_normalized_counts_reps

            # TODO IMPORTANT: For guides with no position, should still optimize
            # If able to use spatial information, replace the unweighted priors with the spatial imputational posterior
            if (each_guide.position != None) and enable_spatial_prior: 
                spatial_imputation_prior_strength, spatial_imputation_likelihood_strength = spatial_imputation_model_weights

                imputation_posterior_alpha, imputation_posterior_beta = perform_score_imputation(each_guide, experiment_guide_sets, negative_control_guide_pop1_total_normalized_counts_reps, negative_control_guide_pop2_total_normalized_counts_reps, spatial_imputation_prior_strength, spatial_imputation_likelihood_strength, replicate_indices, spatial_bandwidth)

                # Propogate the imputation posterior to the shrinkage prior
                unweighted_prior_alpha = imputation_posterior_alpha
                unweighted_prior_beta = imputation_posterior_beta

            print("Shrinkage Prior: a={}, b={}".format(unweighted_prior_alpha,unweighted_prior_beta))
            shrinkage_result: ShrinkageResult = perform_score_shrinkage(each_guide, negative_control_guide_pop1_total_normalized_counts_reps, negative_control_guide_pop2_total_normalized_counts_reps, shrinkage_prior_strength, unweighted_prior_alpha, unweighted_prior_beta, baseline_proportion,  monte_carlo_trials, random_seed, replicate_indices)


            # NOTE: List[List[float]], first list is each replicate, second list is the monte-carlo samples. We want the mean of the monte-carlo samples next
            guide_count_posterior_LFC_samples_normalized_list: List[List[float]] = shrinkage_result.guide_count_posterior_LFC_samples_normalized_list 
            

            guide_count_posterior_LFC_samples_normalized_average = np.mean(guide_count_posterior_LFC_samples_normalized_list, axis=0)

            LFC_estimate_combined = None
            if posterior_estimator.upper() == "MEAN":
                LFC_estimate_combined = np.mean(guide_count_posterior_LFC_samples_normalized_average)
                LFC_estimate_per_replicate = np.mean(guide_count_posterior_LFC_samples_normalized_list, axis=1)
            elif posterior_estimator.upper() == "MODE":
                LFC_estimate_combined = StatisticalHelperMethods.calculate_map(guide_count_posterior_LFC_samples_normalized_average)
                LFC_estimate_per_replicate = np.asarray([StatisticalHelperMethods.calculate_map(guide_count_posterior_LFC_samples_normalized) for guide_count_posterior_LFC_samples_normalized in guide_count_posterior_LFC_samples_normalized_list])

            each_guide.LFC_estimate_combined = LFC_estimate_combined
            each_guide.LFC_estimate_per_replicate = LFC_estimate_per_replicate
        
        return guide_set
    
    print("NEGATIVE_CONTROLS")
    negative_control_guides = inference_guide_set(negative_control_guides, experiment_guide_sets)

    print("\nOBSERVATIONS")
    observation_guides = inference_guide_set(observation_guides, experiment_guide_sets)

    print("\nPOSITIVES")
    positive_control_guides = inference_guide_set(positive_control_guides, experiment_guide_sets)

    return CrisprShrinkageResult(
        adjusted_negative_control_guides=negative_control_guides,
        adjusted_observation_guides=observation_guides,
        adjusted_positive_control_guides=positive_control_guides,
        shrinkage_prior_strength=shrinkage_prior_strength,
        spatial_imputation_model_weights=spatial_imputation_model_weights,
        raw_negative_control_guides=raw_negative_control_guides,
        raw_positive_control_guides=raw_positive_control_guides,
        raw_observation_guides=raw_observation_guides,
        num_replicates=num_replicates,
        include_observational_guides_in_fit=include_observational_guides_in_fit,
        include_positive_control_guides_in_fit=include_positive_control_guides_in_fit,
        pop1_amplification_factors=pop1_amplification_factors,
        pop2_amplification_factors=pop2_amplification_factors,
        monte_carlo_trials=monte_carlo_trials,
        enable_spatial_prior=enable_spatial_prior,
        spatial_bandwidth=spatial_bandwidth,
        baseline_proportion=baseline_proportion, # TODO: Perform validation between (0,1), also accept None value for perfrming no normalization (or have that be another argument)
        posterior_estimator=posterior_estimator,
        random_seed=random_seed
    )


if __name__ == "__main__":

    from scipy.stats import binom
    from scipy.stats import uniform
    from scipy.stats import expon
    import numpy as np 

    null_proportion = 0.5
    positive_proportion = 0.6
    target_null_proportion = 0.5
    target_positive_population = 0.6

    num_ctrl_guides = 100
    num_pos_guides = 50

    reps = 3
    max_dup_factor = 30
    max_guide_molecule_factor = 30

    pop1_dup_factor_list = np.asarray([np.round(uniform.rvs(1, max_dup_factor)) for _ in range(reps)])
    pop2_dup_factor_list = np.asarray([np.round(uniform.rvs(1, max_dup_factor)) for _ in range(reps)])

    #expon.rvs(loc=1, scale=1000, size=num_guides)
    #uniform.rvs(2, 200, size=num_guides)
    def get_counts(num_guides, proportion):
        pop1_list_reps = []
        pop2_list_reps = []

        for rep_i in range(reps):
            n_list = np.round(uniform.rvs(2, max_guide_molecule_factor, size=num_guides)).astype(int)
            pop1_list = binom.rvs(n_list, proportion, size=num_guides) 
            pop2_list = n_list - pop1_list

            pop1_list_reps.append(pop1_list * pop1_dup_factor_list[rep_i])
            pop2_list_reps.append(pop2_list * pop2_dup_factor_list[rep_i])
        
        return np.asarray(pop1_list_reps), np.asarray(pop2_list_reps)


    tiling_length = 100

    observation_guides = []
    for position in range(tiling_length):
        counts = get_counts(1, target_null_proportion)
        pop1_raw_count_reps = counts[0].transpose()[0] + 1
        pop2_raw_count_reps = counts[1].transpose()[0] + 1
        guide = Guide(identifier="observation_{}".format(position), position=position, pop1_raw_count_reps= pop1_raw_count_reps, pop2_raw_count_reps=pop2_raw_count_reps)

        observation_guides.append(guide)

    observation_guides = np.asarray(observation_guides)

    negative_guides = []
    for i in range(num_ctrl_guides):
        counts = get_counts(1, null_proportion)
        pop1_raw_count_reps = counts[0].transpose()[0] + 1
        pop2_raw_count_reps = counts[1].transpose()[0] + 1
        guide = Guide(identifier="negative_{}".format(i), position=None, pop1_raw_count_reps= pop1_raw_count_reps, pop2_raw_count_reps=pop2_raw_count_reps)

        negative_guides.append(guide)

    negative_guides = np.asarray(negative_guides)

    positive_guides = []
    for i in range(num_pos_guides):
        counts = get_counts(1, positive_proportion)
        pop1_raw_count_reps = counts[0].transpose()[0] + 1
        pop2_raw_count_reps = counts[1].transpose()[0] + 1
        guide = Guide(identifier="positive_{}".format(i), position=None, pop1_raw_count_reps= pop1_raw_count_reps, pop2_raw_count_reps=pop2_raw_count_reps)

        positive_guides.append(guide)

    positive_guides = np.asarray(positive_guides)

    perform_adjustment(
        negative_control_guides = negative_guides,
        positive_control_guides = positive_guides,
        observation_guides = observation_guides,
        num_replicates = reps,
        include_observational_guides_in_fit = True,
        include_positive_control_guides_in_fit = False,
        pop1_amplification_factors = pop1_dup_factor_list,
        pop2_amplification_factors = pop2_dup_factor_list,
        monte_carlo_trials = 1000,
        enable_spatial_prior =  True,
        spatial_bandwidth = 3,
        spatial_imputation_model_weights = None,
        baseline_proportion = 0.5, # TODO: Perform validation between (0,1), also accept None value for perfrming no normalization (or have that be another argument)
        shrinkage_prior_strength = None, 
        posterior_estimator = "mean",
        random_seed = 234
        )