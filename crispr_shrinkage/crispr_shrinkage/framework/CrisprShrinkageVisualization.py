#!/usr/bin/env python
from typing import List, Union, Tuple
from matplotlib import pyplot as plt
import numpy as np
from CrisprShrinkage import CrisprShrinkageResult, Guide
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass


@dataclass
class CrisprShrinkageVisualizationInput:
    replicate_indices: List[int]
    explanatory_guides: List[Guide]
    
    observational_position: List[Union[float, int]]
    positive_control_positions: List[Union[float, int]]
    negative_control_positions: List[Union[float, int]]
    explanatory_positions: List[Union[float, int]]

    observational_lfc: List[float]
    positive_control_lfc: List[float]
    negative_control_lfc: List[float]
    explanatory_lfc: List[float]

    observational_lfc_CI: List[Tuple[float,float]]
    positive_control_lfc_CI: List[Tuple[float,float]]
    negative_control_lfc_CI: List[Tuple[float,float]]
    explanatory_lfc_CI: List[Tuple[float,float]]

    observational_lfc_rep: List[List[float]]
    positive_control_lfc_rep: List[List[float]]
    negative_control_lfc_rep: List[List[float]]
    explanatory_lfc_rep: List[List[float]]

    sample_population_negative_control_total_normalized_count: List[float]
    control_population_negative_control_total_normalized_count: List[float]

    observational_raw_lfc_rep: List[List[float]]
    positive_control_raw_lfc_rep: List[List[float]]
    negative_control_raw_lfc_rep: List[List[float]]
    explanatory_raw_lfc_rep: List[List[float]]

    observational_count_rep: List[List[Union[int, float]]]
    positive_control_count_rep: List[List[Union[int, float]]]
    negative_control_count_rep: List[List[Union[int, float]]]
    explanatory_count_rep: List[List[Union[int, float]]]


def prepare_crispr_shrinkage_visualization_input(crispr_shrinkage_result: CrisprShrinkageResult):
    replicate_indices = range(crispr_shrinkage_result.num_replicates)
    explanatory_guides = [guide for guide in np.concatenate([crispr_shrinkage_result.adjusted_observation_guides, crispr_shrinkage_result.adjusted_positive_control_guides, crispr_shrinkage_result.adjusted_negative_control_guides]) if guide.is_explanatory is True]


    observational_position = np.asarray([guide.position for guide in crispr_shrinkage_result.adjusted_observation_guides])
    positive_positions = np.asarray([guide.position for guide in crispr_shrinkage_result.adjusted_positive_control_guides])
    negative_positions = np.asarray([guide.position for guide in crispr_shrinkage_result.adjusted_positive_control_guides])
    explanatory_positions =  np.asarray([guide.position for guide in explanatory_guides])

    negative_lfc = np.asarray([guide.LFC_estimate_combined for guide in crispr_shrinkage_result.adjusted_negative_control_guides])
    positive_lfc = np.asarray([guide.LFC_estimate_combined for guide in crispr_shrinkage_result.adjusted_positive_control_guides])
    observational_lfc = np.asarray([guide.LFC_estimate_combined for guide in crispr_shrinkage_result.adjusted_observation_guides])
    explanatory_lfc = np.asarray([guide.LFC_estimate_combined for guide in explanatory_guides])

    observational_lfc_CI = [guide.LFC_estimate_combined_CI for guide in crispr_shrinkage_result.adjusted_observation_guides]
    positive_control_lfc_CI = [guide.LFC_estimate_combined_CI for guide in crispr_shrinkage_result.adjusted_positive_control_guides]
    negative_control_lfc_CI = [guide.LFC_estimate_combined_CI for guide in crispr_shrinkage_result.adjusted_negative_control_guides]
    explanatory_lfc_CI =  [guide.LFC_estimate_combined_CI for guide in explanatory_guides]

    observational_lfc_rep = np.asarray([[guide.LFC_estimate_per_replicate[rep_i] for guide in crispr_shrinkage_result.adjusted_observation_guides] for rep_i in replicate_indices])
    negative_lfc_rep = np.asarray([[guide.LFC_estimate_per_replicate[rep_i] for guide in crispr_shrinkage_result.adjusted_negative_control_guides] for rep_i in replicate_indices])
    positive_lfc_rep = np.asarray([[guide.LFC_estimate_per_replicate[rep_i] for guide in crispr_shrinkage_result.adjusted_positive_control_guides] for rep_i in replicate_indices])
    explanatory_lfc_rep = np.asarray([[guide.LFC_estimate_per_replicate[rep_i] for guide in explanatory_guides] for rep_i in replicate_indices])

    sample_population_negative_control_total_normalized_count = np.asarray([guide.sample_population_normalized_count_reps for guide in crispr_shrinkage_result.adjusted_negative_control_guides]).sum(axis=0)
    control_population_negative_control_total_normalized_count = np.asarray([guide.control_population_normalized_count_reps for guide in crispr_shrinkage_result.adjusted_negative_control_guides]).sum(axis=0)


    calculate_raw_lfc = lambda guide, rep_i: np.log((guide.sample_population_normalized_count_reps[rep_i] * control_population_negative_controls[rep_i])/(guide.control_population_normalized_count_reps[rep_i] * sample_population_negative_controls[rep_i]))


    observational_raw_lfc_rep = np.asarray([[calculate_raw_lfc(guide, rep_i) for guide in crispr_shrinkage_result.adjusted_observation_guides]  for rep_i in replicate_indices])
    positive_raw_lfc_rep = np.asarray([[calculate_raw_lfc(guide, rep_i) for guide in crispr_shrinkage_result.adjusted_positive_control_guides] for rep_i in replicate_indices])
    negative_raw_lfc_rep = np.asarray([[calculate_raw_lfc(guide, rep_i) for guide in crispr_shrinkage_result.adjusted_negative_control_guides]  for rep_i in replicate_indices])
    explanatory_raw_lfc_rep = np.asarray([[calculate_raw_lfc(guide, rep_i) for guide in explanatory_guides]  for rep_i in replicate_indices])


    observational_count_rep = np.asarray([[guide.sample_population_normalized_count_reps[rep_i] + guide.control_population_normalized_count_reps[rep_i] for guide in crispr_shrinkage_result.adjusted_observation_guides] for rep_i in replicate_indices])
    positive_count_rep = np.asarray([[guide.sample_population_normalized_count_reps[rep_i] + guide.control_population_normalized_count_reps[rep_i] for guide in crispr_shrinkage_result.adjusted_positive_control_guides] for rep_i in replicate_indices])
    negative_count_rep = np.asarray([[guide.sample_population_normalized_count_reps[rep_i] + guide.control_population_normalized_count_reps[rep_i] for guide in crispr_shrinkage_result.adjusted_negative_control_guides] for rep_i in replicate_indices])
    explanatory_count_rep = np.asarray([[guide.sample_population_normalized_count_reps[rep_i] + guide.control_population_normalized_count_reps[rep_i] for guide in explanatory_guides] for rep_i in replicate_indices])


    #  TODO: add properties that contain descriptive information ont he screen for labeling, such as the population names
    crispr_shrinkage_visualization_input = CrisprShrinkageVisualizationInput(
        replicate_indices=replicate_indices,
        explanatory_guides=explanatory_guides,
        
        observational_position=observational_position,
        positive_control_positions=positive_positions,
        negative_control_positions=negative_positions,
        explanatory_positions=explanatory_positions,

        observational_lfc=observational_lfc,
        positive_control_lfc=positive_lfc,
        negative_control_lfc=negative_lfc,
        explanatory_lfc=explanatory_lfc,

        observational_lfc_CI=observational_lfc_CI,
        positive_control_lfc_CI=positive_control_lfc_CI,
        negative_control_lfc_CI=negative_control_lfc_CI,
        explanatory_lfc_CI=explanatory_lfc_CI,

        observational_lfc_rep=observational_lfc_rep,
        positive_control_lfc_rep=positive_lfc_rep,
        negative_control_lfc_rep=negative_lfc_rep,
        explanatory_lfc_rep=explanatory_lfc_rep,

        sample_population_negative_control_total_normalized_count=sample_population_negative_control_total_normalized_count,
        control_population_negative_control_total_normalized_count=control_population_negative_control_total_normalized_count,

        observational_raw_lfc_rep=observational_raw_lfc_rep,
        positive_control_raw_lfc_rep=positive_raw_lfc_rep,
        negative_control_raw_lfc_rep=negative_raw_lfc_rep,
        explanatory_raw_lfc_rep=explanatory_raw_lfc_rep,

        observational_count_rep=observational_count_rep,
        positive_control_count_rep=positive_count_rep,
        negative_control_count_rep=negative_count_rep,
        explanatory_count_rep=explanatory_count_rep
    )

    return crispr_shrinkage_visualization_input


def visualize_lfc_histogram(crispr_shrinkage_visualization_input: CrisprShrinkageVisualizationInput):
    plt.hist(crispr_shrinkage_visualization_input.negative_control_lfc, density=True, label="Negative")
    plt.hist(crispr_shrinkage_visualization_input.positive_control_lfc, density=True, label="Positive")
    plt.hist(crispr_shrinkage_visualization_input.observational_lfc, density=True, label="Observation")
    plt.title("Adjusted LFC Distribution of Each Set")
    plt.legend()
    plt.show()

def visualize_lfc_by_count(crispr_shrinkage_visualization_input: CrisprShrinkageVisualizationInput):
    for rep_i in crispr_shrinkage_visualization_input.replicate_indices:
        plt.scatter(crispr_shrinkage_visualization_input.observational_count_rep[rep_i], crispr_shrinkage_visualization_input.observational_raw_lfc_rep[rep_i], alpha=0.6, label="observation")
        plt.scatter(crispr_shrinkage_visualization_input.positive_control_count_rep[rep_i], crispr_shrinkage_visualization_input.positive_control_raw_lfc_rep[rep_i], alpha=0.6, label="positive")
        plt.scatter(crispr_shrinkage_visualization_input.negative_control_count_rep[rep_i], crispr_shrinkage_visualization_input.negative_raw_lfc_rep[rep_i], alpha=0.6, label="negative")
        plt.xlabel("Total Normalized Count")
        plt.ylabel("Raw LFC")
        plt.title("Replicate {}".format(rep_i+1))
        plt.legend()
        plt.show()

        plt.scatter(crispr_shrinkage_visualization_input.observational_count_rep[rep_i], crispr_shrinkage_visualization_input.observational_lfc_rep[rep_i], alpha=0.6, label="observation")
        plt.scatter(crispr_shrinkage_visualization_input.positive_control_count_rep[rep_i], crispr_shrinkage_visualization_input.positive_control_lfc_rep[rep_i], alpha=0.6, label="positive")
        plt.scatter(crispr_shrinkage_visualization_input.negative_control_count_rep[rep_i], crispr_shrinkage_visualization_input.negative_control_lfc_rep[rep_i], alpha=0.6, label="negative")
        plt.xlabel("Total Normalized Count")
        plt.ylabel("Adjusted LFC")
        plt.title("Replicate {}".format(rep_i+1))
        plt.legend()
        plt.show()


def visualize_raw_vs_adjusted_score_scatter(crispr_shrinkage_visualization_input: CrisprShrinkageVisualizationInput):
    for rep_i in crispr_shrinkage_visualization_input.replicate_indices:
        plt.scatter(crispr_shrinkage_visualization_input.positive_control_raw_lfc_rep[rep_i], crispr_shrinkage_visualization_input.positive_control_lfc_rep[rep_i], c=crispr_shrinkage_visualization_input.positive_control_count_rep[rep_i], alpha=0.3, marker="o")
        plt.scatter(crispr_shrinkage_visualization_input.negative_control_raw_lfc_rep[rep_i], crispr_shrinkage_visualization_input.negative_control_lfc_rep[rep_i], c=crispr_shrinkage_visualization_input.negative_control_count_rep[rep_i], alpha=0.3, marker="o")
        plt.scatter(crispr_shrinkage_visualization_input.observational_control_raw_lfc_rep[rep_i], crispr_shrinkage_visualization_input.observational_control_lfc_rep[rep_i], c=crispr_shrinkage_visualization_input.observational_control_count_rep[rep_i], alpha=0.3, marker="o")
        plt.colorbar(label="Total Normalized Count")
        plt.xlabel("Raw LFC")
        plt.ylabel("Adjusted LFC")
        plt.title("Replicate {}".format(rep_i + 1))
        plt.show()

def visualize_raw_and_adjusted_score_by_position_scatter(crispr_shrinkage_visualization_input: CrisprShrinkageVisualizationInput):
    for rep_i in crispr_shrinkage_visualization_input.replicate_indices:
        plt.scatter(crispr_shrinkage_visualization_input.explanatory_position, crispr_shrinkage_visualization_input.explanatory_lfc_rep[rep_i], s=crispr_shrinkage_visualization_input.explanatory_count_rep[rep_i])
        plt.title("Adjusted LFC")
        plt.show()
        plt.scatter(crispr_shrinkage_visualization_input.explanatory_position, crispr_shrinkage_visualization_input.explanatory_raw_lfc_rep[rep_i], s=crispr_shrinkage_visualization_input.explanatory_count_rep[rep_i])
        plt.title("Raw LFC")
        plt.show()

def visualize_combined_adjusted_score_by_position_scatter(crispr_shrinkage_visualization_input: CrisprShrinkageVisualizationInput):
    fig = plt.figure(figsize=(20, 4))
    ax = fig.add_subplot(111)

    for rep_i in crispr_shrinkage_visualization_input.replicate_indices:
        ax.scatter(crispr_shrinkage_visualization_input.explanatory_position, crispr_shrinkage_visualization_input.explanatory_lfc_rep[rep_i], s=crispr_shrinkage_visualization_input.explanatory_count_rep[rep_i], alpha=0.3, label="Replicate {}".format(rep_i))
    ax.scatter(crispr_shrinkage_visualization_input.explanatory_position, crispr_shrinkage_visualization_input.explanatory_lfc, marker="s", s=2, label="Combined")
    ax.plot(crispr_shrinkage_visualization_input.explanatory_position, crispr_shrinkage_visualization_input.explanatory_lfc,color="red", alpha=0.2)
    ax.set_title("Region Scores")
    ax.set_xlabel("Coordinate")
    ax.set_ylabel("Adjusted LFC")
    ax.legend()
    plt.show()

def visualize_combined_score_credible_interval_scatter(crispr_shrinkage_visualization_input: CrisprShrinkageVisualizationInput):
    explanatory_lfc_CI_low = np.asarray([CI[0] for CI in crispr_shrinkage_visualization_input.explanatory_lfc_CI])
    explanatory_lfc_CI_up = np.asarray([CI[1] for CI in crispr_shrinkage_visualization_input.explanatory_lfc_CI])
    
    fig = plt.figure(figsize=(20, 4))
    ax = fig.add_subplot(111)

    ax.scatter(crispr_shrinkage_visualization_input.explanatory_position, crispr_shrinkage_visualization_input.explanatory_lfc, marker="s",color="black", s=2, label="Combined")

    ax.errorbar(crispr_shrinkage_visualization_input.explanatory_position, crispr_shrinkage_visualization_input.explanatory_lfc, (crispr_shrinkage_visualization_input.explanatory_lfc-explanatory_lfc_CI_low, explanatory_lfc_CI_up-crispr_shrinkage_visualization_input.explanatory_lfc), solid_capstyle='projecting', capsize=1, alpha=0.3)
    ax.plot(crispr_shrinkage_visualization_input.explanatory_position, crispr_shrinkage_visualization_input.explanatory_lfc,color="red", alpha=0.2)
    ax.set_title("Region Scores")
    ax.set_xlabel("Coordinate")
    ax.set_ylabel("Adjusted LFC")
    ax.legend()
    plt.show()

def visualize_standardized_raw_vs_adjusted_score_by_position_scatter(crispr_shrinkage_visualization_input: CrisprShrinkageVisualizationInput):
    for rep_i in crispr_shrinkage_visualization_input.replicate_indices:
        fig = plt.figure(figsize=(20, 4))
        ax = fig.add_subplot(111)
        explanatory_lfc_rep_rep_standardized = stats.zscore(crispr_shrinkage_visualization_input.explanatory_lfc_rep[rep_i])
        explanatory_raw_lfc_rep_standardized = stats.zscore(crispr_shrinkage_visualization_input.explanatory_raw_lfc_rep[rep_i])

        ax.scatter(crispr_shrinkage_visualization_input.explanatory_position,explanatory_lfc_rep_rep_standardized, s=np.asarray(crispr_shrinkage_visualization_input.explanatory_count_rep[rep_i])*0.1, color="blue", alpha=0.85, label="Adjusted")
        ax.scatter(crispr_shrinkage_visualization_input.explanatory_position,explanatory_raw_lfc_rep_standardized, s=np.asarray(crispr_shrinkage_visualization_input.explanatory_count_rep[rep_i])*0.1, color="black", alpha=0.85, label="Raw") # Can color based on which of the guides are in the observational set and which are not (specifically which are in the positive, negative.
        for guide_i in range(len(crispr_shrinkage_visualization_input.explanatory_position)):
            ax.arrow(crispr_shrinkage_visualization_input.explanatory_position[guide_i], explanatory_raw_lfc_rep_standardized[guide_i],0, explanatory_lfc_rep_rep_standardized[guide_i] - explanatory_raw_lfc_rep_standardized[guide_i], head_width=1, head_length=0.2, alpha=0.4, length_includes_head=True, color="black")

        ax.set_xlabel("Position")
        ax.set_ylabel("Standardized LFC")
        ax.legend()
        ax.set_title("Replicate {}".format(rep_i+1))
        plt.show()

def visualize_standardized_raw_vs_adjusted_score_by_position_scatter(crispr_shrinkage_visualization_input: CrisprShrinkageVisualizationInput):
    explanatory_lfc_CI_low = np.asarray([CI[0] for CI in crispr_shrinkage_visualization_input.explanatory_lfc_CI])
    explanatory_lfc_CI_up = np.asarray([CI[1] for CI in crispr_shrinkage_visualization_input.explanatory_lfc_CI])
    
    
    fig = plt.figure(figsize=(20, 4))
    ax = fig.add_subplot(111)

    for rep_i in crispr_shrinkage_visualization_input.replicate_indices:
        ax.scatter(crispr_shrinkage_visualization_input.explanatory_position, crispr_shrinkage_visualization_input.explanatory_lfc_rep[rep_i], s=np.asarray(crispr_shrinkage_visualization_input.explanatory_count_rep[rep_i])*0.5, alpha=0.3, label="Replicate {}".format(rep_i))
    ax.scatter(crispr_shrinkage_visualization_input.explanatory_position, crispr_shrinkage_visualization_input.explanatory_lfc, marker="s",color="black", s=10, label="Combined")
    ax.errorbar(crispr_shrinkage_visualization_input.explanatory_position, crispr_shrinkage_visualization_input.explanatory_lfc, (crispr_shrinkage_visualization_input.explanatory_lfc-explanatory_lfc_CI_low, explanatory_lfc_CI_up-crispr_shrinkage_visualization_input.explanatory_lfc), solid_capstyle='projecting', capsize=1, alpha=0.3, linestyle='')

    ax.set_title("Region Scores")
    ax.set_xlabel("Coordinate")
    ax.set_ylabel("Adjusted LFC")
    ax.legend()
    plt.show()