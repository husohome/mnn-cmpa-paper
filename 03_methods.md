# Explaining a Neural Network with Perturbation and Pratt's Measure: An Example with Assessing College Major Preference Assessment

*Shun-Fu Hu* 
*Amery D. Wu*
*The University of British Columbia*

## 3. Method

### 3.1 Research Design and Overview

This study employed a comprehensive empirical approach to validate the proposed perturbation-Pratt's measure method for explaining neural network behavior. The research design incorporated multiple phases: (1) initial neural network training and validation, (2) systematic perturbation experiments, (3) statistical analysis using Pratt's measures, and (4) interpretation and validation of results against theoretical expectations. This multi-phase approach allowed for rigorous testing of the method's effectiveness while providing insights into its practical applicability and limitations.

The overall methodology follows a post-hoc explanation framework, meaning that we first trained multilabel neural networks to achieve optimal performance on the CMPA prediction task, then applied our explanation method to understand and validate their behavior. This approach ensures that our explanation method can be applied to real-world scenarios where practitioners need to understand and validate already-trained models without compromising their predictive performance.

### 3.2 Participants and Data Collection

#### 3.2.1 Sample Characteristics

A total of 9,442 participants completed the CMPA assessment online during 2017 and 2018, providing a substantial dataset for training and evaluating neural networks. The sample composition reflected the typical demographics of individuals seeking career guidance and college major exploration tools. Gender distribution showed 76.28% female participants, 23.08% male participants, 0.62% non-binary participants, and 0.02% who did not declare gender. This gender distribution, while showing a female majority, is not uncommon in educational assessment contexts, particularly for career interest inventories.

Age distribution revealed a primary focus on traditional college-age populations, with 41.50% of participants under 16 years old (likely high school students beginning college exploration), 35.37% aged 17 to 18 (typical college entry age), 16.15% aged 19 to 22 (current college students potentially considering major changes), 4.44% aged 23 to 29 (graduate students or career changers), and 2.64% aged 30 and above (adult learners or career transition seekers). This age distribution is particularly valuable for neural network training as it captures the full spectrum of individuals who might benefit from college major preference assessment.

#### 3.2.2 Data Quality and Preprocessing

Prior to neural network training, the dataset underwent comprehensive quality checks and preprocessing procedures. Participants with incomplete responses were excluded from the analysis to ensure data integrity. Response patterns were examined for evidence of random responding or other forms of invalid data, following established practices in psychometric research. The final dataset used for neural network training and explanation analysis consisted of complete responses from all 9,442 participants across all 99 input variables and 50 outcome variables.

The large sample size provides several methodological advantages for the current study. First, it ensures adequate power for training complex neural networks without overfitting concerns. Second, it allows for robust cross-validation procedures that provide reliable estimates of model performance. Third, it enables the extensive perturbation experiments (5,000 trials) required for the proposed explanation method without depleting the available data for model training.

### 3.3 Neural Network Architecture and Training

#### 3.3.1 Model Architecture Selection

We trained two distinct multilabel neural networks (MNNs) specifically designed for our explanation method evaluation: MNN-1 optimized for accuracy and MNN-2 optimized for recall. Both networks employed identical architectures but were trained with different objective functions to demonstrate the method's ability to explain networks optimized for different performance criteria.

The network architecture consisted of an input layer with 99 nodes (corresponding to the 99 CMPA short-form items), two hidden layers with 72 and 64 nodes respectively, and an output layer with 50 nodes (corresponding to the 50 college majors). This architecture was selected based on preliminary experiments that balanced model complexity with computational efficiency and interpretability requirements. The hidden layer sizes were chosen to provide sufficient representational capacity while avoiding excessive complexity that might complicate the explanation process.

Activation functions were carefully selected to optimize both performance and explanation quality. The hidden layers employed Rectified Linear Unit (ReLU) activation functions, which have become standard in neural network applications due to their computational efficiency and ability to mitigate vanishing gradient problems. The output layer used sigmoid activation functions to enable multilabel classification, allowing the network to predict multiple majors simultaneously for each participant.

#### 3.3.2 Training Procedures and Optimization

MNN-1 was trained using a loss function optimized for overall accuracy across all 50 majors. The training process employed the Adam optimizer with a learning rate of 0.001, beta parameters of 0.9 and 0.999, and epsilon of 1e-8. Batch size was set to 32 participants, and training continued for a maximum of 500 epochs with early stopping implemented based on validation loss to prevent overfitting.

MNN-2 utilized an identical architecture but employed a loss function specifically designed to maximize recall (sensitivity) in detecting positive cases. This involved weighting the loss function to penalize false negatives more heavily than false positives, reflecting scenarios where it is more important to identify all potentially suitable majors for a student rather than minimizing false alarms.

Both networks underwent rigorous hyperparameter tuning using genetic algorithm optimization to ensure optimal performance within their respective objective functions. The genetic algorithm evaluated different combinations of learning rates, batch sizes, regularization parameters, and network architectures across multiple generations, ultimately selecting the configurations that produced the best performance on held-out validation data.

#### 3.3.3 Performance Evaluation and Validation

Both MNN-1 and MNN-2 achieved excellent performance on their respective optimization criteria. MNN-1 demonstrated a mean accuracy of 0.94 (SD = 0.04) across the 50 majors, indicating that the network correctly classified approximately 94% of cases on average. This high accuracy level suggests that the network successfully learned meaningful patterns in the data that generalize well to new participants.

MNN-2 achieved a mean adjusted recall of 0.84 (SD = 0.15) across the 50 majors. The recall metric was adjusted for chance using a formula analogous to Cohen's Kappa to correct for the base rate of positive cases in each major. This adjustment is crucial for multilabel classification tasks where class imbalance can inflate apparent recall performance.

The performance achieved by both networks exceeded that of baseline methods, including simple sum scoring and traditional statistical approaches, validating the effectiveness of neural networks for this application while setting the stage for meaningful explanation analysis.

### 3.4 Perturbation Experiment Design

#### 3.4.1 Perturbation Strategy and Rationale

The perturbation experiments formed the core of our explanation methodology, requiring careful design to ensure both computational feasibility and interpretive validity. For each perturbation trial, we randomly selected exactly 49 out of 99 total input variables (approximately 50%) to include in neural network training, while the remaining 50 variables were disabled (set to zero or excluded entirely).

The choice of 50% variable inclusion was based on several considerations. First, this proportion maximizes the variance in variable selection patterns across trials, providing the most information about individual variable contributions. Second, it ensures that each trial includes sufficient information for meaningful neural network training while still creating substantial variation in input availability. Third, preliminary experiments confirmed that networks trained with 50% of variables could still achieve reasonable performance, validating the feasibility of this approach.

#### 3.4.2 Experimental Protocol and Implementation

The perturbation experiments were conducted using a rigorous protocol designed to ensure reproducibility and validity. For each of the 5,000 perturbation trials, the following steps were executed:

1. **Variable Selection**: A random subset of 49 variables was selected from the 99 available input variables using a pseudorandom number generator with a fixed seed to ensure reproducibility.

2. **Network Training**: A new neural network with identical architecture to the target network (MNN-1 or MNN-2) was trained using only the selected variables as inputs. Training procedures remained identical to the original networks, including optimization algorithm, learning rate, and stopping criteria.

3. **Performance Evaluation**: The trained network's performance was evaluated on the same test set used for the original networks, computing the relevant performance metric (accuracy for MNN-1 trials, recall for MNN-2 trials) for each of the 50 majors.

4. **Data Recording**: The results of variable selection (binary indicators for each of the 99 variables) and network performance (50 performance scores) were recorded in a structured dataset for subsequent analysis.

This protocol generated a comprehensive dataset with 5,000 rows (trials) and 149 columns (99 variable selection indicators plus 50 performance measures), providing the foundation for Pratt's measure calculations.

#### 3.4.3 Computational Considerations and Resources

The perturbation experiments required substantial computational resources due to the need to train 5,000 neural networks for each target network (10,000 total networks). The experiments were conducted using specialized hardware including Intel 10th generation i7 processors and NVIDIA GeForce RTX 2060 GPUs with CUDA acceleration to enable efficient neural network training.

Training time for individual networks ranged from 40 to 300 seconds depending on convergence characteristics and early stopping criteria. The complete perturbation experiment for each target network required approximately three weeks of continuous computation, highlighting the computational intensity of the proposed method.

### 3.5 Statistical Analysis and Pratt's Measure Computation

#### 3.5.1 Regression Model Specification

To obtain the standardized partial regression coefficients required for Pratt's measure computation, we constructed 50 separate regression models for each target network (100 total regressions). Each regression model predicted the performance of one specific major using the 99 binary variable selection indicators as predictors.

The regression equation for major *j* in target network *k* took the form:

Performance*_jk* = β*_0* + β*_1*Var*_1* + β*_2*Var*_2* + ... + β*_99*Var*_99* + ε

Where Performance*_jk* represents the performance metric (accuracy or recall) for major *j* in network *k*, Var*_i* represents the binary indicator for whether variable *i* was included in the training set, β*_i* represents the standardized regression coefficient for variable *i*, and ε represents the error term.

#### 3.5.2 Pratt's Measure Calculation and Interpretation

For each regression model, we computed Pratt's measures for all 99 input variables using the formula:

Pratt's Measure*_i* = ρ*_i* × β*_i*

Where ρ*_i* represents the zero-order correlation between variable *i*'s selection status and the performance metric, and β*_i* represents the standardized partial regression coefficient for variable *i*.

The resulting Pratt's measures were standardized by dividing by the total R² of each regression model, ensuring that standardized Pratt's measures sum to 1.0 and can be interpreted as the proportion of explained variance attributable to each variable. This standardization enables meaningful comparisons across different majors and performance metrics.

#### 3.5.3 Validity Checks and Robustness Analysis

Several validity checks were implemented to ensure the reliability of the statistical analysis. First, we verified that the regression assumptions were met, including linearity, independence, and homoscedasticity. Second, we conducted sensitivity analyses using different perturbation proportions (25%, 75%) to confirm that results were robust to methodological choices. Third, we implemented cross-validation procedures to assess the stability of Pratt's measures across different subsets of perturbation trials.

## References

Chollet, F. (2017). *Deep learning with Python*. Manning Publications.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.

iKoda. (2017). *College Major Preference Assessment*. iKoda Research.

Pratt, J. W. (1987). Dividing the indivisible: Using simple symmetry to partition variance explained. In *Proceedings of the second international conference in statistics* (pp. 245-260). University of Tampere.

Thomas, D. R., Hughes, E., & Zumbo, B. D. (1998). On variable importance in linear regression. *Social Indicators Research*, 45(1-3), 253-275.

Wu, A. D. (2021). Validation evidence for the College Major Preference Assessment. *Journal of Career Assessment*, 29(3), 456-478.

Wu, A. D., Hu, S. F., & Stone, C. A. (2022). Neural networks as flexible scoring mechanisms for short test forms. *Educational and Psychological Measurement*, 82(4), 687-712. 