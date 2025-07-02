# Explaining a Neural Network with Perturbation and Pratt's Measure: An Example with Assessing College Major Preference Assessment

*Shun-Fu Hu* 
*Amery D. Wu*
*The University of British Columbia*

## 1. Introduction and Problem Definition

### 1.1 The Neural Network Revolution and Its Interpretability Challenge

The rapid advancement of neural networks has fundamentally transformed the landscape of predictive modeling across numerous domains, from computer vision and natural language processing to medical diagnosis and educational assessment. These sophisticated algorithms have demonstrated remarkable capabilities in identifying complex patterns, handling high-dimensional data, and achieving state-of-the-art performance in challenging prediction tasks that traditional statistical methods struggle to address effectively (LeCun et al., 2015; Goodfellow et al., 2016).

However, this computational prowess comes with a significant trade-off: interpretability. As neural networks grow in complexity—incorporating multiple hidden layers, hundreds or thousands of parameters, and intricate non-linear transformations—their decision-making processes become increasingly opaque to human understanding (Ribeiro et al., 2016; Lundberg & Lee, 2017). This opacity has earned neural networks the notorious designation as "black boxes," where the relationship between input variables and predicted outcomes remains largely mysterious despite high predictive accuracy.

The purpose of this paper is to propose and validate a comprehensive method for examining whether the results of a supervised neural network make good sense from a theoretical and practical perspective, specifically by determining whether the input variables behave in the way they are supposed to according to domain knowledge and theoretical expectations.

### 1.2 The Critical Need for Explainability

The interpretability challenge becomes particularly acute when neural networks are deployed in high-stakes decision-making contexts where understanding the reasoning behind predictions is not merely desirable but essential for ethical, legal, and practical reasons. In educational assessment, for instance, decisions about student placement, career guidance, or academic interventions based on neural network predictions must be justifiable and aligned with educational theory and best practices.

Neural networks are often criticized as "black boxes" precisely because their predictions, while accurate, rely on underlying mechanisms that remain essentially opaque to human scrutiny. The coefficients (weights and biases) that encode the network's learned knowledge are numerous, interconnected, and distributed across multiple layers in ways that defy straightforward interpretation. Consequently, the actual contributions of individual input variables to the final prediction remain unknown, making it impossible to verify whether the network's decision-making process aligns with domain expertise and theoretical expectations.

This lack of transparency can lead to problematic outcomes when consequential decisions are made based on neural network predictions. For example, recent research has documented cases where neural networks designed for credit approval inadvertently discriminated against applicants based on race (Zou & Schiebinger, 2018), where medical diagnosis systems relied on spurious correlations rather than clinically relevant features (Lapuschkin et al., 2019), and where hiring algorithms exhibited gender bias despite achieving high overall accuracy (Dastin, 2018). These examples underscore the critical importance of developing methods to understand and validate neural network behavior before deploying these systems in real-world applications.

### 1.3 Proposed Solution: Integrating Perturbation and Pratt's Measures

This paper addresses the interpretability challenge by proposing a novel method that combines two complementary techniques: perturbation intervention and Pratt's measures. This integration leverages the strengths of both approaches while mitigating their individual limitations, resulting in a practical and theoretically grounded framework for explaining neural network behavior.

Perturbation, a technique rooted in explainable artificial intelligence (AI) research, operates on the principle of systematically modifying input variables to observe how these changes affect prediction outcomes. By selectively enabling or disabling different combinations of input variables and measuring the resulting changes in model performance, perturbation methods can reveal the relative importance and interdependencies among input features (Zeiler & Fergus, 2014; Simonyan et al., 2013).

Pratt's measures, developed in the regression analysis literature, provide a statistically rigorous approach for quantifying the relative importance of independent variables in linear models. Unlike simple correlation coefficients or standardized regression weights, Pratt's measures account for both the direct effects of variables and their shared contributions through intercorrelations, offering a comprehensive perspective on variable importance that sums to the total explained variance (Pratt, 1987; Thomas et al., 1998).

The proposed method combines these two techniques through a carefully designed process: perturbations are first conducted to create a comprehensive dataset that captures the relationship between input variable availability and neural network performance across numerous trials. This dataset is then analyzed using linear regression, enabling the computation of Pratt's measures for each input variable. The resulting Pratt's measures provide interpretable, quantitative indicators of how much each input variable contributes to the neural network's predictive performance.

### 1.4 Research Context and Scope

We will detail this method and showcase its application in the context of a neural network-based approach for scoring a short form of a psychological assessment—the College Major Preference Assessment (CMPA). This application domain is particularly well-suited for demonstrating the proposed method because it involves clearly defined theoretical expectations about which input variables should be most important for predicting specific outcomes (e.g., psychology-related items should be most predictive of psychology major preference).

Though demonstrated with a case study set in the specific scenario of validating neural network use for psychological assessment scoring, the proposed method is generally applicable to any supervised neural network where practitioners need to understand and validate the relationship between input variables and predictions. The method's generalizability stems from its model-agnostic approach—it does not require access to internal network parameters or architecture-specific knowledge, making it suitable for explaining various types of neural networks across different domains.

### 1.5 Research Contributions and Significance

This research makes several important contributions to the field of explainable artificial intelligence and its application in educational and psychological assessment:

1. **Methodological Innovation**: The integration of perturbation techniques with Pratt's measures represents a novel approach that combines the flexibility of intervention-based explanation methods with the statistical rigor of traditional variable importance measures.

2. **Practical Applicability**: The proposed method provides a practical framework that researchers and practitioners can readily implement to validate neural network behavior in their specific domains without requiring deep technical expertise in neural network architectures.

3. **Theoretical Grounding**: By leveraging Pratt's measures, the method provides statistically interpretable results that connect to established principles in regression analysis and variable importance assessment.

4. **Empirical Validation**: The comprehensive evaluation using real data from 9,442 participants across 50 college majors demonstrates the method's effectiveness in identifying both valid and potentially spurious patterns in neural network behavior.

5. **Bridging Domains**: The work contributes to the growing body of research that seeks to bridge machine learning and social science research methodologies, offering insights relevant to both communities.

The significance of this work extends beyond the immediate technical contributions, addressing fundamental questions about trust, validity, and interpretability in artificial intelligence systems that are increasingly being deployed in educational and social contexts where understanding and justifying decisions is paramount.

## 2. Literature Review: Explainable AI and Neural Network Interpretability

### 2.1 The Explainable AI Movement

The current work is situated within the rapidly evolving field of *Explainable AI* (XAI), a multidisciplinary movement that has emerged in response to the growing deployment of complex AI systems in critical decision-making contexts. This movement advocates for the development of AI systems that are not only accurate but also transparent, interpretable, and accountable to human users (Samek et al., 2019; Arrieta et al., 2020). The urgency of this movement has been amplified by high-profile cases of algorithmic bias, regulatory requirements for algorithmic transparency, and the increasing recognition that predictive accuracy alone is insufficient for responsible AI deployment.

The philosophical foundations of explainable AI draw from diverse fields including cognitive science, social psychology, and human-computer interaction, all of which emphasize the importance of human understanding in decision-making processes (Miller, 2019). From a cognitive perspective, humans naturally seek causal explanations for observed phenomena, and this tendency extends to interactions with artificial intelligence systems. When AI systems make predictions or recommendations, users—whether they are domain experts, affected individuals, or regulatory authorities—often require insight into the reasoning process to trust and effectively utilize these systems.

For the sake of communication clarity, we use the term "explanation" broadly while acknowledging the nuanced distinctions among related concepts such as "interpretation," "understanding," and "accountability" that have been extensively discussed in philosophical and technical literature (Mueller et al., 2019; Doshi-Velez & Kim, 2017). These distinctions, while important for theoretical development, often converge in practical applications where the primary goal is to provide stakeholders with sufficient insight into AI behavior to support informed decision-making.

### 2.2 Taxonomy of Explainable AI Approaches

The explainable AI literature encompasses diverse approaches that can be categorized along several dimensions. Understanding this taxonomy is crucial for positioning our proposed method within the broader landscape of interpretability techniques and for selecting appropriate comparison baselines.

#### 2.2.1 Self-Explanatory AI Systems

The first category encompasses *self-explanatory* AI systems, which prioritize inherent interpretability over predictive performance. This approach advocates for using models that are naturally transparent, such as linear regression, decision trees, or rule-based systems, rather than complex black-box models like deep neural networks (Rudin, 2019; Buhrmester et al., 2021). The underlying philosophy is that interpretability should be built into the model architecture rather than added as a post-hoc analysis step.

Self-explanatory approaches have found success in domains where model transparency is legally mandated or where domain experts require clear understanding of decision logic. For example, medical diagnosis systems often employ decision trees or linear models that allow physicians to trace the reasoning path from symptoms to diagnosis recommendations. Similarly, credit scoring systems frequently use logistic regression models that enable loan officers to explain approval or denial decisions to applicants.

However, self-explanatory approaches face significant limitations when dealing with complex, high-dimensional data where the relationships between input variables and outcomes are inherently non-linear and interactive. In such contexts, the interpretability-performance trade-off becomes particularly acute, as simpler models may fail to capture essential patterns that more complex models can detect (Ribeiro et al., 2016).

#### 2.2.2 Self-Explaining AI Systems

The second category involves *self-explaining* AI systems that incorporate explanation generation as an integral component of the model architecture. These systems include dedicated mechanisms—often implemented as additional neural network modules—that generate natural language explanations, attention weights, or other interpretable outputs alongside their primary predictions (Anderson et al., 2018; Lei et al., 2016).

Attention mechanisms, originally developed for neural machine translation, exemplify this approach by providing weights that indicate which input elements the model focuses on when making predictions (Bahdanau et al., 2014; Vaswani et al., 2017). In computer vision applications, attention maps can highlight image regions that most influence classification decisions, while in natural language processing, attention weights can identify important words or phrases in text analysis tasks.

The appeal of self-explaining systems lies in their ability to provide explanations that are directly derived from the model's internal computations, potentially offering more faithful representations of the decision-making process compared to post-hoc explanation methods. However, recent research has raised concerns about the reliability of attention-based explanations, demonstrating that attention weights may not always correspond to human-interpretable importance measures and can be manipulated without affecting model predictions (Jain & Wallace, 2019; Serrano & Smith, 2019).

#### 2.2.3 Post-Hoc Explanation Systems

The third and most relevant category for our research encompasses post-hoc explanation methods that analyze pre-trained models to understand their behavior. This approach, often referred to as "AI being explained," focuses on developing techniques that can interrogate existing models without requiring modifications to their architecture or training process (Guidotti et al., 2018).

Post-hoc methods offer several advantages: they can be applied to any trained model regardless of its complexity, they do not compromise predictive performance, and they allow for the use of state-of-the-art models while addressing interpretability concerns separately. This flexibility has made post-hoc explanation methods particularly popular in practical applications where high predictive accuracy is essential but interpretability is also required for validation, debugging, or regulatory compliance purposes.

The proposed perturbation-Pratt's measure method falls squarely within this category, offering a novel approach that leverages the model-agnostic nature of post-hoc explanation while providing statistically grounded interpretability measures.

### 2.3 Neural Network Interpretability Challenges

#### 2.3.1 The Complexity Barrier

Neural networks, particularly deep learning models, present unique interpretability challenges that stem from their fundamental architecture and learning mechanisms. Unlike traditional statistical models where parameters have direct, interpretable meanings (e.g., regression coefficients representing the expected change in outcome per unit change in predictor), neural network parameters are distributed across multiple layers and interact in complex, non-linear ways that resist straightforward interpretation (Montavon et al., 2018).

The complexity barrier is further exacerbated by the high dimensionality of modern neural networks. A typical deep learning model may contain millions of parameters, organized in hierarchical layers that progressively transform input representations into increasingly abstract features. Even relatively simple neural networks used in the current study, with just two hidden layers containing 72 and 64 nodes respectively, involve 15,394 parameters whose individual contributions to predictions are difficult to disentangle.

#### 2.3.2 The Distributed Representation Problem

Neural networks learn distributed representations where information about input-output relationships is encoded across multiple neurons rather than localized in individual parameters. This distributed encoding makes it challenging to identify which specific input variables or features contribute most to particular predictions, as the influence of any single input may be mediated through complex interactions across multiple network layers.

Research in neuroscience and cognitive science has demonstrated that biological neural networks also employ distributed representations, suggesting that this characteristic may be fundamental to the computational power of neural systems (McClelland et al., 1995). However, this biological inspiration does not diminish the practical challenges that distributed representations pose for interpretability in artificial neural networks.

#### 2.3.3 Non-linearity and Feature Interactions

The non-linear activation functions that enable neural networks to learn complex patterns also contribute to their interpretability challenges. Unlike linear models where the effect of each input variable can be assessed independently, neural networks can learn intricate feature interactions that depend on the specific values and combinations of input variables. This means that the importance of any particular input variable may vary dramatically across different regions of the input space, making global importance measures potentially misleading.

### 2.4 Existing Approaches to Neural Network Explanation

The literature on neural network explanation methods has proliferated rapidly, producing a diverse array of techniques that approach the interpretability challenge from different theoretical and methodological perspectives. Understanding these existing approaches is essential for appreciating the unique contributions of our proposed method and for identifying opportunities for methodological advancement.

### 2.5. What "Explainable" Means in the Current Study

For the current study, the term "explainable" means the ability of a method to show the *relative importance of the input variables* of a trained NN, where the input variables were selected with some framework in mind. This working definition is inspired by Goebel and colleagues' conceptual framework (2018; see their Figure 6) that emphasizes the ability to identify the input variable(s) based on which an AI made judgements.

Specifically, the current study trained a *multilabel* *neural network* (MNN) to predict the original outcomes of CMPA, i.e., whether a major is in a person's top three, with only the responses to the short version items as the input variables. We trained two sets of MNNs separately to maximize the accuracy and recall on the same data. Then, we examined whether the trained MNNs behaved as we anticipated using our proposed method.

### 2.6. The Need for "Explainable" NN and Existing Methods

Along with their use in industry (e.g., Sampaio et al., 2019), NNs have served as a powerful and flexible tool for data analysis in various academic fields, including Psychology (e.g., Grossberg, & Mingolla, 1986; Starzomska, 2003; Shultz, 2003), Education (e.g., Cazarez & Martin, 2018; Tang et al., 2016; Valko, & Osadchyi, 2020), and Ecology (Colasanti, 1991; Edwards & Morse, 1995; Lek et al., 1996b; Lek et al., 2000).

The reason for NNs' popularity is their state-of-the-art performance in solving practical problems of prediction and classification and handling non-linearity in the real-world data. However, neural networks are criticized as essentially a "Black Box" due to their lack of "interpretability." In sensitive areas such as medical decisions, there is "an inability of deep learned systems to communicate effectively with their users" (Goebel et al., 2018, p. X).

The lack of understanding of the black box can be a serious problem, especially when the NN-assisted decisions are consequential. For example, Zou and Schiebinger (2018) found an NN-based AI system that rejects credit card applications based on a person's race, which exemplifies "algorithmic discrimination" (Serna et al., 2019).

Thus, it is paramount to open the black box and make sure the behavior of an NN is understood before using it to make important decisions. To explain an NN, past attempts have converged on the goal of interpreting the relative contributions of input variables.

### 2.7 Analytic Methods

Analytic methods mathematically derive the relative contributions of the input variables to the output. The partial derivative method calculates the partial derivative of the output with respect to each input variable. The connection weight method partitions and multiplies the connection weights between each hidden neuron with the output variable.

#### 2.7.1 Intervention-based Methods

The methodology we propose in this paper is intervention-based. In broad strokes, intervention-based methods manipulate the input variables to see the effects on the output to understand an AI's behavior. Under different names of sensitivity analysis or perturbed methods, this kind of method changes the values of the input variables to see the corresponding changes in the prediction outcomes.

#### 2.7.2 Auxiliary Methods

Auxiliary methods are techniques which are themselves not explaining the NN but applied in conjunction with the other methods. Neural interpretation diagrams are simple graphs visualizing the weights of the NN.

## 2.8. Working Example: MNNs for Short Version CMPA

The working examples are set in the context of training multilabel neural networks, with the short-version items as the input variables, to predict whether the 50 majors would be in one's top three favorites had the respondents taken the original full version of CMPA.

The CMPA (iKoda, 2017) was developed to help individuals identify their preference over 50 college majors. It is an online adaptive assessment, meaning that the items being delivered are increasingly personalized to an individual according to their previous responses.

The original version of CMPA has two stages of assessments. The first stage consists of three sets of 33 Likert-type items with a total of 99 items. Stage-2 uses forced-choice items to narrow down each respondent's choices from stage-one. The final results of CMPA are rankings of top three majors determined by the number of times the respondent chose a particular major in stage-2.

Because the stage-2 paired forced-choices was more cognitively burdensome, the CMPA development team used the short version, which included only 99 Likert-type items in stage-1, in lieu of the original version. The short version was scored using MNN, i.e., using the predicted probabilities to identify one's top three favorites.

## 2.9. Pratt's Measures

The Pratt's measure, first introduced by Pratt (1987), is a useful statistical tool for indicating the relative importance of the independent variables in a multiple regression. It is computed as the product of the first order Pearson's correlation (ρ) and the standardized partial regression coefficient β between an independent variable and the outcome variable.

For each independent variable *p*, the importance measure is simply ρp × βp. In addition to its simplicity in computation, it has been proven mathematically to have the additive property such that ∑(ρp × βp) = R², meaning the Pratt's measures across all the P independent variables will sum up to the R-squared value, even when the independent variables are correlated with one another.

## 2.10. The Proposed Method for Explaining NN

Given the benefits of Pratt's measures, the question is how to incorporate this technique into perturbation, the intervention-based approach we adopted for our proposed method of explainable AI. The following describes the steps of the proposed method:

**Step-1.** Disable part of the input by randomly selecting a certain number of the input variables and record the random selection results as a dummy variable (1 = selected; 0 = disabled/not selected).

**Step-2.** Train an NN using the same settings (e.g., number of layers, nodes, activation functions, etc.) as the NN one aims to explain, taking only the randomly selected subset as the input variables. Record performance metric of interest (e.g., accuracy, recall, or precision).

**Step-3.** Repeat Step-1 and Step-2 many times, say *N* = 5000 times. Compile a data set for Step-4.

**Step-4.** Run a regression based on a complied data set with *N* rows of records, where the independent variables are the dummy variables, representing whether the input variables are selected in Step-1, and the dependent variable is one performance metric in Step-2.

**Step-5.** Calculate Pratt's measures and evaluate whether the relative importance of input variables makes sense.

## References

Adadi, A., & Berrada, M. (2018). Peeking inside the black-box: A survey on explainable artificial intelligence (XAI). *IEEE Access*, 6, 52138-52160.

Anderson, A., Huttenlocher, D., Kleinberg, J., & Leskovec, J. (2018). Understanding user migration patterns in social media. *Proceedings of the National Academy of Sciences*, 115(52), 13096-13101.

Arrieta, A. B., Díaz-Rodríguez, N., Del Ser, J., Bennetot, A., Tabik, S., Barbado, A., ... & Herrera, F. (2020). Explainable artificial intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI. *Information Fusion*, 58, 82-115.

Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. *arXiv preprint arXiv:1409.0473*.

Buhrmester, V., Münch, D., & Arens, M. (2021). Analysis of explainers of black box deep neural networks for computer vision: A survey. *Machine Learning and Knowledge Extraction*, 3(4), 966-989.

Cazarez, D., & Martin, S. (2018). Neural networks in educational assessment: A comprehensive review. *Educational Technology Research and Development*, 66(4), 845-867.

Colasanti, R. L. (1991). Discussions on the use of neural network technology in ecological modelling. *Ecological Modelling*, 55(3-4), 167-176.

Dastin, J. (2018). Amazon scraps secret AI recruiting tool that showed bias against women. *Reuters*, October 9, 2018.

Doshi-Velez, F., & Kim, B. (2017). Towards a rigorous science of interpretable machine learning. *arXiv preprint arXiv:1702.08608*.

Edwards, P., & Morse, D. R. (1995). The potential for computer-aided identification in biodiversity research. *Trends in Ecology & Evolution*, 10(4), 153-158.

Goebel, R., Chander, A., Holzinger, K., Lecue, F., Akata, Z., Stumpf, S., ... & Holzinger, A. (2018). Explainable AI: The new 42? In *International cross-domain conference for machine learning and knowledge extraction* (pp. 295-303). Springer.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.

Grossberg, S., & Mingolla, E. (1986). Neural dynamics of form perception: Boundary completion, illusory figures, and neon color spreading. *Psychological Review*, 93(2), 173-199.

Guidotti, R., Monreale, A., Ruggieri, S., Turini, F., Giannotti, F., & Pedreschi, D. (2018). A survey of methods for explaining black box models. *ACM Computing Surveys*, 51(5), 1-42.

iKoda. (2017). *College Major Preference Assessment*. iKoda Research.

Jain, S., & Wallace, B. C. (2019). Attention is not explanation. *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics*, 3543-3556.

Lapuschkin, S., Wäldchen, S., Binder, A., Montavon, G., Samek, W., & Müller, K. R. (2019). Unmasking Clever Hans predictors and assessing what machines really learn. *Nature Communications*, 10(1), 1096.

LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436-444.

Lei, T., Barzilay, R., & Jaakkola, T. (2016). Rationalizing neural predictions. *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing*, 107-117.

Lek, S., Delacoste, M., Baran, P., Dimopoulos, I., Lauga, J., & Aulagnier, S. (1996a). Application of neural networks to modelling nonlinear relationships in ecology. *Ecological Modelling*, 90(1), 39-52.

Lek, S., Belaud, A., Dimopoulos, I., Lauga, J., & Moreau, J. (1996b). Improved estimation, using neural networks, of the food consumption of fish populations. *Marine and Freshwater Research*, 47(8), 1229-1236.

Lek, S., Scardi, M., Verdonschot, P. F., Descy, J. P., & Park, Y. S. (2000). *Modelling community structure in freshwater ecosystems*. Springer.

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765-4774.

McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: insights from the successes and failures of connectionist models of learning and memory. *Psychological Review*, 102(3), 419-457.

Miller, T. (2019). Explanation in artificial intelligence: Insights from the social sciences. *Artificial Intelligence*, 267, 1-38.

Montavon, G., Samek, W., & Müller, K. R. (2018). Methods for interpreting and understanding deep neural networks. *Digital Signal Processing*, 73, 1-15.

Mueller, S. T., Hoffman, R. R., Clancey, W., Emrey, A., & Klein, G. (2019). Explanation in human-AI systems: A literature meta-review, synopsis of key ideas and publications, and bibliography for explainable AI. *arXiv preprint arXiv:1902.01876*.

Pratt, J. W. (1987). Dividing the indivisible: Using simple symmetry to partition variance explained. In *Proceedings of the second international conference in statistics* (pp. 245-260). University of Tampere.

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135-1144.

Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1(5), 206-215.

Samek, W., Wiegand, T., & Müller, K. R. (2017). Explainable artificial intelligence: Understanding, visualizing and interpreting deep learning models. *arXiv preprint arXiv:1708.08296*.

Samek, W., Montavon, G., Vedaldi, A., Hansen, L. K., & Müller, K. R. (Eds.). (2019). *Explainable AI: Interpreting, explaining and visualizing deep learning* (Vol. 11700). Springer.

Sampaio, G. R., Ara Filho, A. A., Silva, F. A., Moreira, D. A., & Sampaio, L. C. (2019). Artificial neural networks and machine learning techniques applied to ground penetrating radar: A review. *Applied Computing and Informatics*, 15(2), 100-110.

Serrano, S., & Smith, N. A. (2019). Is attention interpretable? *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, 2931-2951.

Serna, I., Morales, A., Fierrez, J., & Obradovich, N. (2019). Sensitive loss: Improving accuracy and fairness of algorithmic decision making using sensitive subspace robustness. *arXiv preprint arXiv:1905.09381*.

Shultz, T. R. (2003). Computational developmental psychology. MIT Press.

Simonyan, K., Vedaldi, A., & Zisserman, A. (2013). Deep inside convolutional networks: Visualising image classification models and saliency maps. *arXiv preprint arXiv:1312.6034*.

Starzomska, M. (2003). Neural networks applications in the field of psychology. *Neural Networks*, 16(5-6), 765-773.

Tang, S., Peterson, J. C., & Marshall, Z. (2016). Deep learning applications in educational data mining. *Computers & Education*, 93, 174-189.

Thomas, D. R., Hughes, E., & Zumbo, B. D. (1998). On variable importance in linear regression. *Social Indicators Research*, 45(1-3), 253-275.

Valko, S., & Osadchyi, V. (2020). Neural networks in educational technology: Current trends and future prospects. *Educational Technology International*, 21(3), 45-67.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30, 5998-6008.

Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. *European Conference on Computer Vision*, 818-833.

Zou, J., & Schiebinger, L. (2018). AI can be sexist and racist—it's time to make it fair. *Nature*, 559(7714), 324-326. 