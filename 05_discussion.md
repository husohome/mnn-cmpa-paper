# Explaining a Neural Network with Perturbation and Pratt's Measure: An Example with Assessing College Major Preference Assessment

*Shun-Fu Hu* 
*Amery D. Wu*
*The University of British Columbia*

## 5. Discussion

### 5.1 Principal Findings and Their Significance

This study proposed and validated a novel explainable AI method that combines perturbation techniques with Pratt's measures to provide statistically grounded insights into neural network behavior. The comprehensive evaluation using the College Major Preference Assessment (CMPA) with 9,442 participants across 50 college majors demonstrates the method's effectiveness in differentiating between theoretically valid and potentially spurious neural network decision-making patterns.

The principal finding that emerges from this research is that high predictive accuracy alone is insufficient for validating neural network behavior in educational and psychological assessment contexts. The proposed method successfully identified instances where networks achieved impressive accuracy (e.g., 95% for Gender Studies, 98% for Statistics) while relying on theoretically inappropriate or diffuse patterns of variable importance. This discovery has profound implications for the responsible deployment of neural networks in high-stakes educational contexts where understanding the reasoning behind predictions is essential for validity and interpretability.

The method's ability to distinguish between valid and spurious patterns represents a significant advance in explainable AI for educational applications. Unlike existing explanation methods that often provide technical insights primarily useful to machine learning experts, the proposed approach generates results that are directly interpretable by domain experts in education and psychology who may lack deep technical expertise in neural network architectures.

### 5.2 Methodological Contributions to Explainable AI

#### 5.2.1 Integration of Classical Statistics with Modern AI

The proposed method represents a novel bridge between classical statistical techniques and modern explainable AI approaches. By leveraging Pratt's measures—a well-established tool in regression analysis—the method provides results that connect to familiar statistical concepts while addressing the interpretability challenges of neural networks. This integration offers several advantages over purely machine learning-based explanation methods.

First, the statistical foundation provides theoretical guarantees about the interpretation of results. Pratt's measures have well-understood mathematical properties, including the crucial additive property that ensures all measures sum to the total explained variance. This characteristic enables practitioners to assess not only which variables are important but also how much of the total predictive power can be attributed to theoretically expected variables versus potentially spurious correlations.

Second, the method produces results that are immediately interpretable by researchers trained in traditional statistical methods, reducing the barrier to adoption in fields where statistical literacy is more common than machine learning expertise. The proportion of variance explained by each variable provides an intuitive metric that domain experts can readily understand and evaluate against their theoretical expectations.

#### 5.2.2 Advantages Over Existing Perturbation Methods

The proposed approach differs fundamentally from typical perturbation methods in ways that address several limitations of existing techniques. Traditional perturbation methods manipulate the values of input variables, often replacing them with random values, mean values, or values sampled from other instances. This approach can produce unrealistic or impossible input combinations that may not reflect how the model would behave with naturally occurring data.

In contrast, our method disables entire variables rather than manipulating their values, ensuring that all analysis is conducted with realistic data patterns. When variables are included in the analysis, they retain their original values from real participants, maintaining the natural covariance structure and realistic value ranges that characterize the domain. This approach provides more ecologically valid insights into how the neural network actually processes real-world data.

Furthermore, the perturbation-retraining approach captures the neural network's learning process rather than just its final state. By training new networks with different variable combinations, the method reveals which variables are truly necessary for the network to discover meaningful patterns, rather than simply which variables the pre-trained network appears to weight heavily in its final form.

#### 5.2.3 Model-Agnostic Applicability

A significant advantage of the proposed method is its model-agnostic nature. The approach requires no knowledge of internal network architecture, no access to gradients or weights, and no modifications to the training process. This characteristic makes the method broadly applicable across different neural network architectures and easily adaptable to emerging modeling approaches.

The model-agnostic property is particularly valuable in practical applications where practitioners may need to explain models developed by others or where the internal architecture of deployed models may be proprietary or complex. The method's ability to provide meaningful insights without requiring specialized knowledge of the underlying model makes it accessible to a broader range of users and applications.

### 5.3 Theoretical Implications for Educational Assessment

#### 5.3.1 Validity Evidence in Neural Network-Based Assessment

The findings contribute significantly to the ongoing discussion about validity evidence in educational and psychological assessment, particularly as these fields increasingly incorporate machine learning approaches. Traditional psychometric theory emphasizes the importance of construct validity—ensuring that assessments measure what they purport to measure—but provides limited guidance for evaluating this concept in the context of neural networks.

The proposed method offers a concrete approach for gathering construct validity evidence for neural network-based assessments. By examining whether networks rely on theoretically appropriate input variables, practitioners can assess whether the networks are likely measuring the intended constructs or are instead capitalizing on irrelevant correlations in the data. This capability addresses a critical gap in the literature on machine learning applications in educational measurement.

The distinction between accuracy-based and validity-based evaluation represents a fundamental shift in how neural networks should be evaluated for educational applications. While predictive accuracy remains important, the current research demonstrates that accuracy alone can be misleading when networks achieve high performance through theoretically inappropriate means. The proposed method provides tools for conducting the more nuanced evaluation required for responsible deployment in educational contexts.

#### 5.3.2 Implications for Automated Scoring and Diagnostic Systems

The research has particular relevance for the growing field of automated scoring and diagnostic systems in education. As neural networks are increasingly deployed for tasks such as essay scoring, automated feedback generation, and learning analytics, understanding whether these systems make decisions based on appropriate criteria becomes paramount.

The method's ability to identify cases where networks achieve high accuracy through potentially spurious means has important implications for fairness and bias in educational AI systems. Networks that rely on unexpected or theoretically inappropriate variables may inadvertently perpetuate existing biases or create new forms of unfairness that are difficult to detect through traditional validation approaches.

### 5.4 Practical Applications and Generalizability

#### 5.4.1 Beyond Educational Assessment

While demonstrated in the context of college major preference assessment, the proposed method has broad applicability across domains where neural networks are used for consequential decision-making. The core principle—validating that networks make decisions based on theoretically appropriate information—is relevant to any application where domain expertise provides clear expectations about which input variables should be most important.

In medical diagnosis, for example, the method could help validate whether diagnostic neural networks rely on clinically relevant symptoms and test results rather than spurious correlations in patient data. In hiring and personnel selection, the approach could identify whether recruitment algorithms base decisions on job-relevant qualifications or potentially discriminatory factors. In financial services, the method could assess whether credit scoring algorithms rely on legitimate financial indicators rather than proxies for protected characteristics.

#### 5.4.2 Implementation Considerations for Practitioners

The practical implementation of the proposed method requires careful consideration of several factors. The computational intensity of training thousands of neural networks represents a significant resource requirement that may limit adoption in some contexts. However, the computational cost should be weighed against the potential consequences of deploying inadequately validated neural networks in high-stakes applications.

For organizations considering implementation, several strategies can help manage computational requirements. First, the method can be applied selectively to the most critical outputs or decisions rather than comprehensively analyzing all aspects of a complex system. Second, the number of perturbation trials can be adjusted based on the stability of results, with fewer trials potentially sufficient for initial validation and more extensive analysis reserved for final validation. Third, the method can be implemented as part of a broader validation framework that includes other, less computationally intensive checks.

### 5.5 Limitations and Considerations

#### 5.5.1 Computational Requirements and Scalability

The most significant limitation of the proposed method is its computational intensity. Training 5,000 neural networks for each target network requires substantial computing resources and time, potentially limiting the method's adoption in resource-constrained environments. The current implementation required approximately three weeks of computation time using specialized hardware, which may not be feasible for all applications.

However, this limitation should be considered in context. The computational cost represents a one-time investment for validating a neural network that may be used to make thousands or millions of consequential decisions. For high-stakes applications where incorrect decisions could have significant negative impacts, the validation cost may be justified by the insights gained and the potential problems avoided.

Future research should investigate methods for reducing computational requirements while maintaining the quality of insights. Possible approaches include more efficient sampling strategies for variable selection, early stopping criteria based on convergence of Pratt's measures, and parallel computing implementations that could reduce wall-clock time.

#### 5.5.2 Dependence on Theoretical Frameworks

The method's effectiveness depends critically on the availability of clear theoretical expectations about which input variables should be most important for specific outcomes. In domains where theory is poorly developed or where multiple competing theoretical frameworks exist, the interpretation of results becomes more challenging.

This limitation is particularly relevant for exploratory applications where researchers may hope to use the method to discover unexpected relationships or generate new theoretical insights. While the method can identify which variables are most important for network performance, it cannot independently determine whether these patterns are theoretically meaningful without external domain expertise.

#### 5.5.3 Assumptions About Linear Relationships

The use of linear regression and Pratt's measures in the final analysis step assumes that the relationship between variable inclusion and network performance can be adequately captured by linear models. While our results suggest that this assumption is reasonable for the current application, it may not hold in all contexts, particularly those involving complex interaction effects or non-linear relationships between variable availability and model performance.

Future extensions of the method could explore non-linear modeling approaches for the final analysis step, potentially using techniques such as random forests or neural networks to model the relationship between variable inclusion and performance. However, such extensions would need to maintain the interpretability advantages of the current approach.

### 5.6 Future Research Directions

#### 5.6.1 Methodological Extensions and Improvements

Several methodological extensions could enhance the proposed approach. First, investigating optimal perturbation strategies beyond random variable selection could potentially improve the efficiency and informativeness of the analysis. Systematic exploration of different inclusion proportions, stratified sampling based on theoretical categories, or adaptive sampling strategies that focus on the most informative variable combinations could yield insights with fewer computational resources.

Second, extending the method to handle different types of neural network architectures, including convolutional neural networks, recurrent neural networks, and transformer-based models, would broaden its applicability. Each architecture type may require adaptations to the perturbation strategy or analysis approach to ensure meaningful results.

Third, developing methods for handling temporal or sequential data could extend the approach to applications such as student learning analytics, where the timing and sequence of information may be as important as the information content itself.

#### 5.6.2 Integration with Other Explainable AI Techniques

Future research should explore how the proposed method can be integrated with other explainable AI techniques to provide more comprehensive insights into neural network behavior. For example, combining perturbation-Pratt's measure analysis with attention visualization, gradient-based explanation methods, or counterfactual explanation techniques could provide multiple perspectives on network behavior that triangulate on a more complete understanding.

Such integration could help address some limitations of individual explanation methods. For instance, gradient-based methods might provide insights into local decision-making patterns that complement the global variable importance patterns revealed by the proposed method.

#### 5.6.3 Applications to Emerging Educational Technologies

The rapid development of AI-powered educational technologies presents numerous opportunities for applying and extending the proposed method. Intelligent tutoring systems, automated essay scoring platforms, adaptive learning systems, and educational recommendation engines all rely on neural networks to make consequential decisions about student learning experiences.

Future research should investigate how the method can be adapted to provide ongoing validation of deployed educational AI systems, potentially enabling real-time monitoring of whether systems continue to make decisions based on appropriate criteria as they encounter new student populations or educational contexts.

### 5.7 Implications for Policy and Practice

#### 5.7.1 Regulatory and Ethical Considerations

The research has important implications for the regulation and governance of AI systems in educational contexts. As regulatory frameworks for AI in education continue to develop, the proposed method could contribute to technical standards for validating neural network behavior. The ability to provide concrete evidence about whether systems make decisions based on appropriate criteria could support compliance with emerging regulations requiring algorithmic transparency and accountability.

Educational institutions and technology providers may need to incorporate similar validation approaches into their development and deployment processes to ensure responsible AI use. The method provides a concrete tool for demonstrating that neural network-based educational systems operate in ways that align with educational theory and best practices.

#### 5.7.2 Professional Development and Training Implications

The successful implementation of methods like the one proposed requires educational professionals who understand both domain expertise and basic principles of machine learning validation. This highlights the need for professional development programs that help educators and educational researchers develop sufficient AI literacy to effectively oversee and validate AI systems in their domains.

Similarly, machine learning practitioners working in educational contexts need sufficient understanding of educational theory and measurement principles to design and validate systems appropriately. The proposed method could serve as a bridge between these communities, providing a technically rigorous but conceptually accessible approach to validation that both groups can understand and use effectively.

## 6. Conclusion

### 6.1 Summary of Contributions

This research addresses one of the most pressing challenges in the application of artificial intelligence to educational and psychological contexts: ensuring that neural networks make decisions for theoretically appropriate reasons rather than achieving high accuracy through spurious correlations. Through the development and validation of a novel perturbation-Pratt's measure method, we have demonstrated a practical approach for explaining and validating neural network behavior that bridges the gap between machine learning performance and domain expertise.

The key contributions of this work can be summarized as follows:

**Methodological Innovation**: The integration of perturbation techniques with Pratt's measures represents a significant methodological advancement in explainable AI. By combining the flexibility of intervention-based explanation methods with the statistical rigor of classical variable importance measures, the proposed approach provides interpretable, quantitative insights into neural network decision-making processes. This methodology offers a unique solution to the longstanding tension between predictive performance and interpretability in machine learning applications.

**Empirical Validation**: The comprehensive evaluation using real-world data from 9,442 participants across 50 college majors provides robust evidence for the method's effectiveness. The analysis successfully differentiated between neural networks that relied on theoretically appropriate variables (such as Psychology items predicting Psychology major preference) and those that achieved high accuracy through potentially spurious means (such as diffuse patterns for Gender Studies prediction). This empirical validation demonstrates the method's practical utility for real-world applications.

**Theoretical Framework**: The research contributes to the theoretical understanding of validity in neural network-based educational assessment. By providing a concrete framework for evaluating whether neural networks make decisions based on construct-relevant information, the study extends traditional psychometric concepts to modern machine learning contexts. This theoretical contribution has implications beyond the immediate application domain, offering insights relevant to any field where understanding the reasoning behind AI decisions is crucial.

**Practical Applicability**: The model-agnostic nature of the proposed method makes it broadly applicable across different neural network architectures and application domains. The approach requires no specialized knowledge of internal network parameters or architecture-specific features, making it accessible to domain experts who may lack deep technical expertise in machine learning. This accessibility is crucial for widespread adoption in educational and psychological research communities.

### 6.2 Implications for Practice

The findings have immediate practical implications for researchers, practitioners, and policymakers involved in the development and deployment of neural network-based systems in educational contexts. Most fundamentally, the research demonstrates that predictive accuracy alone is insufficient for validating AI systems in high-stakes applications. Organizations considering the deployment of neural networks for educational assessment, student placement, or career guidance must implement validation procedures that examine not just whether the systems are accurate, but whether they are accurate for the right reasons.

The proposed method provides a concrete tool for conducting such validation. Educational technology companies can use the approach to verify that their AI systems rely on educationally relevant factors rather than spurious correlations that might lead to unfair or inappropriate decisions. Educational institutions can employ the method to evaluate AI systems before adoption, ensuring that automated decision-making tools align with educational theory and best practices.

For researchers developing neural network applications in psychology and education, the method offers a systematic approach for demonstrating the construct validity of their models. This capability is particularly valuable for publication in venues that require evidence of theoretical grounding and interpretability, not just predictive performance.

### 6.3 Broader Impact on Explainable AI

Beyond its immediate applications in educational assessment, this research contributes to the broader explainable AI movement in several important ways. First, it demonstrates the value of integrating classical statistical techniques with modern explanation methods, showing how established tools from statistics can enhance the interpretability of contemporary machine learning approaches.

Second, the research highlights the importance of domain expertise in interpreting explanation results. While many explainable AI methods focus on generating explanations that are technically sophisticated, the current work emphasizes the need for explanations that are meaningful within specific application domains. The proposed method's ability to connect neural network behavior to theoretical expectations provides a model for developing domain-relevant explanation approaches in other fields.

Third, the work contributes to ongoing discussions about the relationship between accuracy and interpretability in machine learning. Rather than viewing these as necessarily competing objectives, the research suggests that proper validation should examine both predictive performance and the appropriateness of the decision-making process. This perspective has implications for the development of responsible AI systems across all application domains.

### 6.4 Limitations and Future Directions

While the proposed method represents a significant advance in explainable AI for educational applications, several limitations suggest directions for future research. The computational intensity of the approach, requiring the training of thousands of neural networks, may limit its adoption in resource-constrained environments. Future work should investigate more efficient implementation strategies, including parallel computing approaches and early stopping criteria that could reduce computational requirements while maintaining the quality of insights.

The method's reliance on linear regression for the final analysis step assumes that relationships between variable inclusion and performance can be captured by linear models. While this assumption appears reasonable for the current application, future research should explore non-linear alternatives that might capture more complex relationships while maintaining interpretability.

The approach also depends on the availability of clear theoretical expectations about variable importance, limiting its applicability in domains where theory is poorly developed. Future extensions could investigate how the method might be adapted for exploratory applications where the goal is to discover rather than validate theoretical relationships.

### 6.5 Vision for Future Applications

Looking forward, the proposed method and its underlying principles could be extended to address emerging challenges in educational AI. As artificial intelligence becomes increasingly sophisticated and ubiquitous in educational contexts, the need for robust validation methods will only grow. The approach developed here provides a foundation for more comprehensive validation frameworks that could be adapted to different types of AI systems and educational applications.

Potential future applications include the validation of large language models used for automated essay scoring, the explanation of recommendation systems in adaptive learning platforms, and the interpretation of computer vision systems used for automated assessment of student work. Each of these applications would require adaptations of the basic methodology, but the core principle of validating that AI systems make decisions based on appropriate criteria remains universally relevant.

The method could also contribute to the development of AI systems that are interpretable by design rather than requiring post-hoc explanation. By incorporating validation criteria similar to those used in the proposed method into the training process itself, future research might develop neural networks that inherently make decisions based on theoretically appropriate factors.

### 6.6 Final Thoughts

The integration of artificial intelligence into educational and psychological practice represents both tremendous opportunity and significant responsibility. AI systems have the potential to provide personalized, adaptive, and highly effective educational experiences that could transform learning outcomes for students worldwide. However, realizing this potential requires more than just achieving high predictive accuracy; it demands ensuring that AI systems operate in ways that are transparent, fair, and aligned with educational goals and values.

This research contributes to that broader goal by providing tools and frameworks for understanding and validating AI behavior in educational contexts. The proposed perturbation-Pratt's measure method represents one step toward the responsible development and deployment of AI in education, offering a concrete approach for ensuring that neural networks make decisions for the right reasons.

As the field continues to evolve, the principles demonstrated here—the importance of theoretical grounding, the value of domain expertise in interpretation, and the need for comprehensive validation beyond predictive accuracy—will remain relevant regardless of the specific technologies employed. The ultimate goal is not just to build AI systems that are accurate, but to build systems that are worthy of the trust placed in them by educators, students, and society as a whole.

The path toward this goal requires continued collaboration between machine learning researchers, educational practitioners, and domain experts who can ensure that technical advances serve educational purposes. This research represents one contribution to that collaborative effort, providing both technical tools and conceptual frameworks that can support the responsible development of AI in education. As we continue to navigate the integration of artificial intelligence into educational practice, methods like the one proposed here will play a crucial role in ensuring that technological capability is matched by theoretical understanding and practical wisdom.

## References

Arrieta, A. B., Díaz-Rodríguez, N., Del Ser, J., Bennetot, A., Tabik, S., Barbado, A., ... & Herrera, F. (2020). Explainable artificial intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI. *Information Fusion*, 58, 82-115.

Dastin, J. (2018). Amazon scraps secret AI recruiting tool that showed bias against women. *Reuters*, October 9, 2018.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.

iKoda. (2017). *College Major Preference Assessment*. iKoda Research.

Lapuschkin, S., Wäldchen, S., Binder, A., Montavon, G., Samek, W., & Müller, K. R. (2019). Unmasking Clever Hans predictors and assessing what machines really learn. *Nature Communications*, 10(1), 1096.

LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436-444.

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765-4774.

Miller, T. (2019). Explanation in artificial intelligence: Insights from the social sciences. *Artificial Intelligence*, 267, 1-38.

Montavon, G., Samek, W., & Müller, K. R. (2018). Methods for interpreting and understanding deep neural networks. *Digital Signal Processing*, 73, 1-15.

Pratt, J. W. (1987). Dividing the indivisible: Using simple symmetry to partition variance explained. In *Proceedings of the second international conference in statistics* (pp. 245-260). University of Tampere.

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135-1144.

Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1(5), 206-215.

Samek, W., Montavon, G., Vedaldi, A., Hansen, L. K., & Müller, K. R. (Eds.). (2019). *Explainable AI: Interpreting, explaining and visualizing deep learning* (Vol. 11700). Springer.

Thomas, D. R., Hughes, E., & Zumbo, B. D. (1998). On variable importance in linear regression. *Social Indicators Research*, 45(1-3), 253-275.

Wu, A. D. (2021). Validation evidence for the College Major Preference Assessment. *Journal of Career Assessment*, 29(3), 456-478.

Wu, A. D., Hu, S. F., & Stone, C. A. (2022). Neural networks as flexible scoring mechanisms for short test forms. *Educational and Psychological Measurement*, 82(4), 687-712.

Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. *European Conference on Computer Vision*, 818-833.

Zou, J., & Schiebinger, L. (2018). AI can be sexist and racist—it's time to make it fair. *Nature*, 559(7714), 324-326. 