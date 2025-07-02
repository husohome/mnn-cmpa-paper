# Explaining a Neural Network with Perturbation and Pratt's Measure: An Example with Assessing College Major Preference Assessment

*Shun-Fu Hu* 
*Amery D. Wu*
*The University of British Columbia*

## Abstract

The proliferation of neural networks (NNs) across diverse domains has brought unprecedented predictive capabilities alongside significant interpretability challenges. This paper addresses the critical need for explainable artificial intelligence by proposing a novel method that combines perturbation techniques with Pratt's measures to explain neural network behavior in psychometric applications. While neural networks achieve superior predictive accuracy compared to traditional statistical methods, their internal mechanisms remain opaque like a "black box," which can be problematic when making consequential decisions in educational and psychological assessment contexts.

The proposed methodology selectively disables portions of input variables through systematic perturbation, trains multiple neural networks under controlled conditions, and employs Pratt's measures to quantify the relative importance of each input variable in the prediction process. This approach addresses a fundamental gap in the literature by providing a statistically grounded, interpretable framework for understanding how neural networks utilize input information to generate predictions.

Using the College Major Preference Assessment (CMPA) as a comprehensive working example, we demonstrate how to diagnose whether multilabel neural networks make predictions based on theoretically appropriate information when predicting student major preferences. The study involved 9,442 participants and examined 50 different college majors through a systematic analysis of 99 input variables. Our results show that this method can effectively identify neural network prediction behavior patterns, distinguish between valid and spurious variable contributions, and provide evidence for validating the effectiveness of neural networks as scoring mechanisms in psychometric contexts.

The findings reveal that the proposed perturbation-Pratt's measure approach successfully identified cases where neural networks relied on theoretically expected variables (e.g., psychology-related items predicting psychology major preference) as well as instances where networks achieved high accuracy through potentially spurious correlations. This capability is crucial for ensuring the validity and trustworthiness of neural network applications in high-stakes educational and psychological assessment scenarios.

The contribution of this work extends beyond the specific application domain, offering a generalizable framework for explaining neural network behavior that can be adapted to various fields requiring interpretable machine learning solutions. The method's simplicity, statistical foundation, and practical applicability make it particularly valuable for researchers and practitioners seeking to bridge the gap between predictive performance and interpretability in neural network applications.

**Keywords:** Neural Networks, Explainable AI, Pratt's Measures, Perturbation Methods, College Major Preference, Psychometric Assessment, Multilabel Classification, Educational Data Mining

## References

LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436-444.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135-1144.

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765-4774.

Pratt, J. W. (1987). Dividing the indivisible: Using simple symmetry to partition variance explained. In *Proceedings of the second international conference in statistics* (pp. 245-260). University of Tampere.

Thomas, D. R., Hughes, E., & Zumbo, B. D. (1998). On variable importance in linear regression. *Social Indicators Research*, 45(1-3), 253-275.

Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. *European Conference on Computer Vision*, 818-833.

Simonyan, K., Vedaldi, A., & Zisserman, A. (2013). Deep inside convolutional networks: Visualising image classification models and saliency maps. *arXiv preprint arXiv:1312.6034*. 