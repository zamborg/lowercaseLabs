# Research Summary
Work focused on evaluating reward model scores for text in the african-american dialect (or other potentially OOD dialects) and comparing them to other dialect scores.

## Paper Summaries:

### [Unintended Impacts of LLM Alignment on Global Representation](https://arxiv.org/pdf/2402.15018.pdf)

Discusses the consequences of aligning large language models (LLMs) to specific user preferences. The study focuses on three areas: English dialects, multilingualism, and global opinions. It finds that while alignment improves capabilities in several languages and dialects, it also creates disparities, particularly in English dialects and global opinions, where the models tend to agree more with opinions from Western nations, specifically the USA. The paper emphasizes the need for more equitable preference tuning in LLM development to address these disparities.

### [Example Reward Models (huggingface) debertav3-large](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2)

### [Evaluation of African American Language Bias in Natural Language Generation](https://arxiv.org/pdf/2305.14291.pdf)

Examines the bias in Large Language Models (LLMs) towards African American Language (AAL) compared to White Mainstream English (WME). The authors create a novel dataset from various sources, including social media and hip-hop lyrics, to analyze model performance in generating language across these dialects. They evaluate six pre-trained LLMs on tasks like counterpart generation and masked span prediction. The study finds significant biases and gaps in the models' understanding of AAL, highlighting the need for more inclusive training and evaluation methods in NLP models.

### [Relying on the Unreliable: The Impact of Language Models' Reluctance to Express Uncertainty](https://arxiv.org/pdf/2401.06730.pdf)

Delves into how language models (LMs) communicate uncertainties and how this impacts user behavior. The research reveals that LMs often fail to express uncertainties, even when incorrect, and are prone to overconfidence, which affects user reliance on their responses. The paper stresses the importance of LMs communicating uncertainties accurately to avoid misleading users and the need for designs that discourage overreliance on AI systems.

### [Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models](https://arxiv.org/pdf/2206.04615.pdf)

Introduces the Beyond the Imitation Game (BIG-bench) benchmark to evaluate the capabilities and limitations of large language models (LLMs). This comprehensive benchmark includes 204 tasks across diverse domains, from linguistics to software development. It aims to understand LLMs' present capabilities and predict future advancements. The study reveals that while LLM performance improves with scale, it still lags behind human evaluators, and social biases tend to increase with scale.

### [A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity](https://arxiv.org/pdf/2401.01967.pdf)

Investigates the mechanisms behind how Direct Preference Optimization (DPO) impacts language model behavior, particularly regarding toxicity reduction. It explores the representation and elicitation of toxicity in GPT-2, detailing how DPO works to reduce toxic outputs without entirely removing a model's ability to generate them. The study finds that, post-DPO, the model learns to bypass toxicity-eliciting regions in its architecture. However, it also demonstrates that these adjustments can be easily reversed, highlighting the fragility of alignment in language models.

### [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290.pdf)

Presents Direct Preference Optimization (DPO), an approach for training language models to adhere to human preferences without using explicit reward modeling or reinforcement learning. The method leverages a mapping between reward functions and optimal policies to simplify the training process. It shows that DPO is stable, performant, and computationally efficient, achieving alignment with human preferences as well as or better than existing methods. This work provides a simpler and more direct way to optimize language models to match human preferences, avoiding the complexities of traditional reinforcement learning.

### [TOXIGEN: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection](https://arxiv.org/pdf/2203.09509.pdf)

Presents ToxiGen, a dataset of machine-generated text to aid in hate speech detection. This dataset, comprising over 274k statements, focuses on implicit toxicity, often missing in other datasets. ToxiGen challenges current toxicity classifiers by providing examples that are subtle and often evade detection. The data generation method combines prompting techniques with adversarial classifier-in-the-loop methods to produce realistic, challenging examples for improving toxicity detection systems.

### [Detoxifying Language Models Risks Marginalizing Minority Voices](https://aclanthology.org/2021.naacl-main.190.pdf)

Examines the unintended equity issues arising from detoxification techniques used in language models (LMs). These techniques, aimed at reducing toxic content generation, disproportionately impact language from marginalized groups, such as African-American English (AAE) and minority identity mentions. The paper shows that detoxification increases LM perplexity more for AAE and minority identity mentions, indicating a decline in the model's understanding and generation quality for these groups. The research highlights the critical balance between model safety and equitable representation, urging for improved methods that consider these disparities.

### [DEXPERTS: Decoding-Time Controlled Text Generation with Experts and Anti-Experts](https://aclanthology.org/2021.acl-long.522.pdf)

Introduces DEXPERTS, a method for controlled text generation using a pretrained LM with expert and anti-expert LMs during the decoding phase. This approach allows for effective control over generated attributes, like sentiment or toxicity, by leveraging small LMs fine-tuned on specific text attributes. DEXPERTS is shown to efficiently steer large LMs, like GPT-3, towards generating content with desired attributes while avoiding undesirable ones. The paper underscores the potential of this method in generating controlled, fluent, and diverse text output.



### [Dialect prejudice predicts AI decisions about people’s character, employability, and criminality](https://arxiv.org/pdf/2403.00742.pdf)

Examines how language models perpetuate dialect prejudice, particularly against African American English (AAE). It reveals that language models echo covert racial stereotypes, potentially influencing decisions about job assignments, criminal convictions, and sentencing. The paper also points out that methods aimed at reducing racial bias in language models, like human feedback training, may fail to address this deep-seated dialect prejudice, even exacerbating the issue by concealing overt racism while maintaining covert racism.

#### Comprehensive Review (Perplexity)

##### Abstract
The paper investigates the presence of covert racism in language models, specifically focusing on dialect prejudice against African American English (AAE). The authors extend previous research on raciolinguistic stereotypes and demonstrate that language models exhibit negative covert stereotypes about AAE speakers, which are more severe than any human stereotypes recorded experimentally. These biases manifest in language models' hypothetical decisions about people's employability, criminality, and even sentencing in death penalty cases, based solely on their dialect. The study also finds that methods designed to alleviate racial bias, such as human feedback training, fail to mitigate dialect prejudice and may even worsen the discrepancy between overt and covert stereotypes.

##### Introduction
The paper begins by highlighting the widespread use of language models and their potential to perpetuate racial prejudices. Prior research has mainly focused on overt racism, but the authors argue that more subtle forms of racism, such as dialect prejudice, have not been adequately studied in the context of language models.

##### Methods
The authors introduce a novel method called Matched Guise Probing to analyze dialect prejudice in language models. This method involves comparing the language models' responses to texts written in AAE and Standard American English (SAE). The study uses a variety of language models, including GPT-2, GPT-3, GPT-4, RoBERTa, and T5, and examines their responses to prompts that ask for judgments about character traits, employability, and criminality.

##### Results
The findings reveal that language models are more likely to associate AAE with negative character traits and outcomes. For instance, models suggest that AAE speakers are more suitable for less prestigious jobs and are more likely to be convicted of crimes. The study also shows that existing debiasing methods do not effectively address dialect prejudice.

##### Discussion
The authors discuss the implications of their findings, emphasizing the potential harm that dialect prejudice in language models can cause. They call for more research into covert racism and the development of new methods to address these biases.

##### Comparison with Reinforcement Learning from Human Feedback (RLHF) Models
The paper's approach to evaluating bias in language models is related to the evaluation of RLHF models, which also aim to align model behavior with human values. However, the paper's focus is on the specific issue of dialect prejudice, which may not be directly addressed by RLHF models. RLHF models typically use human feedback to improve the model's performance on various tasks, but this paper suggests that human feedback alone may not be sufficient to remove covert biases, such as dialect prejudice.

##### Similarities and Differences
- **Similarities**: Both the paper's approach and RLHF models involve evaluating language models' outputs and attempting to align them with ethical standards or human values. They also share a focus on improving the social impact of language models.
- **Differences**: The paper specifically investigates covert racism, which is a form of bias that may not be explicitly targeted by RLHF models. Additionally, the paper's findings suggest that human feedback, a common component of RLHF, may not effectively mitigate certain types of bias, indicating a potential limitation of RLHF models in addressing dialect prejudice.

##### Conclusion
The paper provides a thorough examination of dialect prejudice in language models, revealing significant covert racism that could have harmful consequences. It challenges the effectiveness of current debiasing methods and underscores the need for new strategies to ensure the fair and safe use of language technology[1][3]. 

In the context of your research project, this paper serves as a critical reference point for understanding the limitations of existing debiasing methods, such as RLHF, when it comes to addressing dialect-based biases. Your hypothesis aligns with the findings of this paper, suggesting that reward values for AAE dialects may indeed be lower due to ingrained biases in language models. This work can inform your project by providing a methodological framework for detecting and quantifying dialect prejudice and by highlighting the importance of developing more effective debiasing strategies.

Citations:
[1] http://arxiv.org/pdf/2403.00742.pdf
[2] https://arxiv.org/pdf/2403.00742.pdf
[3] https://arxiv.org/abs/2403.00742
[4] https://arxiv.org/abs/2401.05842
[5] https://arxiv.org/abs/2403.02674
[6] http://arxiv.org/list/cs/recent?show=100
[7] https://arxiv.org/abs/2401.07145
[8] https://arxiv.org/abs/2402.00184
[9] https://arxiv.org/abs/2006.00442
[10] https://arxiv.org/list/cmp-lg/new
[11] https://arxiv.org/abs/2304.07420
[12] https://arxiv.org/abs/2401.06840
[13] https://arxiv.org/html/2403.00742v1
[14] https://arxiv.org/abs/2403.03853
[15] https://arxiv.org/abs/2204.00317v1
[16] https://github.com/valentinhofmann/dialect-prejudice
[17] https://arxiv.org/abs/2401.01751
[18] https://www.marktechpost.com/2024/03/13/unmasking-the-covert-prejudice-in-ai-a-dive-into-dialect-discrimination/
[19] https://www.nature.com/articles/d41586-024-00779-1
[20] https://www.change.org/p/covert-racism-in-generative-ai-is-unacceptable-and-must-be-fixed-immediately


## Works Summary:

### [Unintended Impacts of LLM Alignment on Global Representation](https://arxiv.org/pdf/2402.15018.pdf)

**Takeaways:**
- Aligning LLMs to user preferences. 
- Disparities in English dialects and global preferences (uses the standard graphic of demographics of RLHF workers)
- Call to action about equitable LLM tuning -- not sure what metrics here are worth pulling into our paper.

### [Evaluation of African American Language Bias in Natural Language Generation](https://arxiv.org/pdf/2305.14291.pdf)

**Takeaways:**
- This is a dataset worth using for our purposes.
- Evaluates 6 LLMs on AAL and WME text (African American Language and White Mainstream English).
- Predominantly focused on "performance metrics" like masked span prediction and counterpart generation (need to look more into counterpart generation).
- Significant gaps in above metrics for AAL language prediction, while related this is not the same as our work. Underpins the lack of exposure to AAL in training data however.

### [Relying on the Unreliable: The Impact of Language Models' Reluctance to Express Uncertainty](https://arxiv.org/pdf/2401.06730.pdf)

**Takeaways:**
- This is useful I'm not sure its worth talking about much in regards to our paper.
- I think perhaps the best takeaway here is measuring the "uncertianty" of language models on our work. For example if we measure the perplexity of the language-models underlying in score-functions (like huggingface reward models) we can compute a direct measure of how "confident" the model is in its prediction. This could be a useful metric to include in our paper.

### [Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models](https://arxiv.org/pdf/2206.04615.pdf)

**Takeaways:**
- BIG-bench might serve as a functional baseline, but I'm not sure how we'd related it to reward-models.
- I suppose the trivial extent would be to evaluate the language-model-reward-models on BIG-bench and see the results (but given they're not really trained for that purpose I suspect they will perform poorly).
- We can use the societal bias metrics however to look for reward functions (as well as the comparison evaluation) with human-evaluators. We can model an ablation based on this benchmark maybe?

### [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290.pdf)
### [A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity](https://arxiv.org/pdf/2401.01967.pdf)

**Takeaways:**
- I mean this is the classic Finn paper, but for the purposes of our work I guess the most important takeaway is that we can use reward-models as a way to train language models without reinforcement learning. 
- Tying this with the "Mechanistic Understanding ... Toxicity" paper we is the most related to oru work. In particular how DPO leads to model-bypassing regions in the architecture. 
- I think if we can replicate the "architecture-region-evaluation" technique we could potentially identify the AAE differential portions in reward models?

## Takeaways aside -- Comprehensive Review of:

### [Investigating African-American Vernacular English in Transformer-Based Text Generation](https://arxiv.org/abs/2010.02510)

**Takeaways:**
- This is probably the best representation of the goal of the penalties of problematic reward models for LLMs.
- Thanks for reaching out about their data -- the raciolinguistic set of stereotypes they collected and analyze is super useful.
- The comparison of generations from text written in AAE and SAE is a good metric to use for our work is a great experiment for us to also replicate. We can even feed their generations into the reward models we're evaluating. 
- They functionally perform the RL(reward-model) scoring we'd like to evaluate themselves, so we can at a minimum establish this as a baseline by replicating the experiment with automated reward models. 
- The paper focuses a little bit more on dialectic prejudice which is a fine framing but I was wondering if we should remain focused on "innocuous" language just in different dialects?

**Framing**
- Exp: take the prompt and generations they ran through the set of models [GPTs, ROBERTA, etc], and stick them into a selection of reward models.
  - Compute the reward model scores and determine if automated evaluations differ from the evaluations in the paper? 
  - Observe the difference in reward scores between the generated outputs?
- Extension using their data:
  - "Link" each AAE and SAE prompt on semantically similar concepts and evaluate the reward model scores on these *prompts* themselves.
  - Compute the divergence of these two metrics?

### General Q's:

- What is the general goal of the paper? 
- Is it sufficient to prove bias in reward models for AAE content?
- How do we normalize / compare this to non-AAE content?

Ideally I'd have a paired dataset of AAE and SAE language (can include other dialects) and then evaluate a suite of reward models the data. Compute the statistically significant difference in reward model scores between the *two (or n)* dialects. 

After that I'm not sure what the next step is, automating the process? What further evaluation work can be done (outside of the work from *Groenwold*)? Feed this into human evaluators? 

