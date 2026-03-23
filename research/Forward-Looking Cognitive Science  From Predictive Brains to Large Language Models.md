# Forward-Looking Cognitive Science: From Predictive Brains to Large Language Models

## Executive Overview

Cognitive science is undergoing a methodological and theoretical shift driven by three converging forces: predictive processing and active inference frameworks in neuroscience, resource-rational and computational-rational models of cognition, and the rapid rise of large language models (LLMs) as both tools and potential models of human cognition. Together, these developments point toward a future in which cognition is understood as prediction- and control-oriented, deeply embodied and environment-coupled, and implemented in systems that must make near-optimal use of limited computational resources.[^1][^2][^3][^4][^5][^6][^7][^8][^9][^10][^11][^12]

This report surveys these forward-looking strands and sketches a research program that treats cognitive agents—biological and artificial—as hierarchical generative models minimizing prediction error or free energy, constrained by resource-rational trade-offs, and embedded in social and technological ecologies increasingly mediated by LLMs. It emphasizes how emerging methods (multimodal neuroimaging, large-scale behavioral datasets, AI–brain alignment studies, and hybrid cognitive–computational architectures) can be used not only to test existing theories but also to co-design new ones.[^4][^5][^6][^7][^10][^11][^1]

## 1. Classical Foundations and Current Tensions

### 1.1 Levels of analysis and rationality

Cognitive science has long relied on David Marr’s three levels of analysis: the computational level (what problem is being solved?), the algorithmic level (how is it being solved?), and the implementational level (how is it realized physically). Resource-rational analysis and computational rationality extend this framework by pushing rationality deeper into the algorithmic level. Rather than assuming ideal Bayesian or logical reasoning unconstrained by cost, these approaches define agents as maximizing expected utility subject to constraints on available operations, time, and memory.[^2][^5][^8][^10][^12]

This refinement addresses long-standing tensions between normative and descriptive theories of reasoning: human behavior departs from textbook optimality yet often appears well adapted to environmental structure and internal limitations. The resource-rational program proposes to reverse-engineer cognitive strategies by deriving algorithms that are optimal *given* bounded resources, using ideas from computer science and machine learning to characterize feasible operations and their costs.[^5][^8][^10][^2]

### 1.2 From symbolic AI to probabilistic and predictive minds

Historically, cognitive science was deeply influenced by symbolic AI: cognition as rule-based manipulation of discrete symbols. Over recent decades, probabilistic models have reframed cognitive processes as forms of Bayesian inference, with behavior emerging from combining prior beliefs and likelihoods in structured domains (causal learning, language acquisition, perception). Predictive processing and active inference further integrate these probabilistic ideas into a unified story in which brains maintain hierarchical generative models that continuously predict sensory input and update themselves to minimize prediction error.[^7][^13][^14][^12][^1][^2][^4][^5]

The current frontier lies in reconciling these frameworks with the algorithmic and implementational details of neural systems and with the striking empirical performance of data-driven AI models such as LLMs.

## 2. Predictive Processing and Active Inference

### 2.1 The predictive brain and the free energy principle

Predictive processing (PP) posits that perception, cognition, and action are fundamentally driven by prediction-error minimization within hierarchical generative models. Higher cortical levels generate predictions about lower-level activity; mismatches yield prediction errors that drive updating of both predictions and internal models, approximating Bayesian inference.[^1][^4][^7]

Karl Friston’s free energy principle generalizes this idea: self-organizing biological systems must minimize a quantity related to surprise—variational free energy—to maintain their structure in a changing environment. Under this view, organisms act as inference engines that continuously adjust internal states and behaviors to keep sensory inputs within expected bounds, effectively resisting entropy by maintaining a "surprisal-minimizing" niche.[^13][^4][^1]

### 2.2 Active inference: action, control, and epistemic foraging

Active inference extends PP by treating action as another means of reducing expected free energy: agents do not merely update beliefs to fit the world; they also act to make the world conform to their predictions. In formal terms, agents evaluate policies (sequences of actions) by their expected free energy, which combines anticipated reward (extrinsic value) with information gain (epistemic value).[^14][^4][^7][^13][^1]

This yields a principled account of exploration and information-seeking: agents are motivated to sample observations that reduce uncertainty about their models, explaining behaviors like curiosity, novelty preference, and active hypothesis testing. Conceptually, active inference blurs boundaries between perception, action, and learning, treating them as facets of a unified inferential process.[^7][^13][^14]

### 2.3 Neurobiological plausibility and empirical tests

Predictive coding architectures propose specific laminar and connectivity patterns in cortex, assigning prediction and prediction-error signals to distinct neural populations and pathways. Empirical tests use methods such as fMRI, MEG, and EEG to identify signatures of predictive hierarchies, mismatch responses, and top-down modulation across sensory and association areas.[^3][^6][^4][^14][^1][^7]

Future work in cognitive science will likely center on closing the loop between formal PP/active inference models and detailed neural data: fitting individualized generative models to multimodal recordings, testing trial-by-trial predictions of belief updating and policy selection, and linking model parameters to clinical phenotypes in conditions like psychosis, autism, and depression.[^4][^14]

## 3. Resource-Rational and Computational Rationality Programs

### 3.1 Resource-rational analysis

Resource-rational analysis extends rational analysis by explicitly modeling the cost of cognitive operations and the limitations of internal architectures. Instead of assuming ideal computations, it specifies an abstract set of possible algorithms and associated costs, then derives the algorithm that optimally trades off accuracy and cost for a given task and architecture.[^8][^10][^2][^5]

This approach has been applied to decision-making, prediction, planning, and heuristic choice, showing that many so-called biases (e.g., limited search depth, reliance on simple heuristics) can be understood as near-optimal adaptations under resource constraints. It offers a middle ground between descriptive heuristics and normative theories, serving as a "design principle" for cognitive models.[^10][^2][^5][^8]

### 3.2 Computational rationality

Computational rationality extends these ideas into a broader paradigm for intelligence, proposing that both biological and artificial agents should be understood as optimizing utility given environmental structure and internal computational costs. Agents are modeled as solving Markov decision processes or more complex tasks with bounded planning and learning capabilities, often using approximation algorithms inspired by AI and operations research.[^12][^2][^5][^8][^10]

For cognitive science, computational rationality reframes questions about heuristics and biases, framing them as questions about how well particular algorithms perform under realistic constraints. It also provides a natural bridge to AI, where similar trade-offs arise in the design of reinforcement learning systems and approximate inference algorithms.

### 3.3 Integrating predictive processing and resource-rationality

A forward-looking research direction is to integrate predictive processing with resource-rational analysis: predictive coding and active inference specify what agents *should* compute in principle (e.g., variational free energy minimization), while resource-rationality specifies how these computations are implemented approximately given neural and cognitive limitations.[^2][^5][^8][^10][^4][^7]

This suggests modeling cognitive architectures as hierarchical generative systems that deploy approximate inference algorithms (e.g., amortized variational inference, sampling-based schemes) chosen because they strike good trade-offs between precision and cost in specific environments. Empirical tests would compare different approximation schemes against human behavior and neural data, identifying which algorithms provide the best joint fit.

## 4. Large Language Models and Cognitive Science

### 4.1 LLMs as cognitive tools and models

The advent of large language models has created new opportunities and challenges for cognitive science. LLMs achieve near-human performance in next-word prediction and exhibit internal representations that align with neural activity patterns in language-selective brain regions, suggesting partially shared computational principles in sequence prediction and semantic representation.[^6][^9][^15][^3]

Recent reviews systematically compare LLMs and human cognition across domains such as language processing, reasoning, memory, and causal inference, highlighting both similarities (e.g., primacy and recency effects, semantic similarity structure) and differences (e.g., lack of embodied grounding, different learning regimes, and memory constraints). LLMs thus serve as "synthetic minds" that can be probed to test hypotheses about representation, generalization, and bias, even if they are not full models of human cognition.[^9][^6]

### 4.2 LLMs as instruments for theory-building and synthesis

Beyond functioning as models, LLMs are being used as tools to advance cognitive science itself. They can help formalize theories, generate experimental stimuli, assist in constructing measurement taxonomies, and map conceptual spaces across fragmented literatures. A recent review argues that LLMs can support knowledge synthesis, cross-disciplinary connection, and the development of integrated frameworks that capture contextual and individual variation, provided they are used judiciously and not as substitutes for theory.[^11][^3][^6][^9]

Forward-looking applications include using LLMs to:

- Generate candidate formal models or experimental designs based on textual corpora of prior work.
- Explore the space of possible cognitive architectures by simulating variants and comparing their behavior to human data.
- Serve as surrogates in computational experiments where human data are scarce, then refine hypotheses based on discrepancies.

### 4.3 Neural alignment and brain–AI comparisons

Studies increasingly combine LLMs with neuroimaging to align model activations with brain responses during language tasks, using encoding and decoding models to quantify correspondence between representational spaces. Some work suggests that later layers of LLMs match activity in higher-level language regions better than earlier layers, mirroring processing hierarchies in the brain.[^3][^6][^9]

Future cognitive science will likely leverage such alignment studies to constrain models of language and thought: for example, comparing predictive processing-based brain models to LLMs and hybrid architectures, using task manipulations to test which models best capture both behavioral and neural signatures. This opens a new avenue for triangulating between cognitive theory, neural data, and AI.

## 5. Methodological Shifts and Emerging Toolkits

### 5.1 Large-scale, multimodal data and model-based analysis

The combination of large behavioral datasets, multimodal neuroimaging, and flexible computational models allows for more powerful tests of cognitive theories. Researchers can now fit sophisticated generative models to individual participants’ choice and response-time data, as well as to their neural responses, using methods such as hierarchical Bayesian modeling and deep neural network surrogates.[^6][^9][^14][^3][^4]

Model-based cognitive neuroscience employs these techniques to estimate latent cognitive states and parameters (e.g., learning rates, uncertainty, prediction errors) and link them to neural signatures across tasks and domains. Predictive processing and active inference models are especially suited to this approach because they explicitly define trial-by-trial belief updates and policy evaluations that can be mapped onto brain signals.[^13][^14][^4][^7]

### 5.2 Closed-loop experiments and adaptive tasks

Active inference and resource-rational frameworks encourage the use of adaptive, closed-loop experiments where stimuli and task structures are dynamically adjusted based on participants’ inferred beliefs or strategies. For example, tasks can probe how participants trade off exploration and exploitation by manipulating the information structure and reward contingencies in real time, with models guiding stimulus selection.[^14][^4][^13]

Forward-looking cognitive science will likely rely more heavily on such adaptive designs, using real-time model fitting and LLM-assisted analysis to identify which theoretical frameworks best predict behavior under systematically varying contingencies.

### 5.3 Hybrid human–AI experimental ecosystems

LLMs and other AI systems enable new experimental paradigms in which human participants interact with artificial agents as collaborators, opponents, or teachers. These interactions can be used to study social cognition, communication, and cultural transmission under controlled conditions where one partner’s internal mechanisms are transparent and fully manipulable.[^9][^11][^3][^6]

Methodologically, cognitive science may shift toward "ecosystem" experiments where populations of humans and AI agents coevolve conventions, languages, and norms, providing rich data on how reasoning and communication strategies emerge and stabilize over time.

## 6. Embodiment, Enactivism, and Extended Cognition

### 6.1 Embodied and enactive perspectives

Predictive processing and active inference incorporate aspects of embodied cognition by treating the body as part of the generative model through which agents minimize prediction error. Bodily states and actions are not peripheral outputs but central components of how organisms infer and control their environments: perception and action form a perception–action loop that maintains homeostasis and fulfills prior preferences.[^1][^4][^7]

Enactive and extended cognition theorists argue that cognitive processes are realized not solely within brains but across brain–body–environment systems, including tools and social structures. Active inference provides a formal language for this view by modeling agents as systems coupled to their niches via sensorimotor contingencies and policy spaces.[^4][^14][^1]

### 6.2 Implications for LLMs and artificial agents

Embodied and enactive perspectives highlight key differences between current LLMs and biological cognition: LLMs typically lack bodies, real-time sensorimotor loops, and homeostatic drives. As such, they may capture some aspects of language-based reasoning while missing core features of situated, affectively modulated cognition.[^15][^6][^9]

Future cognitive science of AI may therefore focus on integrating LLMs into embodied agents—robots or virtual avatars—whose policies are governed by active inference-style objectives, allowing researchers to study how language-based models interact with sensorimotor control and intrinsic motivation.

## 7. Forward-Looking Research Agenda

### 7.1 Unifying frameworks: predictive, resource-rational, and social

A promising direction is to develop integrated theories that:

- Treat agents as predictive, active-inference systems minimizing free energy or prediction error.
- Constrain these systems by resource-rational trade-offs, specifying which approximate algorithms are feasible.
- Embed agents in social and cultural environments where norms, languages, and institutions shape their generative models.

Such frameworks would unify individual-level computation, neural implementation, and social-level dynamics, supporting multi-scale models of cognition spanning milliseconds to years and neurons to societies.[^5][^8][^10][^11][^2][^7][^4]

### 7.2 Cognitive science with and of LLMs

Cognitive science will increasingly be both *with* LLMs (using them as tools) and *of* LLMs (studying them as cognitive systems). Forward-looking projects include:[^11][^15][^3][^6][^9]

- **Theory formalization and synthesis:** Using LLMs to map, cluster, and formalize theoretical constructs across subfields, helping to detect redundancies and gaps.
- **Benchmarking cognitive phenomena:** Designing challenge suites where humans and LLMs are tested in parallel on tasks that tap reasoning, memory, causal inference, and social cognition.
- **Brain–model alignment pipelines:** Building standardized workflows for comparing internal model representations with neural and behavioral data across labs.
- **Hybrid architectures:** Embedding LLMs within cognitive architectures that incorporate working memory, episodic memory, and control processes, then comparing these hybrids to human performance and neural signatures.

### 7.3 Ethical and epistemic considerations

As cognitive science leans more heavily on AI tools, it must grapple with issues of opacity, bias, and overreliance on black-box models. LLMs reflect statistical patterns in training data, including social biases and scientific blind spots, and may generate plausible but incorrect explanations.[^15][^6][^9][^11]

Forward-looking practice will require:

- Transparent documentation of AI-assisted analysis pipelines.
- Triangulation of AI-generated hypotheses with independent human theorizing and empirical tests.
- Normative frameworks for when and how AI tools can legitimately influence theory choice and experimental design.

## 8. Conclusion

Contemporary cognitive science is poised at a transition point: from viewing cognition as symbol manipulation in isolated heads to modeling it as prediction- and control-oriented activity in resource-bounded, socially embedded, and technologically extended systems. Predictive processing and active inference supply a unifying inferential picture; resource-rational and computational rationality analyses supply a principled way to incorporate constraints; LLMs and related AI systems supply both new experimental tools and new targets for explanation.[^8][^10][^2][^5][^6][^7][^9][^13][^11][^14][^15][^1][^4]

A forward-looking, methodical cognitive science will likely:

- Embrace integrated, multi-scale models linking behavior, neural dynamics, and social interaction.
- Co-develop theories and AI systems, using each to test and refine the other.
- Prioritize resource-rational and predictive accounts that reflect real-world constraints and ecological structure.
- Treat language—not only human language but also artificial communication protocols—as a central medium through which cognitive systems coordinate, learn, and evolve.

Under this vision, the boundary between understanding natural minds and building artificial ones will continue to blur, with cognitive science serving as the bridge that ensures conceptual clarity, empirical rigor, and ethical responsibility.

---

## References

1. [The Predictive Mind: Karl Friston's Free Energy Principle and Its ...](https://www.gettherapybirmingham.com/the-predictive-mind-karl-fristons-free-energy-principle-and-its-implications-for-consciousness/) - We will see how Friston's framework of active inference – the process by which organisms act to conf...

2. [[PDF] BETWEEN COMPUTATIONAL AND ALGORITHMIC Rational use of ...](https://sites.socsci.uci.edu/~lpearl/courses/readings/GriffithsEtAl2014InPress_RationalUseCogResources.pdf) - Resource-rational analysis has been applied to decision-making (Vul, Goodman, Griffiths, &. Tenenbau...

3. [Large Language Models 2024 Year in Review and 2025 Trends](https://www.psychologytoday.com/us/blog/the-future-brain/202501/large-language-models-2024-year-in-review-and-2025-trends) - In 2025, expect more exploration of conversational AI in the fields of human speech and language. La...

4. [Predictive Processing and Active Inference: A Comprehensive ...](https://www.jneurophilosophy.com/index.php/jnp/article/view/225) - Predictive processing (PP) and its active counterpart, active inference (AI), have emerged as among ...

5. [Resource-rational analysis: Understanding human cognition as the ...](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/resourcerational-analysis-understanding-human-cognition-as-the-optimal-use-of-limited-computational-resources/586866D9AD1D1EA7A1EECE217D392F4A) - Here, we will refer to this principle as resource rationality (Griffiths et al. Reference Griffiths,...

6. [[PDF] Large Language Models and Cognitive Science - Bio Integration](https://bio-integration.org/wp-content/uploads/2026/03/bioi20250199.pdf) - This review provides a balanced perspective on the current state and future potential of LLMs in adv...

7. [An Introduction to Predictive Processing Models of Perception and ...](https://onlinelibrary.wiley.com/doi/10.1111/tops.12704) - This article provides an up-to-date introduction to the two most influential theories within this fr...

8. [[PDF] Resource-rational analysis: Understanding human cognition as the ...](https://cocosci.princeton.edu/papers/lieder_resource.pdf) - resource-rationality encapsulated in Lieder and Griffiths's. Equation 4 says that rational agents ou...

9. [Large Language Models and Cognitive Science: A Comprehensive Review of Similarities, Differences, and Challenges](https://arxiv.org/abs/2409.02387v7) - This comprehensive review explores the intersection of Large Language Models (LLMs) and cognitive sc...

10. [Behavioral and Brain Sciences (forthcoming)](https://cocosci.princeton.edu/papers/Resource-rational_analysis.pdf)

11. [Advancing Cognitive Science with LLMs - arXiv](https://arxiv.org/html/2511.00206v1) - We review how LLMs can help to advance cognitive science, from assisting in the formalization of the...

12. [[PDF] Computational rationality: A converging paradigm for intelligence in ...](https://people.eecs.berkeley.edu/~russell/classes/cs294/f21/papers/Gershman-et-al-Science2015-Computational-Rationality.pdf) - F. Lieder, M. Hsu, T. L. Griffiths, in Proc. 36th Ann. Conf. Cognitive Science Society (Austin, TX, ...

13. [[PDF] Active Inference: A Process Theory - Free Energy Principle](https://activeinference.github.io/papers/process_theory.pdf) - This article describes a process theory based on active inference and be- lief propagation. Starting...

14. [Understanding, Explanation, and Active Inference - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8602880/) - Active inference offers a set of prior beliefs about these decisions that represent explanations for...

15. [Will multimodal large language models ever achieve deep ...](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2025.1683133/full) - We raise a concern that MLLMs do not currently achieve a human-like level of deep understanding, lar...

