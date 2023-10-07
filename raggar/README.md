# RAGGAR

RAGGAR is a highly opinionated RAG + GAR python package (not really a package I couldn't be bothered to make it that way).

## What is RAG?

RAG is retrival augmented generation. We can leverage in-context-learning (fancy prompting) for language models by retriving relevant content for prompting. Here is a CLAUDE explanation of RAG:

> Retrieval Augmented Generation (RAG) is a powerful technique in the field of Natural Language Processing (NLP) that combines the strengths of pre-trained language models with the benefits of retrieval-based models. RAG is designed to address the limitations of traditional language models that generate responses based solely on the input they receive and the information they have been trained on. While these models are effective in many scenarios, they can struggle with tasks that require specific, factual knowledge or the ability to reference multiple sources of information. RAG is useful for generating more accurate responses and reducing hallucinations by retrieving relevant documents from a large corpus of text and using these documents to inform the generation process.

## What is GAR?

GAR (generation augmented retreival) is "my" conception of the inverse of RAG. A standard problem in information retrieval is that our queries and documents are often vastly *different.* 

**Different** has a particular meaning in this context. Take an example query: `Q = What is the Nueva School?`

There are a variety of documents in our search space that could satisfiy this query, but lets pretend the most suitable document is: `D = The Nueva School is an innovative pre-k through highschool in San Mateo. The first person to graduate from the Upper-School was Zubin Aysola (mostly because he walked across the stage first).` 

This is a cheeky document that explains *some* historical fact about the Nueva School in conjuction with an introductory sentence. Importantly here, we can see that there is only one reference to the school's name in that entire document. This means our query-document similarity score (provided we have some vector representation of each) might not be very high. Other documents,perhaps in spanish because of the word *nueva*, could have MUCH higher scores. 

There are ways around this of course. NLP has had BERT similarity rankings using query-document concatenations for the last 5 years, but this problem still persists.

**GAR** is a different method leveraging the generative power of modern language models to assist in this query-document alignment. GAR posits that we can generate a better query from $Q$ that more closely aligns with our target document $D$ and as a result has a higher likelihood of matching. We can use this to augment and boost standard retrival methods!

## Why RAGGAR?

I mean I texted some friends about calling it GARRAG because thats more similar to the order of operations, but `RAGGAR` seemed like the name of a creature from Star Wars. RAGGAR also lives within my vision of the future where segmented "mini-intelligence-machines" (that one might call AI agents) ingest information, perform tasks, and trigger eachother, all to accomplish a unified goal.

You can imagine that RAGGAR is a system of two agents. A generation agent, and a retrival agent. The generation agent's job is to ingest queries and return document search terms. The retrieval agent's job is to ingest documents and return some result (TBD... you would make this obv).

## Zubin's Fake-y math for RAGGAR

Consider a space of documents $\mathbb{D}$ and a space for queries $\mathbb{Q}$. Lastly, consider a neutral embedding space $\mathbb{N}$ that you can imagine is a vector-space of embeddings from some model $\mathcal{M}(q_i, d_j | d \in \mathbb{D}, q\in \mathbb{Q})$

More simply $\mathbb{N} = \mathcal{M}(t | t \in T)$ where T is every possible piece of text. Basically $\mathbb{N}$ is just the embedding space of our model M (you might pool or take a single token embedding etc etc.)

We can intutively partition $\mathbb{N}$ into $\mathbb{D}$ and $\mathbb{Q}$ where each vector space (not really vector spaces but rather sets of vectors) is the embeddings of all the documents and queries independently. 

Based on our assumption (common sense) that the queries and documents have semantically distinct information we would expect the $INTERSECTION\_ AREA(\mathbb{D}, \mathbb{Q})$ to be low. You can imagine this as the overlap of the two sets of vectors in the embedding space $\mathbb{N}$.

Our goal with the `GAR` agent is to learn a mapping of queries such that that overlap increases. Given this principle, we could imagine doing this with an MLP or some differentiable operation, however in this project we just use LM generations. Importantly, `GAR` is about the NOTION of aligning these vector-spaces, hence ANY method that does this IS a GAR method. 

## Idk conclusion? 

Now you've read about RAGGAR. Any code that is not in this project that causes it to break is because I moved it to some non-public repo. Sorry! 

I like writing code but I'm busy so hopefully whatever is here is useful.


### GAR from Claude?

Does `Claude` know about `GAR`? No:

>Generation-Augmented Retrieval (GAR) is a technique in Natural Language Processing (NLP) that combines the strengths of retrieval-based models with the benefits of generation-based models. GAR is designed to improve the accuracy of open-domain question answering by using a retriever to retrieve relevant documents or passages and a generator to generate the answer based on the retrieved information. GAR has been shown to be effective in improving the performance of retrieval augmented generation (RAG) models for open-domain question answering[1][3]. Some sources to learn more about GAR include ResearchGate[3][4], LinkedIn[5], and arXiv[6].

>Citations:
[1] https://arxiv.org/pdf/2101.00774.pdf
[2] https://aclanthology.org/2023.findings-acl.46.pdf
[3] https://www.researchgate.net/publication/353484995_Generation-Augmented_Retrieval_for_Open-Domain_Question_Answering
[4] https://www.researchgate.net/publication/366497844_Generation-Augmented_Query_Expansion_For_Code_Retrieval
[5] https://www.linkedin.com/posts/sudheer-kolachina-21a53127_linguistics-languageai-naturallanguageprocessing-activity-7088813142218006528-h0J_
[6] https://www.arxiv-vanity.com/papers/2101.00774/