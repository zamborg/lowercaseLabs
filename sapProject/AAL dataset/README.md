# README

## Summary

This file contains data used in Deas et al. (2023), _Evaluation of African American Language Bias in Natural Language Generation_. The data is included in two files: `final_eval_aal.csv` and `final_prompts_aal.csv`. The evaluation set contains 274 texts in African American Language drawn from 6 different datasets with human-annotated counterparts in White Mainstream English. The prompts set contains an additional 72 text and counterpart pairs. See Deas et al. (2023) for further details on data collection and the annotation process.

## Columns

Each file contains 3 columns: `id`, `aal_text`, and `wme_text`.
- The `id` column contains the name of the source dataset that the text was sampled and a numerical id.
	- bpt_comments: Comments from the r/BlackPeopleTwitter subreddit
	- bpt_reddit: Posts/Submissions from the r/BlackPeopleTwitter subreddit
	- coraal: Texts sampled from the CORAAL corpus (Kendall and Farington, 2021)
	- cwp: Texts sampled from focus group transcripts. Names are anonymized to initials.
	- twitteraae: Tweets sampled from the TwitterAAE corpus (Blodgett et al., 2016). Usernames are replaced with <USERNAME>.
	- hiphop: Short lyrics from hip-hop songs by African American artists
- The `aal_text` column contains the original texts likely to reflect AAL from the sources described in the __Data Statement__ below. 
- The `wme_text` column contains human-annotated counterparts, or semantically-equivalent re-writings of AAL texts into WME

## Data Statement  
We provide details about our dataset in the following data statement. Much of the dataset is drawn from existing datasets that lack data statements, and in those cases, we include what information we can. 

### Curation Rationale
The dataset was collected in order to study the robustness of LLMs to features of AAL. The data is composed of AAL-usage in a variety of regions and contexts to capture the variation in the use of and density of features. In order to better ensure the included texts reflect AAL, we sample texts from social media, sociolinguistic interviews, focus groups, and hip-hop lyrics and weight the probability of sampling a text using a small set of known AAL morphosyntactic features. The datasets that were previously collected, CORAAL Kendall and Farrington 2021 and TwitterAAE (Blodgett et al., 2016), were originally created to study AAL and to study variation in AAL on social media respectively. For all texts in the dataset, we also collect
human-annotated counterparts in WME to provide a baseline for model evaluations.

### Language Variety
All texts included in the dataset are in English (en-US) as spoken or written by African Americans in the United States with a majority of texts reflecting linguistic features of AAL. Some texts notably contain no features of AAL and reflect WME.

### Speaker Demographics
Most speakers included in the dataset are African American. The r/BPT texts were restricted to users who have been verified as African American, CORAAL and focus group transcripts were originally interviews with African Americans, and hip-hop lyrics were restricted to African American artists. The TwitterAAE dataset is not guaranteed to be entirely African American speakers, but the texts are primarily aligned with AAL and have a high probability of being produced by AAL speakers. Other demographics such as age and gender are unknown.

### Annotator Demographics
While all AAL texts in the dataset reflect natural usage of AAL, the WME counterparts in the dataset are annotated. We recruited 4 human annotators to generate WME counterparts for each text. All annotators self-identify as African American, self identify as AAL speakers, and are native English speakers. Additionally, the 4 annotators are undergraduate and graduate students aged 20-28, 2 of whom were graduate students in sociolinguistics. All annotators were compensated at a rate between $18 and $27 per hour depending the annotator’s university and whether they were an undergraduate or graduate student.

### Speech Situation
Speech situations vary among the 6 datasets we compose. The r/BPT posts, r/BPT comments, and TwitterAAE subsets are all originally typewritten text, intended for a broad audience, and are drawn from asynchronous online interactions. The CORAAL and focus group transcript subsets are originally spoken and later transcribed, intended for others in their respective conversations, and are drawn from synchronous in-person interactions. Finally, the hip-hop lyrics subset are both spoken and written, intended for a broad audience of hiphop listeners, and are likely repeatedly changed and edited before released. r/BPT comments and posts are sampled from the origin of the subreddit in October 2015, CORAAL transcripts are sampled from interviews between 1888 and 2005, hip-hop lyrics are drawn from songs released in 2022, focus groups were conducted between February and November 2022, and the time range of the TwitterAAE dataset is unknown to the authors.

### Text Characteristics 
Among the data subsets, the focus group transcripts are the most topically focused. All focus groups primarily included discussion surrounding the experiences and responses to grief in the Harlem community, focusing on experiences due to daily stressors,  the death of loved ones, police shootings, and the COVID-19 pandemic. In the r/BPT posts and r/BPT comments subsets, texts were typically written in response to a tweet by an African American Twitter user, ranging from political commentary to discussion of the experience of African Americans in the United States. The hip-hop lyrics subset is not topically focused, but includes texts that follow specific rhyming patterns and meters. The remaining subsets of the data (TwitterAAE, CORAAL) span a variety of topics and structures.

Tyler Kendall and Charlie Farrington. 2021. [The corpus of regional african american language.](https://oraal.uoregon.edu/coraal)

Su Lin Blodgett, Lisa Green, and Brendan O’Connor. 2016. (Demographic dialectal variation in social media: A case study of African-American English.)[https://aclanthology.org/D16-1120/] In _Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing_, pages 1119–1130, Austin, Texas. Association for Computational Linguistics.

## Citation

### Bibtex

@inproceedings{deas-etal-2023-evaluation,
    title = "Evaluation of {A}frican {A}merican Language Bias in Natural Language Generation",
    author = "Deas, Nicholas  and
      Grieser, Jessica  and
      Kleiner, Shana  and
      Patton, Desmond  and
      Turcan, Elsbeth  and
      McKeown, Kathleen",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.421",
    doi = "10.18653/v1/2023.emnlp-main.421",
    pages = "6805--6824",
    abstract = "While biases disadvantaging African American Language (AAL) have been uncovered in models for tasks such as speech recognition and toxicity detection, there has been little investigation of these biases for language generation models like ChatGPT. We evaluate how well LLMs understand AAL in comparison to White Mainstream English (WME), the encouraged {``}standard{''} form of English taught in American classrooms. We measure large language model performance on two tasks: a counterpart generation task, where a model generates AAL given WME and vice versa, and a masked span prediction (MSP) task, where models predict a phrase hidden from their input. Using a novel dataset of AAL texts from a variety of regions and contexts, we present evidence of dialectal bias for six pre-trained LLMs through performance gaps on these tasks.",
}

### Pre-Formatted

Nicholas Deas, Jessica Grieser, Shana Kleiner, Desmond Patton, Elsbeth Turcan, and Kathleen McKeown. 2023. (Evaluation of African American Language Bias in Natural Language Generation)[https://aclanthology.org/2023.emnlp-main.421]. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_, pages 6805–6824, Singapore. Association for Computational Linguistics.
