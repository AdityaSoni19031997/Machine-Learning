- As per the task assigned, I have beated both the baselines set by the notebook(nbs will be used in future) which was provided to me. The main goal was to
beat the baselines without overfitting on the small dataset provided(file named sentiment.csv), re-writing the code in the spirit of software development best practises and to handle the class-imbalancedness associated with the data.

- Little thoughts about the dataset provided.

	- Dataset isn't too big as such (just 30k rows).
	- It also seems that the actual sentiments were mapped by a pre-defined dictionary mapping to bring down the count from 13 to 5 labels by considering some heuristics like combining minority classes etc.
	- The above mapping wasn't completely correct. Had Found around 10% counter-examples.
	
- For the pre-processing of the data, had followed standard manual inspection and improving the pre-processing step-by-step to improve the vocab and the embeddings coverage as depicted in the nbs attached. In a gist, you don't remove puncts, emojis, emoticons as just like that because if you have an embedding vector for the same, you should try to utilize
it as much as possible.

- Regarding data split, i randomly sampled 25k and 5k rows as my train and dev data.

- The models are mainly based on Deep-Learning architecture's and the class-imbalanced was particularly handled by using the Focal-Loss rather than the normal CE Loss (with/without class-weights etc). Both were tried but focal-loss one had an extra edge in terms of performance.
	
	- 1st architecture is simply based on using the embeddings and the LSTM followed by few linear layers and some meta-nlp feats added tot he linear layers  (concated with a layer)
	
	- 2nd architecture is using Conv2D on texts, the idea behind doing such is that when using text we only have a single channel, the text itself. The out_channels is the number of filters and the kernel_size is the size of the filters. Each of our kernel_sizes is going to be [n x emb_dim] where n is the size of the n-grams. [2-3-4-5-6-7] ngrams were used.
	
	1) The sentiment of a sentence might change when sentences get shorter.
	2) The sentiment of a long sentence sequence is close to other long sentence sequences build from the same base sentence.
	3) So 2,7-grams will help with learning under these circumstances.

- Few more things were tried like few more archs experiments, different embeddings, Vowpal Wabbit(VW) i.e. blazingly fast learning. VW didn't perform quite good enough as it had only 40% accuracy on the text which wasn't pre-processed.

