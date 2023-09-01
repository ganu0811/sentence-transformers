I have  created a copy of the original train_bi-encoder_mrl.py file with the name train_bi-encoder_mnrl_ganesh.py, where I have added my part of the code.
I have set the default model in this file as 'distilroberta-base'. I have made changes in line 229, where I have called 'corpus=corpus' for the training total size of the corpus.
In line 231, I have changed the loss as MultipleNegativesRankingLossGanesh. In this loss our gBCE loss gets initialised.

MultipleNegativesRankingLossGanesh:
In MultipleNegativesRankingLossGanesh, line 40, i have introduced the 't' parameter and have set its value as 0.75 which a float datatype.
In line 51, self.loss calls our gBCEloss.
