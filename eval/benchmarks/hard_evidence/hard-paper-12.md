# hard-paper-12

Question: The best-practices RAG paper trains a classifier to decide whether a query needs retrieval. Which pretrained model is it built on, and what precision and recall does it reach? In the Formal Algorithms for Transformers paper, which algorithm gives the forward pass for that model family, and which activation function does that family use instead of ReLU?

Gold answer: BERT-base-multilingual-cased (batch size 16, learning rate 1e-5), with accuracy 0.95, precision 0.96, recall 0.94 and F1 0.95. Formal Algorithms describes BERT, an encoder-only transformer, in Algorithm 9 (ETransformer; its training is Algorithm 12), and BERT uses the GELU nonlinearity instead of ReLU.

Documents: rag-best-practices-2024.pdf, formal-algorithms-for-transformers-2022.pdf

## Evidence

- Table 1 columns
  - rag-best-practices-2024.pdf, page 4
  - "Acc Prec Rec F1"

- classifier scores: Acc 0.95, Prec 0.96, Rec 0.94, F1 0.95
  - rag-best-practices-2024.pdf, page 4
  - "BERT-base-multilingual 0.95 0.96 0.94 0.95"

- the exact checkpoint and settings
  - rag-best-practices-2024.pdf, page 19
  - "We choose BERT-base-multilingual-cased as our classifier, with a batch size of 16 and a learning rate of 1e-5."

- BERT is encoder-only, Algorithms 9 and 12
  - formal-algorithms-for-transformers-2022.pdf, page 7
  - "BERT [DCLT19], which is an instance of an encoder-only transformer"

- same bullet
  - formal-algorithms-for-transformers-2022.pdf, page 7
  - "Algorithms 9 and 12."

- Algorithm 9 is the BERT architecture
  - formal-algorithms-for-transformers-2022.pdf, page 8
  - "It is described in detail in Algorithm 9."

- GELU instead of ReLU
  - formal-algorithms-for-transformers-2022.pdf, page 8
  - "It uses the GELU nonlinearity instead of ReLU"

- Algorithm 9's name in the pseudocode header (p12; the symbols around it do not extract)
  - formal-algorithms-for-transformers-2022.pdf, page 12
  - "ETransformer"

## What makes it hard

Chain of three: the classifier table (p4) and appendix (p19) name BERT, which is the key into Formal Algorithms (p7 lists the algorithms, p8 names Algorithm 9 and GELU).
