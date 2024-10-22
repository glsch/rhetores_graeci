# Fine-Tuning Pre-Trained Language Models for Authorship Attribution of the Pseudo-Dionysian Ars Rhetorica
The repository contains the code and the data for the paper "Fine-Tuning Pre-Trained Language Models for Authorship Attribution of the Pseudo-Dionysian _Ars Rhetorica_" by Gleb Schmidt, Veronika Vybornaya, and Ivan P. Yamshchikov, presented as a poster at the Computational Humanities Research Conference (December 4–6, 2024, Aarhus, Denmark):
```bibtex
@inproceedings{plm_aa_ars_rhetorica,
  title={Fine-Tuning Pre-Trained Language Models for Authorship Attribution of the Pseudo-Dionysian <i>Ars Rhetorica</i>},
  author={Schmidt, Gleb and Vybornaya, Veronika and Yamshchikov, Ivan P.},
  booktitle={Proceedings of the Computational Humanities Research Conference},
  eventtitle={Computational Humanities Research Conference},
  eventdate = {2024-12-04/2024-12-06},
  year={2024}
}
```
## Structure of the repository
``data/preprocessed`` Chunked (i.e., tokenized - chunked - back-decoded) data used for MLM fine-tuning and classifiers training. 
``data/classification``  
``notebooks`` Jupyter notebooks used for fine-tuning and classification. 
``src`` Python scripts used for fine-tuning and classification. 
``models`` Fine-tuned models. 
``results`` Results of the classification. 
``requirements.txt`` Required Python packages. 
``README.md`` This file.
### Dataset
Before training classifiers, all the models were tuned with a MLM objective on a 
corpus of 75 authors of rhetorician and orators, who were selected based on 'Rhet.' and 'Orat.' epithets in the TLG.



### Preprocessing
Each work was split into non-overlapping chunks of 512 tokens. The tokenizer used was 'bowphs/GreBerta'. The dataset was split into 80% training, 10% validation, and 10% test sets.



### Tuning setup
All the models were tuned as long as Cross-Entropy loss continued to decrease more than 0.05 over 3 epochs (for more parameters, see the article).
Mask probability 0.15. 

|      Base transformer          | Stopped after epoch | Best CE on val set | Test CE | LR   |
|--------------------------------|---------------------|--------------------|---------|------|
| bowphs/GreBerta                | 15                  | 1.82               | 1.83    | 5e-5 |
| altsoph/bert-base-ancientgreek-uncased              | 20                  | 1.77               | 1.73    | 1e-4 |
| pranaydeeps/Ancient-Greek-BERT              | 13                  | 1.63                   | 1.59    |   1e-4   |

## Classifier training
### Dataset
Each transformer and its derivative tuned on our corpus of orators and rhetorician were used to train a sequence classification model. Therefore, we trained 6 models in total.

|              |                                                          |
|--------------|----------------------------------------------------------|
| Authors      | 19, 10 classes and 9 added to the test corpus as "<UNK>" |
| Tokenizer    | bowphs/GreBerta                                          |
| Total chunks | 15,245                                                   |
| Chunk length | 128                                                      |
| Overlap      | 0.5 %                                                    |



### 







