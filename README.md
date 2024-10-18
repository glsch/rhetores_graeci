## MLM Tuning
### Dataset
Before training classifiers, all the models were tuned with a MLM objective on a 
corpus of 75 authors of rhetorician and orators, who were selected based on 'Rhet.' and 'Orat.' epithets in the TLG.

| TLG Author ID | Name                                                        | Epithets               | Region                                             | Period         | Num. works |
|---------------|-------------------------------------------------------------|------------------------|----------------------------------------------------|----------------|------------|
| 10            | ISOCRATES                                                   | Orat.                  | Atheniensis                                        | 5–4 B.C.       | 31         |
| 14            | DEMOSTHENES                                                 | Orat.                  | Atheniensis                                        | 4 B.C.         | 64         |
| 17            | ISAEUS                                                      | Orat.                  | Atheniensis fort. Chalcidicus                      | 5–4 B.C.       | 13         |
| 26            | AESCHINES                                                   | Orat.                  | Atheniensis                                        | 4 B.C.         | 4          |
| 27            | ANDOCIDES                                                   | Orat.                  | Atheniensis                                        | 5–4 B.C.       | 5          |
| 28            | ANTIPHON                                                    | Orat.                  | Atheniensis                                        | 5 B.C.         | 9          |
| 29            | DINARCHUS                                                   | Orat.                  | Corinthius Atheniensis                             | 4–3 B.C.       | 5          |
| 30            | HYPERIDES                                                   | Orat.                  | Atheniensis                                        | 4 B.C.         | 7          |
| 34            | LYCURGUS                                                    | Orat.                  | Atheniensis                                        | 4 B.C.         | 2          |
| 81            | DIONYSIUS HALICARNASSENSIS                                  | Hist. Rhet.            | Halicarnassensis                                   | 1 B.C.         | 18         |
| 87            | Aelius HERODIANUS et Pseudo–HERODIANUS                      | Gramm. Rhet.           | Alexandrinus Romanus                               | A.D. 2         | 50         |
| 165           | DIODORUS                                                    | Rhet.                  | Sardianus                                          | 1 B.C.         | 2          |
| 186           | Marcus Cornelius FRONTO                                     | Rhet.                  | Numidianus                                         | A.D. 2         | 5          |
| 257           | PHILISCUS                                                   | Rhet.                  | Milesius                                           | 5–4 B.C.       | 1          |
| 284           | Aelius ARISTIDES                                            | Rhet.                  | Mysius                                             | A.D. 2         | 57         |
| 535           | DEMADES                                                     | Orat. Rhet.            | Atheniensis                                        | 4 B.C.         | 3          |
| 540           | LYSIAS                                                      | Orat.                  | Atheniensis                                        | 5–4 B.C.       | 36         |
| 547           | ANAXIMENES                                                  | Hist. Rhet.            | Lampsacenus                                        | 4 B.C.         | 3          |
| 560           | [LONGINUS]                                                  | Rhet.                  | NaN                                                | A.D. 1?        | 1          |
| 591           | ANTISTHENES                                                 | Phil. Rhet.            | Atheniensis                                        | 5–4 B.C.       | 2          |
| 592           | HERMOGENES                                                  | Rhet.                  | Tarsensis                                          | A.D. 2–3       | 6          |
| 593           | GORGIAS                                                     | Rhet. Soph.            | Leontinus                                          | 5–4 B.C.       | 3          |
| 594           | ALEXANDER                                                   | Rhet. Soph.            | NaN                                                | A.D. 2         | 2          |
| 598           | RHETORICA ANONYMA                                           | Rhet.                  | NaN                                                | Varia          | 22         |
| 605           | POLYBIUS                                                    | Rhet.                  | Sardianus                                          | Incertum       | 2          |
| 607           | Aelius THEON                                                | Rhet.                  | Alexandrinus                                       | A.D. 1/2       | 1          |
| 610           | ALCIDAMAS                                                   | Rhet.                  | Atheniensis                                        | 4 B.C.         | 1          |
| 613           | 〈DEMETRIUS〉                                                 | Rhet.                  | NaN                                                | 1 B.C./A.D. 1? | 1          |
| 616           | POLYAENUS                                                   | Rhet.                  | Macedo                                             | A.D. 2         | 4          |
| 640           | ALCIPHRON                                                   | Rhet. Soph.            | NaN                                                | A.D. 2/3       | 1          |
| 649           | LESBONAX                                                    | Rhet.                  | NaN                                                | A.D. 2         | 3          |
| 666           | ADRIANUS                                                    | Rhet. Soph.            | Tyrius                                             | A.D. 2         | 1          |
| 698           | ALEXANDER                                                   | Rhet.                  | Ephesius                                           | 1 B.C.         | 1          |
| 1150          | APHAREUS                                                    | Rhet.                  | Atheniensis                                        | 4 B.C.         | 2          |
| 1219          | BATO                                                        | Hist. Rhet.            | Sinopensis                                         | 2 B.C.         | 1          |
| 1249          | CEPHALION                                                   | Hist. Rhet.            | NaN                                                | A.D. 2         | 1          |
| 1302          | DEMETRIUS                                                   | Rhet.                  | NaN                                                | 2/1 B.C.       | 2          |
| 1303          | DEMOCHARES                                                  | Hist. Orat.            | Atheniensis                                        | 4–3 B.C.       | 1          |
| 1318          | DIODORUS                                                    | Rhet.                  | NaN                                                | Incertum       | 1          |
| 1376          | EUDEMUS                                                     | Rhet.                  | fort. Argivus                                      | A.D. 2?        | 1          |
| 1377          | FAVORINUS                                                   | Phil. Rhet.            | Arelatensis                                        | A.D. 2         | 1          |
| 1729          | THRASYMACHUS                                                | Rhet. Soph.            | Chalcedonius                                       | 5 B.C.         | 2          |
| 2001          | THEMISTIUS                                                  | Phil. Rhet.            | Constantinopolitanus                               | A.D. 4         | 41         |
| 2002          | ANONYMUS SEGUERIANUS                                        | Rhet.                  | NaN                                                | A.D. 3         | 1          |
| 2025          | MAXIMUS                                                     | Rhet.                  | Byzantius vel Epirota                              | A.D. 4?        | 1          |
| 2027          | Valerius APSINES                                            | Rhet.                  | Gadarensis Atheniensis                             | A.D. 3         | 2          |
| 2031          | SOPATER                                                     | Rhet.                  | Atheniensis                                        | A.D. 4         | 3          |
| 2047          | SYRIANI, SOPATRI ET MARCELLINI SCHOLIA AD HERMOGENIS STATUS | Rhet.                  | NaN                                                | p. A.D. 7      | 1          |
| 2178          | Cassius LONGINUS                                            | Phil. Rhet.            | Atheniensis Palmyrenus                             | A.D. 3         | 3          |
| 2200          | LIBANIUS                                                    | Rhet. Soph.            | Antiochenus Constantinopolitanus Nicomediensis     | A.D. 4         | 12         |
| 2417          | CINEAS                                                      | Rhet.                  | Thessalius                                         | 4/3 B.C.?      | 1          |
| 2586          | MENANDER                                                    | Rhet.                  | Laodicensis                                        | A.D. 3/4       | 2          |
| 2592          | Joannes PEDIASIMUS                                          | Philol. Rhet.          | Thessalonicensis Constantinopolitanus              | A.D. 13–14     | 1          |
| 2598          | PROCOPIUS                                                   | Rhet. Scr. Eccl.       | Gazaeus                                            | A.D. 5–6       | 10         |
| 2601          | TIBERIUS                                                    | Rhet.                  | NaN                                                | A.D. 3/4       | 1          |
| 2604          | ULPIANUS                                                    | Gramm. Rhet.           | NaN                                                | A.D. 4         | 1          |
| 2697          | TIMOLAUS                                                    | Rhet.                  | Macedo Larissaeus                                  | 4 B.C.         | 1          |
| 2699          | AMPHICRATES                                                 | Rhet.                  | Atheniensis                                        | a. A.D. 2      | 1          |
| 2866          | OECUMENIUS                                                  | Phil. Rhet.            | NaN                                                | A.D. 6         | 15         |
| 2903          | MINUCIANUS Junior                                           | Rhet.                  | Atheniensis                                        | A.D. 3         | 1          |
| 2904          | NICOLAUS                                                    | Rhet. Soph.            | Myrensis (Lyciae) Atheniensis Constantinopolitanus | A.D. 5         | 3          |
| 2946          | PRISCUS                                                     | Hist. Rhet.            | Panites                                            | A.D. 5         | 2          |
| 3027          | Joannes DOXAPATRES                                          | Rhet.                  | Constantinopolitanus                               | A.D. 11        | 4          |
| 3094          | NICETAS CHONIATES                                           | Hist. Scr. Eccl. Rhet. | Choniates Constantinopolitanus                     | A.D. 12–13     | 1          |
| 4001          | AENEAS                                                      | Phil. Rhet.            | Gazaeus                                            | A.D. 6         | 2          |
| 4026          | ANONYMI IN ARISTOTELIS ARTEM RHETORICAM                     | Rhet.                  | NaN                                                | Varia          | 3          |
| 4094          | CHORICIUS                                                   | Rhet. Soph.            | Gazaeus                                            | A.D. 6         | 2          |
| 4100          | APHTHONIUS                                                  | Rhet.                  | Antiochenus                                        | A.D. 4/5       | 1          |
| 4157          | JOANNES                                                     | Rhet.                  | Sardianus                                          | a. A.D. 10     | 1          |
| 4235          | JOANNES                                                     | Rhet.                  | Siculus                                            | A.D. 11        | 2          |
| 4236          | TROPHONIUS                                                  | Rhet. Soph.            | NaN                                                | A.D. 6         | 1          |
| 4242          | CYRUS                                                       | Rhet.                  | NaN                                                | Incertum       | 1          |
| 5024          | ANONYMI IN HERMOGENEM                                       | Rhet.                  | NaN                                                | Varia          | 21         |
| 5045          | ANONYMI IN APHTHONIUM                                       | Rhet.                  | NaN                                                | Varia          | 2          |
| 5046          | SCHOLIA IN THEONEM RHETOREM                                 | Schol. Rhet.           | NaN                                                | Varia          | 1          |

### Preprocessing
Each work was split into non-overlapping chunks of 512 tokens. The tokenizer used was 'bowphs/GreBerta'. The dataset was split into 80% training, 10% validation, and 10% test sets.

|              |                 |
|--------------|-----------------|
| Authors      | 75              |
| Works        | 523             |
| Total words  | 5,674,974       |
| Tokenizer    | bowphs/GreBerta |
| Total chunks | 15,245          |
| Chunk length | 512             |
| Overlap      | 0.0 %           |

### Tuning setup
All the models were tuned as long as Cross-Entropy loss continued to decrease more than 0.05 over 3 epochs (for more parameters, see the article).
Mask probability 0.15. 

|      Base transformer          | Stopped after epoch | Best CE on val set | LR   |
|--------------------------------|---------------------|--------------------|------|
| bowphs/GreBerta                | 15                  | 1.82               | 5e-5 |
| altsoph/bert-base-ancientgreek-uncased              | 20                  | 1.77               | 1e-4 |
| pranaydeeps/Ancient-Greek-BERT              | 13                  | 1.63               |   1e-4   |

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







