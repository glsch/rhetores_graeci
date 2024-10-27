# Fine-Tuning Pre-Trained Language Models for Authorship Attribution of the Pseudo-Dionysian Ars Rhetorica
The repository contains the code and the data for the paper "Fine-Tuning Pre-Trained Language Models for Authorship Attribution of the Pseudo-Dionysian _Ars Rhetorica_" by Gleb Schmidt, Veronika Vybornaya, and Ivan P. Yamshchikov, presented as a poster at the Computational Humanities Research Conference (December 4–6, 2024, Aarhus, Denmark):
```bibtex
@inproceedings{plm_aa_ars_rhetorica_2024,
  title={Fine-Tuning Pre-Trained Language Models for Authorship Attribution of the Pseudo-Dionysian <i>Ars Rhetorica</i>},
  author={Schmidt, Gleb and Vybornaya, Veronika and Yamshchikov, Ivan P.},
  booktitle={Proceedings of the Computational Humanities Research Conference},
  eventtitle={Computational Humanities Research Conference 2024},
  eventdate = {2024-12-04/2024-12-06},
  year={2024}
}
```
## Structure of the repository
* ``data/preprocessed`` Chunked (i.e., tokenized - chunked - back-decoded) data used for MLM fine-tuning and classifiers training. 
    * ``data/preprocessed/mlm_preprocessed_dataset.csv)`` Chunks of 512 tokens used for MLM.
    * ``data/preprocessed/classification_preprocessed_dataset.csv)`` Chunks of 64 tokens overlapping by 32 tokens used for classification training.
* ``src``
    * ``datasets`` LightningDataModule and dataset classes for managing the data used in the study.
    * ``classification`` Contains LightningModule used for the experiments with classifiers and their training.
    * ``MlmTuningModule.py`` LightningModule for MLM fine-tuning (mostly wrapping the Huggingface's AutoModelForMaskedLM).

## Performance of the classifiers
### Overall performance on the test set
| Model | F1 Score | Accuracy |
|-------|-------|-------|
| pranaydeeps/Ancient-Greek-BERT (R) | 82.90 | 80.96 |
| [pranaydeeps/Ancient-Greek-BERT](https://huggingface.co/pranaydeeps/Ancient-Greek-BERT) | 83.68% | 81.83% |
| nlpaueb/bert-base-greek-uncased-v1 (R) | 78.34 | 74.98 |
| nlpaueb/bert-base-greek-uncased-v1 | 79.22 | 76.02 |
| [bowphs/GreBerta (R)](https://huggingface.co/glsch/bowphs_GreBerta_rhetores) | 90.14% | 90.12% |
| bowphs/GreBerta | 89.34 | 89.27 |
(R) means that before classifier training the model was fine-tuned with a MLM objective on the same corpus.
### Per-class F1-score on the test set
![Chart](test_f1_class.jpeg)
`<UNK>` label was present only in the validation data. The model was trained to assign it when the top prediction probability was below 0.8.

## Running the code

Clone the repository: 
```bash
git clone https://github.com/glsch/rhetores_graeci.git
```
and navigate to it:
```bash
cd rhetores_graeci
```
Install the requirements:
```bash
pip install -r requirements.txt
```
Depending on which sets of paths you want to use (specified in paths.json), set environment variables:
```bash
env LOCATION=colab
```
or 
```
env LOCATION=local
```
Make sure that the paths in `paths.json` are correct. 

Make sure that you have also downloaded ``nltk``'s additional resources:
```bash
python -c "import nltk; nltk.download('punkt')"
```
The module was designed to be used with WandbLogger and was not tested with other loggers. Therefore, you will have to log in to Wandb:
```bash
wandb login --relogin <YOUR_WANDB_ACCESS_TOKEN>
```
Now everything is set for running the module via Lightning CLI. 
For example, to fine-tune a model on the MLM task, run:
```bash 
python -m src.main fit --model=MlmTuningModule --config mlm_config.yaml --trainer.logger.init_args.project RhetoresGraeciMLM --trainer.logger.save_dir <PATH_TO_LOGGER_OUTPUT> --data.batch_size 32 --data.chunk_type CHUNK --data.chunk_length 512 --data.overlap 0.0 --model.init_args.num_warmup_steps 5 --model.init_args.optimizer.init_args.lr 1e-5 --data.num_workers 12 --data.persistent_workers True --model.init_args.push_to_hub True --data.base_transformer <TRANSFORMER_NAME_OR_PATH> --trainer.max_epochs 3
```
Similarly, you can train a classifier:
```bash
python -m src.main fit --model=ClassificationModule --config classification_config.yaml --trainer.logger.init_args.project RhetoresGraeciClassificationTraining --trainer.logger.save_dir <PATH_TO_LOGGER_OUTPUT> --data.batch_size 32 --model.init_args.num_warmup_steps 70 --model.init_args.optimizer.init_args.lr 3e-5 --data.num_workers 12 --data.persistent_workers True --model.init_args.push_to_hub True --data.base_transformer <TRANSFORMER_NAME_OR_PATH> --model.init_args.confidence_threshold 0.80 --model.init_args.rejection_method THRESHOLD --data.chunk_type CHUNK --data.chunk_length 64 --data.overlap 0.5 --trainer.max_steps 700 --trainer.val_check_interval 350 
```
In both cases, all the run parameters can be adjusted in the corresponding .yaml file or passed as run arguments in the command line.

Unfortunately, we cannot publish the full data. For this reason, the arguments `--data.chunk_type` `--data.chunk_length` and `--data.overlap` will not have any effect: the module will work with the files already in the `data/preprocessed` folder. There, the texts are already chunked.

To run prediction on the _Ars Rhetorica_ with the best-performing `bowphs/GreBerta` fine-tuned on the study dataset run:
```bash
python -m src.main predict --model=ClassificationModule --config classification_config.yaml --trainer.logger.init_args.project RhetoresGraeciClassificationPrediction --trainer.logger.save_dir <PATH_TO_LOGGER_OUTPUT> --data.batch_size 32 --data.base_transformer "glsch/bowphs_GreBerta_rhetores_classification" --model.init_args.rejection_method THRESHOLD --model.init_args.confidence_threshold 0.80
```
The resulting tables and images will be logged to the specified Wandb project and directory.
