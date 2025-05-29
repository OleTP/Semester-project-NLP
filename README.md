# Music Genre Classification Using Lyrics and Metadata

This project explores music genre classification using deep learning models (BERT, DistilBERT, and LLaMA) with song lyrics and optional metadata (release year, danceability).


*This project was primarily developed locally and later uploaded to GitHub.*

<br>

### Notebooks

`BERT_nb.ipynb`

- Fine-tuning BERT for genre classification with optional metadata

`DistilBERT_nb.ipynb`

- Fine-tuning DistilBERT for genre classification with optional metadata

`LLaMA_nb.ipynb`
- Zero-shot genre classification using LLaMA 3.1 8B Instruct

`exploration_and_split.ipynb`
- Data exploration, cleaning, and stratisfied split

`results_visualization.ipynb`
- Comparison of F1 scores and Ollama debugging chat with LLaMA 3.1 8B Instruct

### Python files
`train_eval_utils.py`
- Helper functions for training, evaluation and saving


<br>

**Each model is evaluated under 4 settings:**

- **No Metadata** – lyrics only
- **+ Year** – adds release year
- **+ Danceability** – adds danceability score
- **+ Both** – combines both metadata features



<br>
<br>

### Folder structure:


```
├── music_dataset_split/                # Preprocessed subsets
│   ├── Train/
│   │   └── training_data.csv
│   ├── Val/
│   │   └── validation_data.csv
│   └── Test/
│       └── test_data.csv
├── run_results
│   ├── models/                         # Trained models
│   │   ├── bert/                       
│   │   └── distilbert/                 
│   └── results/                        # Saved classification reports
│       ├── bert/
│       ├── distilbert/
│       └── llama/    
├── BERT_nb.ipynb                     # Fine-tuning BERT
├── DistilBERT_nb.ipynb               # Fine-tuning DistilBERT
├── LLaMA_nb.ipynb                    # Zero-shot classification with LLaMA
├── exploration_and_split.ipynb       # Data exploration and stratified splitting
├── results_visualization.ipynb       # F1 scores and Ollama
├── tcc_ceds_music.csv                # Raw dataset
├── train_eval_utils.py               # Utility functions for training and evaluation
├── Info371_semester_project.pdf      # Semester project - Paper
└── README.md                                          
```