**BART + LoRA Title Generation for arXiv Dataset**

This project implements an automatic scientific title generation system using BART-base fine-tuned with LoRA (Low-Rank Adaptation) on the arXiv scientific paper dataset. The model takes a paper summary (abstract) as input and generates a concise and relevant title.

---

**1. Project Overview**

* Task: Text-to-text generation (Summary → Title)
* Base Model: `facebook/bart-base`
* Optimization: LoRA + FP16 mixed precision
* Training Framework: Hugging Face Transformers & Datasets
* Evaluation Metrics:

  * BLEU
  * ROUGE-1 / ROUGE-2 / ROUGE-L
  * METEOR
  * BERTScore (F1)

This setup is designed to achieve efficient fine-tuning with significantly fewer trainable parameters while maintaining strong generation quality.

---

**2. Dataset**

* Source: arXiv Scientific Papers Dataset (CSV)
* Used Columns:

  * `summary`: paper abstract
  * `title`: paper title

**Preprocessing Steps**

* Convert text to lowercase
* Remove extra whitespace and newline characters
* Drop missing values
* Format input as:

  ```text
  generate title: <summary>
  ```

**Dataset Split**

* Train: 80%
* Validation: 10%
* Test: 10%

---

**3. Exploratory Data Analysis**

Histograms are plotted to analyze:

* Summary length distribution (word count)
* Title length distribution (word count)

This helps define:

* MAX_INPUT_LENGTH = 512
* MAX_TARGET_LENGTH = 32

---

**4. Tokenization**

* Tokenizer: `BartTokenizer / AutoTokenizer`
* Input truncation enabled
* Target text tokenized separately
* Padding labels with `-100` for loss masking

---

**5. Model Architecture**

**Base Model**

* BartForConditionalGeneration

**LoRA Configuration**

```python
LoraConfig(
    task_type=SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1
)
```

**Benefits of LoRA**

* Drastically reduces trainable parameters
* Faster training
* Lower GPU memory usage

---

**6. Training Configuration**

Key training settings:

* Batch size: 16
* Learning rate: 1e-3
* Epochs: 1
* Mixed precision: FP16
* Evaluation strategy: Steps
* Generation enabled during evaluation

Training is performed using Seq2SeqTrainer.

---

**7. Evaluation Metrics**

To improve evaluation reliability, multiple complementary metrics are used:

* BLEU: Measures n-gram precision
* ROUGE: Measures recall-based overlap
* METEOR: Considers synonymy and word order
* BERTScore (F1): Semantic similarity using contextual embeddings

The final BERTScore is reported as the average F1 score across the test set.

---

**8. Evaluation on Test Set**

After training, the model is re-evaluated on the test set using a Fast Tokenizer to ensure decoding consistency.

Reported metrics:

* BLEU
* ROUGE-1 / ROUGE-2 / ROUGE-L
* METEOR
* BERTScore F1

---

**9. How to Run**

**Requirements**

```bash
pip install transformers datasets evaluate rouge-score sacrebleu bert_score peft
```

**Training & Evaluation**

1. Mount Google Drive
2. Load and preprocess dataset
3. Tokenize inputs
4. Apply LoRA to BART
5. Train with Seq2SeqTrainer
6. Evaluate on validation and test sets

---

**10. Output**

* Fine-tuned LoRA-adapted BART model
* Evaluation metrics printed to console
* Saved checkpoints in Google Drive

---

**11. Notes**

* This project focuses on parameter-efficient fine-tuning rather than full model training.
* The approach is suitable for limited GPU resources (e.g., Google Colab).
* Can be extended to:

  * Larger BART models
  * Multi-lingual title generation
  * Longer summaries

---

**12. Author**

Developed for research and experimentation in scientific text generation using efficient transformer fine-tuning techniques.
