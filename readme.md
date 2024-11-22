# Data Augmentation for Text using GAN

This repository provides a data augmentation pipeline for text using a Generative Adversarial Network (GAN) architecture, along with baseline methods based on POS tagging and word replacement by thresholding. It focuses on augmenting text data while preserving label information, which is crucial for improving the performance of downstream classification tasks, especially in scenarios with limited training data.

## Key Features and Functionality

* **GAN-based Augmentation**:  The core of the repository is a GAN that learns to generate new text samples similar to the input training data.  It uses a reward function based on the Discriminator's output and potentially other metrics (like BLEU score) to guide the Generator's learning process.
* **Mixed Adversarial Training**:  The `main_mixed.py` script supports a mixed adversarial training strategy where the generator is trained against both a discriminator for the text and a discriminator for the associated POS tag sequences. This encourages the generator to produce samples that are both grammatically correct and semantically consistent with the training data.
* **Baseline Augmentation Methods**:  The repository also implements two baseline augmentation methods:
    * **POS Tagging-based Augmentation**: Replaces words with the same POS tag with their most similar counterparts from a pre-trained word embedding.
    * **Threshold-based Augmentation**: Replaces words with similar words based on a cosine similarity threshold from a word embedding.
* **Pluggable Metrics**: The framework allows for incorporating different metrics (BLEU score, subset accuracy, Hamming loss, ranking loss) as part of the reward function during GAN training.
* **Conditional Sample Generation**:  The Generator can generate samples conditioned on noise or hidden embeddings, providing more control over the generation process.
* **Label Preservation**: The augmentation methods are designed to preserve the label information of the input data, making the augmented samples suitable for training classification models.
* **Multilabel Support**: The toolkit includes functions for handling multilabel datasets, including calculating specific multilabel metrics.


## Running the Code


1. **Install Dependencies**:  Install the necessary Python packages using `pip install -r requirements.txt`.
2. **Prepare Data**:  The input data should be in a text file, one sentence per line.  You also need to prepare a pickled file containing the corresponding labels for your training data. For using mixed training, you need to also prepare a file containing the POS tag sequences for the sentences in your data. Use the provided scripts for POS tagging and other data preparation steps as needed.
3. **Configuration**: Create a JSON file (like `sample_input_json.json`) specifying all the parameters for the augmentation method you want to use (GAN or baseline).
4. **Run the Script**: Execute `python main_mixed.py <your_config.json>` for the GAN-based method or `python main_run.py <your_config.json>` for the baseline methods.
5. **Augmented Data**: The generated augmented data will be saved in the location specified in your configuration file.

## Notes

* The repository uses pre-trained word embeddings (like Google News vectors) for the baseline methods. You may need to download and load these embeddings separately.
* The GAN-based method requires more computational resources and training time compared to the baseline methods.
* The effectiveness of the augmentation methods can vary depending on the dataset and task.

## Future Work

* Improved reward functions and training strategies for the GAN.
* More sophisticated language modeling for the Generator.
* Integration with other augmentation techniques.
* More comprehensive evaluation of the generated samples.


This README provides a comprehensive overview of the repository.  Refer to the individual Python files and the sample configuration file for more details on the specific parameters and functionalities.
