# Image Classification Using CNN and Vision Transformer (ViT)

## Overview

This project compares two pretrained deep-learning models for image
classification: - **MobileNetV2 (CNN):** extracts visual features using
convolutional layers. - **Vision Transformer (ViT):** processes image
patches using self-attention.

The models are evaluated on the **Oxford-IIIT Pet Dataset**, which
contains 37 cat and dog breeds.

## Objectives

-   Implement transfer learning for image classification.
-   Fine-tune pretrained CNN and ViT models.
-   Compare accuracy, loss, precision, recall, F1-score, and training
    time.
-   Visualize model performance and example predictions.

## Dataset

-   **Name:** Oxford-IIIT Pet Dataset
-   **Classes:** 37 pet breeds
-   **Input size:** 224 × 224 RGB images
-   **Splits:** Official training and testing splits provided by
    TensorFlow Datasets

The notebook streams images in batches to reduce RAM usage.

## Models

### MobileNetV2

A lightweight CNN pretrained on ImageNet. The classification head is
trained for the pet-breed classes, followed by fine-tuning selected
later layers.

### Vision Transformer (ViT)

The pretrained `google/vit-base-patch16-224` model is adapted to
classify 37 pet breeds and fine-tuned on the dataset.

## Requirements

-   Python
-   TensorFlow
-   TensorFlow Datasets
-   Transformers
-   `tf-keras`
-   scikit-learn
-   NumPy
-   pandas
-   Matplotlib
-   `importlib_resources`

The notebook's first code cell installs the required packages.

## How to Run

1.  Open the notebook in Google Colab.
2.  Select a GPU runtime if available.
3.  Run the installation cell and restart the runtime if prompted.
4.  Execute all seven notebook blocks in order.
5.  Wait for both models to train and evaluate.
6.  Review the comparison table, plots, and sample predictions.

**Memory tip:** Do not run older cells that load the entire dataset into
NumPy arrays. Use the streaming `tf.data` pipeline in the corrected
notebook. If GPU memory runs out during ViT training, reduce its batch
size.

## Evaluation Metrics

-   **Accuracy:** proportion of correct predictions.
-   **Precision:** proportion of predicted positives that are correct.
-   **Recall:** proportion of actual positives correctly identified.
-   **F1-score:** harmonic mean of precision and recall.
-   **Loss:** measure of prediction error.
-   **Training time:** time taken to train each model.

## Results

The notebook prints the measured results for both models and visualizes
their accuracy and training time. Add your actual output here after
running the experiment.

  Metric                 MobileNetV2 (CNN)                ViT
  -------------------- ------------------- ------------------
  Test accuracy           Fill from output   Fill from output
  Test loss               Fill from output   Fill from output
  Weighted precision      Fill from output   Fill from output
  Weighted recall         Fill from output   Fill from output
  Weighted F1-score       Fill from output   Fill from output
  Training time           Fill from output   Fill from output

An accuracy of 80% is a target, not a guaranteed result. Report the
values actually produced by your run.

## Conclusion

This project demonstrates transfer learning with a convolutional model
and a Transformer-based vision model. MobileNetV2 learns image features
through convolution, while ViT uses attention across image patches.
Their performance is compared on the same test set to understand
differences in predictive performance and computational cost.

## License and Dataset

Refer to the Oxford-IIIT Pet Dataset terms and the licenses of the
pretrained models and software libraries before redistributing data or
model weights.
