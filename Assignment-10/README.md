# Audio Classification Using Deep Learning with STT and TTS

## Project Overview

This project implements an audio classification system using a Convolutional Neural Network (CNN) to recognize spoken commands. It also integrates Speech-to-Text (STT) to convert speech into text and Text-to-Speech (TTS) to generate spoken audio responses.

The project uses the **Mini Speech Commands dataset** and TensorFlow/Keras to train and evaluate the deep learning model.

## Objectives

- Classify audio recordings into different speech-command categories.
- Implement a CNN model for audio classification.
- Integrate Speech-to-Text (STT) functionality.
- Integrate Text-to-Speech (TTS) functionality.
- Evaluate model performance using suitable metrics.

## Dataset

**Dataset:** Mini Speech Commands

The dataset contains recordings of eight spoken commands:

- Down
- Go
- Left
- No
- Right
- Stop
- Up
- Yes

The dataset is downloaded and extracted automatically by the notebook.

## Technologies Used

- Python
- TensorFlow and Keras
- NumPy
- Matplotlib
- Scikit-learn
- SpeechRecognition
- pyttsx3
- Librosa
- Jupyter Notebook

## Model Architecture

The CNN model processes audio waveforms and learns features that help distinguish between spoken commands.

The architecture includes:

1. Input audio waveform
2. One-dimensional convolutional layers
3. Activation and pooling layers
4. Dense layers
5. Dropout for regularization
6. Softmax output layer for classification

The model predicts the most likely command among the eight classes.

## Methodology

1. Download and extract the Mini Speech Commands dataset.
2. Load audio files and standardize their length.
3. Split the dataset into training and validation sets.
4. Build and train the CNN model.
5. Evaluate the model using classification metrics.
6. Predict the class of sample audio recordings.
7. Convert speech to text using STT.
8. Generate spoken responses using TTS.

## Performance Evaluation

The model can be evaluated using the following metrics:

- **Accuracy:** Overall proportion of correctly classified audio samples.
- **Precision:** Proportion of predicted samples for a class that are correct.
- **Recall:** Proportion of actual samples of a class that are correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Shows correct and incorrect predictions for each class.

Actual performance results depend on the training process and dataset split. Run the notebook to obtain the results.

## Speech-to-Text (STT)

The project uses the SpeechRecognition library with Google's online speech-recognition service to convert spoken input into text. An internet connection is required for this service.

## Text-to-Speech (TTS)

The pyttsx3 library converts text into speech using the available system speech engine. Depending on the environment, audio playback or saving generated audio may require additional configuration.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace `YOUR-USERNAME` and `YOUR-REPOSITORY` with your GitHub details.

### 2. Install dependencies

```bash
pip install tensorflow numpy matplotlib scikit-learn librosa SpeechRecognition pyttsx3 soundfile
```

### 3. Launch Jupyter Notebook

```bash
pip install notebook
jupyter notebook
```

### 4. Run the notebook

Open `Audio_Classification_STT_TTS_One_File.ipynb` and execute the cells in order. The dataset is downloaded automatically if it is not already available.

**Note:** STT requires internet access. TTS playback may behave differently in cloud notebooks than on a local computer.

## Applications

- Voice-command recognition
- Voice-controlled systems
- Basic speech interfaces
- Assistive technologies
- Human-computer interaction
- Smart-home command systems

## Future Enhancements

- Add more speech commands and audio classes.
- Improve accuracy through data augmentation and hyperparameter tuning.
- Integrate a graphical user interface.
- Support real-time microphone input.
- Explore advanced architectures such as recurrent neural networks or transformer-based audio models.

## Conclusion

This project demonstrates how deep learning can be applied to audio classification and combined with STT and TTS technologies to build a basic speech-enabled application. It provides practical experience in audio preprocessing, CNN model development, performance evaluation, and speech processing.

## Author

**Radhyeshyam lahoti**

## License

This project is intended for educational and academic purposes. Add a suitable open-source license if you want others to reuse or modify the code.
