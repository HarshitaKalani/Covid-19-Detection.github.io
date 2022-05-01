# Covid-19 Detection using Chest X-Ray


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TEC2gr6KILXHPzBMFObF3EFKH5zMRFH-?usp=sharing)
## Overview:
COVID- 19 global pandemic affects health care and lifestyle worldwide,
and its early detection is critical to control cases’ spreading and mortality.
The actual leader diagnosis test is the Reverse transcription Polymerase
chain reaction (RT-PCR), result times and cost of these tests are high,
so other fast and accessible diagnostic tools are needed. This projects’
approach uses existing deep learning models (VGG19 and U-Net) and
various machine learning models to process these images and classify them
as positive or negative for COVID-19. The proposed system involves a
preprocessing stage with lung segmentation, removing the surroundings
which does not offer relevant information for the task and may produce
biased results; after this initial stage comes the classification model trained
under the transfer learning scheme; and finally, results analysis. The best
models achieved a detection accuracy of COVID-19 around 96%
## Dataset Description
The dataset contains two main folders, one for the X-ray images, which includes two separate sub-folders of 5500 Non-COVID images and 4044 COVID images.
## Built using:
- [Scikit Learn: ](https://scikit-learn.org/stable/) ML Library used
- [TensorFlow Keras: ](https://www.tensorflow.org/api_docs/python/tf/keras) ML Libraries used
- [HTML: ](https://developer.mozilla.org/en-US/docs/Web/HTML) HTML documentation used
- [Javscript: ](https://developer.mozilla.org/en-US/docs/Web/JavaScript) Javscript framework used
- [Pandas: ](https://pandas.pydata.org/) Python data manipulation libraries
- [Seaborn: ](https://seaborn.pydata.org/) Data visualisation library
- [OpenCV2: ](https://pypi.org/project/opencv-python/) Image Preprocessing library
## Pipeline:
### 1. Covid-19 Detection using Chest X-Ray.ipynb
This is the main file with all the preprocessing, EDA, various Machine learning and Deep Learning Models.
- Installing libraries and dependency
- Importing the dataset - [Flight Price Prediction Dataset ](https://drive.google.com/drive/folders/1tHNt5vPyCyKRQIitvGmf48AI2tna5xSk) 
- Exploratory Data Analysis and Visualisation
- Data Preprocessing - Basic preprocessing and cleaning the dataset
- Extra regressor model to determine feature importance
- Dividing the dataset into train and test
- Applying Machine Learning models
- Applying Deep Learning models
  - Saving the weights and .json file for deployment
## How to run:
- Run the cells according to above mentioned pipeline

