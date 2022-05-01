# Covid-19 Detection using Chest X-Ray


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TEC2gr6KILXHPzBMFObF3EFKH5zMRFH-?usp=sharing)
[![Google Chrome](https://img.shields.io/badge/Google%20Chrome-4285F4?style=for-the-badge&logo=GoogleChrome&logoColor=white)](https://harshitakalani.github.io/Covid-19-Detection.github.io/)
[![LaTeX](https://img.shields.io/badge/latex-%23008080.svg?style=for-the-badge&logo=latex&logoColor=white)](https://github.com/Pranavchiku/Covid-19-Detection.github.io/blob/90057b83233dd44ba541d11faaedfaf6a96b5588/B20CS016_B20CS019_PRML_COURSE_PROJECT.pdf)
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
## Source of Dataset
[X Ray Images](https://data.mendeley.com/datasets/8h65ywd2jr/3)\
[Lung segmentation images](https://www.kaggle.com/code/nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset/data)
## Built using:
- [Scikit Learn: ](https://scikit-learn.org/stable/) ML Library used
- [TensorFlow Keras: ](https://www.tensorflow.org/api_docs/python/tf/keras) ML Libraries used
- [HTML: ](https://developer.mozilla.org/en-US/docs/Web/HTML) HTML documentation used
- [Javscript: ](https://developer.mozilla.org/en-US/docs/Web/JavaScript) Javscript framework used
- [Pandas: ](https://pandas.pydata.org/) Python data manipulation libraries
- [Seaborn: ](https://seaborn.pydata.org/) Data visualisation library
- [OpenCV2: ](https://pypi.org/project/opencv-python/) Image Preprocessing library
## Pipeline:
### [1. Covid-19 Detection using Chest X-Ray.ipynb](https://github.com/HarshitaKalani/Covid-19-Detection.github.io/blob/bd96ea2f33edb90149cf1f73b2c1bb0f9f9cf441/COVID_19_Detection_using_Chest_X_Ray.ipynb)
This is the main file containing EDA, preprocessing, application of various machine learning and deep learning models.
- Installing libraries and dependencies
- Importing the dataset
- Exploratory Data Analysis and Visualisation
- Data Preprocessing
  - Normalisation
  - Resizing image
  - Lung Segmentation
- Machine Learning Models
  - Dimensonality reduction using PCA
  - Decision tree Clasifier
  - Random Forest Classifier
  - XG Boost
  - Light GBM
  - Support vector machine
  - Logistic Regression
  - Comparitive Analysis
- Deep learning Models (CNN)
  - VGG 19
  - Resnet50
  - EfficientNet B3
  - U Net (For lung Segmentation)
  - Saving .json file and weights for deployment
### [2. Lung Segmentation.ipynb](https://github.com/Pranavchiku/Covid-19-Detection.github.io/blob/e08fa7499a7e5ecd283af59e45b3fe134f65cefc/Lung_Segmenation.ipynb)
Lung segmentation is performed in this file, using pre-trained weights of u net model.
- U Net 
- Segmenting original images
## How to run:
- Run the cells in main file according to above mentioned pipeline
## Collaborators:
| Name | Year | Branch|
| ------------- | ------------- | ------------- |
| Harshita Kalani (B20CS019)  | Sophomore  | CSE |
| Pranav Goswami (B20CS016) | Sophomore  | CSE |

