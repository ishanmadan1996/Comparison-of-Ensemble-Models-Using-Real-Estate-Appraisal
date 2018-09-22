# Comparison of Ensemble Models Using Real Estate Appraisal

In this project we made a comparative analysis of four ensemble methods, viz. Bootstrap Aggregating, Random Forest, Gradient Boosting and Extreme Gradient Boosting for real estate appraisal. The data extracted from sources such as 99acres was utilized for training and testing the above-mentioned ensemble models and comparing their performances. Grid search was used for fine-tuning the hyper-parameters of the learning models. Also, preprocessing techniques such as dimensionality reduction and one-hot encoding were used for improving the accuracy of the models. 

## Extracting the data from https://www.99acres.com/

Clone the repository:

```
git clone https://github.com/ishanmadan1996/Comparison-of-Ensemble-Models-Using-Real-Estate-Appraisal.git
```

Install the pre-requisite libraries:

```
pip install requirements.txt
```

Run the Python script for data extraction using:

```
python Data_Extraction.py
```

For cleaning the data to comply with the machine learning model, run the data cleaning script using:
```
python Data_Cleaning.py
```
Execute the scripts given in 'Ensemble Models' folder to train different ensemble models using the extracted and cleaned data.

## Built With

* [Python](https://www.python.org/doc/) - The scripting language used

* [Scikit-learn](http://scikit-learn.org/stable/supervised_learning.html#supervised-learning) - The machine learning framework used.

## Authors

* **Ishan Madan** - [ishanmadan1996](https://github.com/ishanmadan1996)
* **Prathamesh Kumkar** - [iPrathamKumkar](https://github.com/iPrathamKumkar)
* **Ashutosh Kale** - [ak1997](https://github.com/ak1997)