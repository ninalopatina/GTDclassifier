# GTDclassifier
# Random Forest Classification of Global Terrorist Attacks
### Author: Nina Lopatina, Ph.D.

Global Terrorism Database (GTD) is an open-source database including information on terrorist events around the world from 1970 through 2014. Some portion of the attacks have not been attributed to a particular terrorist group. Original data and information can be found here: http://www.start.umd.edu/gtd/

One very exciting application of machine learning is in finding relationships in large datasets that would escape the human eye. Random Forest Classifiers (RFC) are supervised ensemble-learning models that perform well with large numbers of classes. In this notebook, I have used location, attack type, weapons used, description of the attack, etc. to build an RFC model that can predict what group may have been responsible for an incident. An accurate model would provide great utility in attributing terrorist attacks to the responsible groups.  

How to run:
1) Download data from this github folder or http://www.start.umd.edu/gtd/
2) Put data folder ('GTD_0617dist/') in a folder where you run the code & set this under dir_main in code block 2:
dir_main = '/Users/yourfoldershere/GTD/'
3) Make sure reading = 1 in code block 2

## Introduction: 

### Rationale:
The volume and sparseness of these data set provides an interesting multilabel classification challenge: the GTD Database has 170349 entries with 135 variables. 78306 of the recorded attacks are unattributed: that's 45% of the data. There are 3454 known groups. The median # of attacks per group is 2, and, 75% of groups commit 5 or fewer attacks. 

### Background:
Others have achieved fairly high accuracy attributing with this dataset. In "Terrorist Group Prediction Using Data Classification" by Faryal Gohar, Wasi Haider Butt, Usman Qamar, the authors  proposes  a  novel ensemble  framework  for  the  classification  and prediction of the terrorist group that consists of four base  classifiers  namely;  naïve  bayes  (NB),  K nearest neighbour (KNN), Iterative Dichotomiser 3 (ID3) and decision stump (DS). Majority vote based ensemble  technique (MV) is  used  to  combine  these classifiers. They obtained 92.75% accuracy with Naïve Bayes & 93.4% with their combined MV method. There are several reasons why this accuracy is higher than I would expect to find in solving this problem: they thresholded their data to the most frequently active groups: 6 or more attacks. Also, their research was done with the data up to 2012, which had much less data than the current data set. Further, they didn't specify how they split up their training & test sets or other specifics of their statistical methods. source: https://www.researchgate.net/publication/268445148_Terrorist_Group_Prediction_Using_Data_Classification

## Methods:

### Visualization:
I started with some numbers to grasp the nature of the problem: how many data points are there? how many groups? how many attacks are unattributed? how much of the data is missing? To better understand the data, I visualized some of the variables in bar plots and scatter plots to see how much variance there is within variables. Some variables I visualized between the attributed & unattributed attacks. 

To better understand the nature of the classification problem, I visualized the distribution of the known groups and the number of missing data points by variable.

In the process of constructing my model, I visualized some data about the features or of the classification results: correlation between model variables, feature contribution to the model. To better understand my model, I examined classification accuracy by sample size. For model comparison, I visualized accuracy & speed of the models I was comparing. 

Lastly, to verify my final results, I visualized locations of attributed and unattributed attacks that the model posited were by the same group.

### Data engineering:
I approximated 4606 missing values for latitude and longitude by passing city/country data through a geolocation package. 

### Model selection: 
A random forest is supervised classification algorithm. It is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting. The output is the mode of the class output of all the trees in the forest. I chose this model because it works well with sparse data sets with a high number of classes. Further, if there are enough trees in the forest, the classifier won’t overfit the model. Lastly, RFC will identify and select the most important features from the training dataset, a critical advantage in a datset with 32 features. 

### Model comparison:
For model comparison, I compared RFC to alternative models that can also be used for multiclass classification, the scikit-learn models: RandomForestClassifier, LinearDiscriminantAnalysis, DecisionTreeClassifier, GaussianNB, KNeighborsClassifier, MLPClassifier, LogisticRegression, LinearSVC, GradientBoostingClassifier.   

### Feature engineering:
I added absolute time to the separate year/month/day data (YMD), as well as day of the week. I used 32 features in my model. The top 5 features that contributed most to the model were: longitude, latitude, nationality, country, YMD, the latter of which I had added to the model. YMD contributed more than just the year. 

While Random Forest (RFC) performs feature importance + selection internally, I also tried removing the features that contributed least to the model to see how performance changed. It did not. I ultimately decided not to remove any features. 

## Results: 

### Classification accuracy ground truth, testing on attributed attacks:
RFC classification accuracy is 85.1% for a 10% test set. There's some overfitting: 99% accuracy for the training set. Most of the groups with few samples have low accuracy, and most of the groups with many examples have pretty high accuracy. But there are a few values that don't fit this trend, as highlighted in Section IX-4. 

### Classification of unattributed attacks:
There isn't a clear way to validate these results. First pass visualization of the location of unknown attacks with known attacks fits some expectations: Groups with a sizable number of attacks have more attacks attributed to them in a similar location as a function of other variables. Neo-nazi extremists have 4x as many unattributed as attributed attacks since their responsibility for their attacks tends to be discovered, not claimed. Boko Haram, despite their high level of terrorist activity, has no unattributed attacks: this is because they claim 100% of their attacks. That the model picked up on these differences (from the 'claimed' variable') is a testament to its viability. The model can be improved as detailed below.

## Conclusions:
Classification accuracy with only a handful of samples for many labels is no small feat: the labels are just so sparse. But figuring out where labels from high sample classes are getting mis-classified would increase classification accuracy, without relying on sparse data. Also, it is reasonable to assume that some unattributed attacks are by groups that were not previously known. The model does not currently address this. 

## Next steps: 
1. There are some highly active groups with low classification accuracy that I think could be improved via feature engineering or combining model outputs. I think this could improve the accuracy of the model. 
2. The model doesn't take into account groups that are unknown and also not in the previously known groups. I would improve this by factoring in classification confidence in the label: i.e., if the classification is below a certain bound, guess that it's an unknown unknown group that did this, rather than one of the pre-existing classes. Alternatively, I could combine model outputs with a classifier that can predict an unknown label. 
3. Tweaking the model: sweep parameters that control the complexity: Tree depth, n features to consider for the split, and the minimal number of samples at terminal leaf node.
4. Improving the accuracy of the latitude & longitude data fill. This would involve cross-referencing other geolocation packages.
5. Processing the text in the features I'm not using to add more data to the model
6. Adding more analysis to compare attributed and unattributed attacks by group. 