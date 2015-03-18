# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import svm
from sklearn import cross_validation
from sklearn import grid_search
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import KFold
from sklearn.feature_selection import SelectPercentile, f_classif
import os
import subprocess

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    http://scikit-learn.org/stable/auto_examples/plot_learning_curve.html
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

livemfcc = np.empty((0,13))
studiomfcc = np.empty((1,13))

if not (os.path.isfile('liveMFCC.csv') and os.path.isfile('studioMFCC.csv')):
    # need to regenrate data
    # should use yaafelib
    #(_,_,filenames) = os.walk("live").next()
    #subprocess.call(["yaafe", "-r", "44100", "-f", "mfcc: MFCC blockSize=1024 stepSize=512", "-b", "mfcc"] + filenames)
    #subprocess.call(['./features-mp3.sh'])
    for f in os.listdir("mfcc/live"):
        if os.path.isfile(os.path.join("mfcc/live",f)):
            print f
            csvfile = np.loadtxt(os.path.join("mfcc/live", f), delimiter=',', skiprows=5)
            livemfcc = np.vstack((livemfcc, np.transpose(csvfile.mean(axis=0))))
    np.savetxt('liveMFCC.csv', livemfcc, delimiter=',')
    print "\nlivemfcc shape: " + str(livemfcc.shape) + "\n"

    for f in os.listdir("mfcc/studio"):
        if os.path.isfile(os.path.join("mfcc/studio",f)):
            print f
            csvfile = np.loadtxt(os.path.join("mfcc/studio", f), delimiter=',', skiprows=5)
            studiomfcc = np.vstack((studiomfcc, np.transpose(csvfile.mean(axis=0))))
    np.savetxt('studioMFCC.csv', studiomfcc, delimiter=',')
    print "\nstudiomfcc shape: " + str(studiomfcc.shape) + "\n"
else:
    livemfcc = np.loadtxt('liveMFCC.csv',delimiter=',')
    studiomfcc = np.loadtxt('studioMFCC.csv',delimiter=',')

xx,yy=np.meshgrid(np.linspace(-5,5,500),np.linspace(-5,5,500))

data=np.concatenate([livemfcc,studiomfcc])
print "data shape: " + str(data.shape)

data=np.concatenate([livemfcc,studiomfcc])
y=np.concatenate([np.zeros((livemfcc.shape[0],1)),np.ones((studiomfcc.shape[0],1))])
print "y shape: " + str(y.shape)

# normalise
normalise = True
if normalise:
    data = preprocessing.MinMaxScaler().fit_transform(data)

###############################################################################
# Univariate feature selection with F-test for feature scoring
# We use the default selection function: the 10% most significant features
selectTopPercentile = True
selector = SelectPercentile(f_classif, percentile=65)
selector.fit(data, y.ravel())
###############################################################################

params = [{'C': [0.1, 1, 10, 20, 50, 70, 100, 1000], 'gamma': [0.08, 5, 10, 1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}]
clf = grid_search.GridSearchCV(svm.SVC(), params, n_jobs=8,cv=cross_validation.ShuffleSplit(data.shape[0], n_iter=10, test_size=0.2,random_state=0))
clf.fit(selector.transform(data) if selectTopPercentile else data, y.ravel())
#clf.fit(data,y.ravel())

print clf.grid_scores_
print clf.best_estimator_

Xtrain,Xtest,ytrain,ytest = cross_validation.train_test_split(data,y,test_size=0.2)
Xtrain.shape,ytrain.shape
Xtest.shape,ytest.shape

title = "Learning Curves (SVM, RBF kernel, $\gamma=" + str(clf.best_params_['gamma']) + ", C=" + str(clf.best_params_['C']) + "$) Score = " + str(clf.best_score_)
# SVC is more expensive so we do a lower number of CV iterations:
cv = cross_validation.ShuffleSplit(data.shape[0], n_iter=100,
                                   test_size=0.2, random_state=0)

plot_learning_curve(clf.best_estimator_, title, selector.transform(data) if selectTopPercentile else data, y.ravel(), [0,1.1], cv=cv, n_jobs=4)

plt.show()

clf = svm.SVC(C=clf.best_params_['C'], gamma=clf.best_params_['gamma'])
clf.fit(selector.transform(data) if selectTopPercentile else data, y.ravel())

#newlevel = np.loadtxt('newlevel.csv',delimiter=',')
#print newlevel
#print "New Level (studio): " + str(clf.predict(newlevel))
#wrathchild = np.loadtxt('wrathchild.csv',delimiter=',')
#print "Wrathchild (live): " + str(clf.predict(wrathchild))
#phantom = np.loadtxt('phantom.csv',delimiter=',')
#print "Phantom (live): " + str(clf.predict(phantom))

#testdata = np.loadtxt('')
#print "First 10: " + str(clf.predict(test_data))
#print "Expected: " + str(test_expected.ravel())

