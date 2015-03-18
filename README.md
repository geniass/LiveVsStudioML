# LiveVsStudioML
Machine learning experiment that can classify music tracks as either live or studio versions

Usage
----
1. First, the user needs to manually classify the training data by creating 2 directories: ``live`` and ``studio`` and placing tracks in their corresponding directories. 
2. Run ``features-mp3.sh``. This will create ``live`` and ``studio`` directories inside a ``mfcc`` directory, and will extract features from the music tracks.
3. Run ``machinelearning.py``. As the name suggests, this does all the machine learning work.

This is still a work in progress so you need to modify ``machinelearning.py`` to get it to classify particular songs.


How it works
----
The ``features-mp3.sh`` script uses [Yaafe](https://github.com/Yaafe/Yaafe) to extract MFCC features from the music tracks.
Then ``machinelearning.py`` uses [scikit-learn](scikit-learn.org) to train a Gaussian SVM with the provided training data. It also does some preprocessing to improve results.
