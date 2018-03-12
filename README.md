**Character recognition with a convolutional neural network (CNN) implemented with Keras.**
by @Seattlefella
========

This project is my realization of a convolutional neural network (CNN) to decode the MNIST handwriting data set.

The Keras  solution demonstrates how to set up the variables, output data to tensorboard
and scores 99.0 to 99.5% accuracy.

The keras solution implements a number of optional features that the user 
may find useful.

This work book expects to read from an excel work book all of the data that will be needed to run the selected model. Once all runs are completed the session data will be saved to a results tab in the workbook.
This allows the user to set up a number of models, vary input and let run all night, saving the results for later processing.



Including:

•    learning rate min/max   
•    Number of epochs to run    
•    Batch Size
•    Dropout rate
•    The specific model to be run
•    Initial model to start the given run on.
•    Measuring the time each run takes.
    
The work book also enables several call back functions:

•    Learning rate scheduler
•    Modified tensor board logging
•    Modified model checkpoint-save
•    Print some interim data 

###Software
- Tensorflow 1.5, Keras, numpy, pandas

###Copyright
This project is released into the public domain. For more information see LICENSE.