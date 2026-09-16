raw data:
12000 rows, 1 col (form parts 1-4)

each of these cells represents a recording from a patient. 
3 rows x N columns in each cell:
row 0: PPG 
row 1: ABP
row 2: ECG

The columns in each cell (matrix) may vary. 
The N cols in a certain matrix represent the number of samples, where 125 samples were taken per second for each of ppg, abpg, ecg. 
e.g. for a 10 second recording, there will be 1250 samples (1250 columns)

Goal: split all recordings into 5 second windows (625 columns)
Thus, from each matrix cell in the raw data, we may obtain multiple 5 second windows 
basic case: cell contains 625 columns for each row of ppg, abp, ecg 
            (perfect 5 second window)

If a raw cell were to be a 10 second window, then this would produce 2 rows for a dataset
(two 5 second windows)

However, we want to have 2 different data sets of 5 second windows, one for ppg and another for abp data
Once we have these datasets we want to extract systolic and diastolic pressure from the ABP signals dataset by finding the peaks (systolic) and valleys (diastolic). if a window contains noisy signals or physiologically impossible values, we can discard the noisy data. 

** Questions:
how do we determine noisy signals or phyisologically impossible values? 
i.e. what are the "limits"
** 

Finally, we must join the systolic and diastolic values from those same windows to the ppg dataset (which will be our actual dataset to work on)
the systolic and diastolic values are our labels!

in all, our final dataset will contain 625 ppg samples (625 columns), which is a 5 second window, and 2 columns for the systolic and diastolic values (labels)
