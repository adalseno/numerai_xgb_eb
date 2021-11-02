# numerai_xgb_eb
Low memory, high performance example script for Numerai's smds

Problem Discription
Using the "medium" feature set will result in a nearly 100% commit charge on a 16GB machine, it may crash some systems depending on what else is open and how you run the code.

I'm currently working on an optimzied feature set that will leave just a little more headroom, but this will likely result is some loss of signal.
To generate the feature sets were created using Michael Oliver's improved Boruta Shap code. This version allows for grouping which is useful for eras.

Boruta Shap determins feature importance by creating a group of shadow features and measuring their maximum imporatance. If actual features don't exceed this threshhold 