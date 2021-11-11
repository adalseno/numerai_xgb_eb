# numerai_xgb_eb

"A rising tide lifts all ships"

The release of the new dataset creates a lot of opportunity for creative modeling and ensembles, but also presented difficulties for those with smaller systems.
The goal of this repo is to provide some of the tools and insight to build robust models with smaller systems.

Intermediate Example Script: A high performance, low memory (HPLM) approach to the Numerai competition.

Creatign a 16GB Feature Set
https://towardsdatascience.com/boruta-explained-the-way-i-wish-someone-explained-it-to-me-4489d70e154a


Simple to Intermediate Modeling

Simple: LightGBM Example Script

Simple 2: XGBoost updated Example Script

Intermediate: XGBoost with Era Boosting
 - Gotchas: Model Wobble, consider ensembling.

Advanced: Multiple Target and FN Ensembles

Parameter Tuning

Walk Forward Cross Validation:

Grid Search:

Feature Neutral Parameters:




Low memory, high performance example script for Numerai's smds

Using the "medium" feature set will result in a nearly 100% commit charge on a 16GB machine, it may crash some systems.

The goal is to develop optimzied feature sets for smaller and medium sized systems
Michael Oliver's improved Boruta Shap code, which accomodates era groupings, provides the "important, unimportant, and tenative" feature sets. 

 