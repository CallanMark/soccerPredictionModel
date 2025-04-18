# soccerPredictionModel
Project using machine learning and stats to predict outcomes in soccer matches 

# Proof of concept 
- Starting of the project with trying to model probabalties for macnchester united 
- I got all the data from fbref.com and cleaning the data using 'util.py'

# Notes 
- This is a somewhat ok proof of concept to start with but needs more data and better feature engineering (Player fatigue , home / away , player position )
- Should transiton to using json instead of csv to reduce bloat (if continue using csv will end up with huge number of files)
- Need to automate scraping data as current method of getting data is not sustainable 
- Needs to be optimzied for outcomes that can be leveraged on polymarket e.g match outcomes , -> when game states change how that affects outcome probabalties

# Example output

![output] [/data/images/prediction_example.png]