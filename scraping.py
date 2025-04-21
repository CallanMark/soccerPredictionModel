import selenium as scraper
import json 
import os 
import time
from util import formatJson
'''
scraping.py | This file is for scraping data from Fbref.com although we can use other data sources if required 
We need to scrape team data as well as opponent data , Our ultimate goal is to predict actionable outcomes e.g goals scored , winner of match , assists , ,etc ....
'''

url = 'https://fbref.com/<relevant link>'

def getTeamStats (url,filepath,teamName):
    '''
    args : 
    url  = 'https://fbref.com/<link to teamName homepage>'
    filepath = data/teamname/json
    teamName = The directory in shorthand that you wish to save the team data to , or what that directory is already called if it exits
    '''
     # If directory does not exist create it 
    if not os.path.isdir(filepath):
        os.makedirs('data/',teamName,'/json')
        os.chdir('data/',teamName,'/json')
        print("Could not locate ",filepath , "Created and cd'd to ",os.getcwd())
        os.system('touch teamStats.json')  # Create teamStats.json to save data to
        print("Creating teamStats.json") 
        filepath = 'data/',teamName,'json'

    '''
    Psuedocode implementation
    scrape.url(url)
    data = capture.frompage(,#,Nation,Pos,Age,Min,Gls,Ast,PK,PKatt,Sh,SoT,CrdY,CrdR,Touches,Tkl,Int,Blocks,xG,npxG,xAG,SCA,GCA,Cmp,Att,Cmp%,PrgP,Carries,PrgC,Att,Succ)
    for all rows 
    append.
    if data is (malformed):
        util.formatjson(data)
    '''
    data = ''  #  Placeholder for actual data 
    with open(os.path.join(filepath, "teamStats.json"), "w") as f:
           json.dump(data, f)
    print("Saving team data for " , teamName ,os.getcwd() ,"teamStats.json")
    
def getTeamGameHistory(url,filepath , teamName):
     '''
     args :
     url  = 'https://fbref.com/<link to teamName most recent match report>'
    filepath = data/teamname/json
    teamName = The directory in shorthand that you wish to save the team data to , or what that directory is already called if it exits
     '''

     '''
     Pseudocode implemenation : 
     gameweek = scraper.scrape('gameweek')
     gameweek = parse to integer value # i.e. if gameweek is 31 , gameweek = 31
     for game in range(gameweek): 
        
     '''
