# Capstone 2 - Proposal 1 - RL+CV
#
## Why am I interested?
Two topics that fascinate me are Rocket League and Computer Vision. One is my ultimate stress release (RL) and the other a current fascination (CV). I would like to combine my interests in a fun and interesting way. One such way would be to create a model which attempts to label a car with varying characteristics.  The ultimate goal would be to input a screenshot of rocket league and have an output be a list of cars on screen and as well as the name and market value for items associated with each car (wheels, body, paint, etc).  Walking that grand goal back a bit, for the scope of this project, I would start with detecting cars and labeling 1 or 2 categories (body, wheels).

This project would be progress towards much larger projects (capstone 3 and beyond)

#
## Where will I get the data?
I just so happen to have a local database with 50,000+ replays from [ballchasing.com](https://ballchasing.com/), which I can load into the game client to stream images or make screenshots.

Items and Prices could be found on any RL trading site.  The price for a particular item could be averaged between:
* [AOEAH](https://www.aoeah.com/rocket-league-items/steam%20pc)
* [RL.EXCHANGE](https://rl.exchange/)
* [RL Garage](https://rocket-league.com/trading)
#
## What will the data look like?
The replay files are big nasty JSONs that can be cleaned with a similar process as my first capstone 1.  The replays would mainly be for getting car and item labels.

Most likely, I will create and label my own dataset (maybe using [sense](https://sixgill.com/try-sense-data-annotation/) or [scribe](https://www.clarifai.com/label) or both)
###### (っ◔◡◔)っ ♥ free trials ♥

#
### If time permits I would like to...
#### Add categories!  
##### Because I will be starting so small (correctly labeling 3 body types -- Dominus, Octane, Fennec), I would like to add categories for wheels, and maybe recognize color data for painted bodies and decals.

#
## What will I learn?
ＭＯＲＥ ＰＹＴＨＯＮ  
Computer Vision
