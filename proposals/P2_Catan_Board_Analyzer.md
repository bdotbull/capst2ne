# Capstone 2 - Proposal 2 - Catan Board Analyzer
#
## Why am I interested?
Catan, previously known as The Settlers of Catan or simply Settlers, is a multiplayer board game designed by Klaus Teuber, and first published in 1995 in Germany by Franckh-Kosmos Verlag as Die Siedler von Catan. Players take on the roles of settlers, each attempting to build and develop holdings while trading and acquiring resources. [*](https://en.wikipedia.org/wiki/Catan)

In the Before Times (pre-covid), I would play Catan with close friends as often as I could and it always brought me great joy. A perfect mix of probability and strategy, Catan is ripe for analysis.

With an astronomical number of possible board layouts, a player's initial move (placement of two settlements and two roads) can be enough to run away with a victory or slowly decend into fruitless poverty. By looking at available resources and taking into account the numbers on each and every resource, the program I create in this project will give suggestions for starting settlement and road layouts for a set of strategies (longest road, development cards, mass expansion/maximal resource earnings, port play).


#
## What will the data look like?
The desired goal would be to have a program which takes a given board state and potential settlement location as input and outputs information related to the move.
For example, the information related to the move could be 
* expected resources for a given turn (perhaps weighted by scarcity)
* impact on a specific strategy (ore will have little or no impact on expansion/road building)

#
### If time permits I would like to...
Have the program output a heatmap of suggested moves for a given strategy

#
## What will I learn?
Generating regression algorithms
