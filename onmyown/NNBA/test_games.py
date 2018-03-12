"""
In this file is where I can run the outcomes of various games, printing the result. 
"""

import NNBA
import numpy as np

todaysNN  = NNBA.make_network('1517_boxscores.csv')


teams, stats =NNBA.make_prediction_data('3-12teamstats.csv')


#Example, trying to make 4 games 
game1 = NNBA.game_maker(stats.values[23],stats.values[18])
game2 = NNBA.game_maker(stats.values[27],stats.values[1])
game3 = NNBA.game_maker(stats.values[21],stats.values[29])
game4 =  NNBA.game_maker(stats.values[28],stats.values[7])



print("The predictions for these games are: " + str(todaysNN.predict(np.array([game1,game2,game3,game4]))))