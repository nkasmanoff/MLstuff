Contained in this respository is the python file containing the functions I use to accumulate the necessary data to build a multi-layer perceptron (MLP) artificial neural network as a means to predict the outcome of current NBA games. The training, testing, and validation data comes from nba stuffer, a pay for usage website containing the box score of every single nba game over the past 2 seasons. This data utilizes the shooting percentages, total rebounds, assists, steals, turnovers, blocks, fouls, and steals, as "X," the input data while the output is a one-hot encoded [1,0] or [0,1], signifying a victory of either the home or away team. By training over these box scores, it should in theory be possible to have a somewhat guided prediction as to who should win an NBA game based on the two competing teams accumulated team statistics over the season. 



Current record (3/14):

20-9
