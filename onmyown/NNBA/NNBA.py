import numpy as np

def make_network(FILENAME):
    from pandas import read_csv,get_dummies
    import numpy as np
    from sklearn import cross_validation
    from sklearn.neural_network import MLPClassifier
    """Given the csv input of all the box scores, arrange it such that the home and away teams are lined up, 
    unnecessary columns removed, and proper hot encoding is done. Other stuff too probably.
    
    Note that this data has already been doctored from its original form, taking out most unnecessary columns but
    those could be useful later on.
    
    
    Parameters
    ----------
    FILENAME : file
        The csv of the data, NBA box scores over the past 2 seasons is what I used from nbastuffer.com
        
    Returns
    -------
    
    model : object
        MLP which can predict the outcome of NBA games
    """
    #Read in file, remove attempted and # and only account for % since that's more predictive in nature. 
    data = read_csv(FILENAME)
    data['3P%'] = np.divide(data['3P'].values,data['3PA'].values)
    del data['3P'],data['3PA']
    data['FG%'] = np.divide(data['FG'].values,data['FGA'].values)
    del data['FG'],data['FGA']
    data['FT%'] = np.divide(data['FT'].values,data['FTA'].values)
    del data['FT'],data['FTA']
    data = get_dummies(data)
    del data['VENUE_Home'],data['VENUE_Road']
    #print(data)
    
    dat = []

    for i in range(len(data.values)):
        data.values[i] = np.reshape(data.values[i],newshape=[1,len(data.values[i])])
    for p in range(int(len(data.values)/2)):
        fullboxgame = np.concatenate((data.values[2*p],data.values[(2*p)+1]))
        dat.append(fullboxgame)
            
    """At this point in the data dat is an array arranged as 
        OR  DR  TOT   A  PF  ST  TO  BL  PTS       3P%       FG%       FT% x2  (for road then home )
    so next up is to hot pull road and home points 
    
    road points is column 8 and home points is column 20. 
    
    So grab those from dat and make y. 
    
    X is every other column, so merge everyone else 
    
    """
    dat = np.array(dat)    
    y = []
    for j in range(len(dat)):
        roadpts = dat[j,8]
        homepts = dat[j,20]
       # print(home,road)
    #distinguish home win or road win. 
        if homepts > roadpts:
            y.append(np.array([0,1]))  # 0 1 mean home won
        else:
            y.append(np.array([1,0]))  #  1 0 means away won. 
    y = np.array(y)
    X1 = np.concatenate((dat[:,0:8],dat[:,9:20]),axis=1)
    X = np.concatenate((X1,dat[:,21:24]),axis=1)    #need to go one further column to snag HFT% 
    X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.4)
    model = MLPClassifier()
    model.batch_size = 200
    model.n_layers_ = 100
    model.n_outputs_= 300
    model.fit(X_train,y_train)
    print(model.score(X_test,y_test))
    return model
    


def make_prediction_data(filename):
	"""Take the current season stats from basketball-reference (https://www.basketball-reference.com/leagues/NBA_2018.html#team-stats-base:) with the converted to CSV version
	of these statistics. This function properly aligns the columns to be compared to the training set, and converts the necessary statistics to per game stats. 

	This does not account for streaks, injuries, etc., but serves as a good start!

	Parameters
	----------
	filename : string
		The csv file from basketball-reference with the current team stats. 

	Returns
	-------
	teams : list
		The list of NBA teams, and their respective numbers of reference. 
	stats : array
		Numpy array containing the stats in the proper order for game predictions. 
	"""
	from pandas import read_csv,get_dummies
	import numpy as np
	from sklearn import cross_validation
	data = read_csv(filename)
	data['ORB'] =  np.divide(data['ORB'].values,data['G'].values)
	data['DRB'] =  np.divide(data['DRB'].values,data['G'].values)
	data['TRB'] =  np.divide(data['TRB'].values,data['G'].values)
	data['AST'] =  np.divide(data['AST'].values,data['G'].values)
	data['STL'] =  np.divide(data['STL'].values,data['G'].values)
	data['BLK'] =  np.divide(data['BLK'].values,data['G'].values)
	data['TOV'] =  np.divide(data['TOV'].values,data['G'].values)
	data['PF'] =  np.divide(data['PF'].values,data['G'].values)
	teams  = data['Team']
	data = data[['ORB' , 'DRB'  ,'TRB' ,  'AST' , 'PF' , 'STL' , 'TOV' , 'BLK' ,'3P%','FG%' ,'FT%']]
	print("Here is every teams index value: ")
	print(teams)
	return teams,data

def game_maker(roadteam,hometeam):
    """After creating a properly formated table, this concats the desired teams so they can be predicted. 
        Based on get team index # based on output of predictor, and make it the input for stats ie GSW are stats[0].
        and so on!

    Parameters
    ----------

    roadteam : array
    	Numpy array of the road team's attributes. 

    hometeam : array
    	Numpy array of the home team's attributes. 

    Returns
    -------

    game : array
    	Concatenated array of the road team + home team, to be easily plugged into model.predict. 

    """
    game = np.concatenate((roadteam,hometeam))
    return game