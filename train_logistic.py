import numpy as np
from datetime import datetime
import load




        
        
class model_logistic:
    def __init__(self,all_match_data,player_list,time):
        self.all_match_data = all_match_data
        self.player_list = player_list
        self.match_weight = {'Grass':1,
                             'Clay':1,
                             'Hard':1,
                             'Carpet':1
                            }
        self.gamma = 0
        self.time = time
        self.N = len(self.player_list)
        self.ratings = np.ones(self.N)
        self.learning_rate = 0.1
            
    #return days between t2 and  t1, time format is 2023-1-1
    def time_duration(self, t2, t1):
        # Parse the string dates into datetime objects
        date_format = "%Y%m%d"
        date1 = datetime.strptime(t1, date_format)
        date2 = datetime.strptime(t2, date_format)

        # Calculate the difference between dates
        delta = date2 - date1

        # Return the number of days as an integer
        return delta.days
        
        
    def log_likely_hood(self,ratings,one_match_data):
        player1 = one_match_data['winner_name']
        player2 = one_match_data['loser_name']
        i1 = self.player_list.index(player1)
        i2 = self.player_list.index(player2)
        g1 = one_match_data['winner_games']
        g2 = one_match_data['loser_games']
        tk = one_match_data['tourney_date']
        delta_t = self.time_duration(self.time,tk)
        assert delta_t >= 0
        theta1 = ratings[i1]
        theta2 = ratings[i2]
        #print('theta1',theta1)
        m = self.match_weight[one_match_data['surface']]
        gamma = self.gamma
        f = np.exp(theta1-theta2) / (1+np.exp(theta1-theta2))
        w = m*np.exp(-gamma*delta_t)
        return w*np.log(f)
    
    def grad_log_likely_hood(self,ratings,one_match_data):
        player1 = one_match_data['winner_name']
        player2 = one_match_data['loser_name']
        i1 = self.player_list.index(player1)
        i2 = self.player_list.index(player2)
        g1 = one_match_data['winner_games']
        g2 = one_match_data['loser_games']
        tk = one_match_data['tourney_date']
        delta_t = self.time_duration(self.time,tk)
        assert delta_t >= 0
        theta1 = ratings[i1]
        theta2 = ratings[i2]
        m = self.match_weight[one_match_data['surface']]
        gamma = self.gamma
        f0 = 1- np.exp(theta1-theta2) / (1+np.exp(theta1-theta2))
        w = m*np.exp(-gamma*delta_t)
        return w*f0, -w*f0, i1, i2
        
    
    def loss_func(self,ratings):
        loss = 1
        for one_match_data in self.all_match_data:
            if self.time_duration(self.time, one_match_data['tourney_date'])>=0:
                loss += self.log_likely_hood(ratings,one_match_data)
        return loss
    
    def grad_loss_func(self,ratings):
        grad_loss = np.zeros(self.N)
        for one_match_data in self.all_match_data:
            if self.time_duration(self.time, one_match_data['tourney_date'])>=0:
                gi1, gi2, i1, i2 = self.grad_log_likely_hood(ratings,one_match_data)
                grad_loss[i1]+=gi1
                grad_loss[i2]+=gi2
        return grad_loss
    
    def update(self):
        grad = self.grad_loss_func(self.ratings)
        self.ratings += self.learning_rate * grad
        #print('grad',grad)
        
    def optimize(self):
        print('init',self.ratings[0])
        for i in range(10):
            self.update()
            print('loss',self.loss_func(self.ratings))
            print(self.player_list[0],self.ratings[0])
        #print(self.ratings)


if __name__ == '__main__':
    file_path = 'cleaned_atp_matches_2020.csv'
    all_match_data = load.get_all_match_data(file_path)
    player_list = load.get_player_list(all_match_data)
    model = model_logistic(all_match_data,player_list,'20201231')
    model.optimize()
    
    
    # print sorted players
    # Combine the lists, sort by the model ratings in descending order, then unzip
    combined = zip(player_list, model.ratings)
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
    player_list_sorted, ratings_sorted = zip(*sorted_combined)

    # Convert the tuples back to lists (if necessary)
    player_list_sorted = list(player_list_sorted)
    ratings_sorted = list(ratings_sorted)

    print("Sorted player list:", player_list_sorted[:3])
    #print("Sorted ratings:", ratings_sorted)