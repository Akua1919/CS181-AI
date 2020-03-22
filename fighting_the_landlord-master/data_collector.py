from board import GameBoardData
from input_parser import InputParser
import farmer_agents
import landlord_agents
from constants import *
import random

from threading import Thread
from time import time, sleep


class DataCollector(object):

    @staticmethod
    def start_game(farmer_agent, landlord_agent, farmer_policy, landlord_policy, f,lunci, landlord_depth=None, farmer_depth=None):
        print('***GAME START***')
        board_state = GameBoardData()
        LandlordAgents = getattr(landlord_agents, landlord_agent)
        FarmerAgents = getattr(farmer_agents, farmer_agent)
        if landlord_agent == MCT_AGENT:
            landlord = LandlordAgents(LANDLORD, landlord_policy)
        else:
            landlord = LandlordAgents(LANDLORD, landlord_policy, depth=landlord_depth)
        if farmer_agent == MCT_AGENT:
            farmer_one = FarmerAgents(FARMER_ONE, farmer_policy)
            farmer_two = FarmerAgents(FARMER_TWO, farmer_policy)
        else:
            farmer_one = FarmerAgents(FARMER_ONE, farmer_policy, depth=farmer_depth)
            farmer_two = FarmerAgents(FARMER_TWO, farmer_policy, depth=farmer_depth)
        while not board_state.is_terminal:
            turn = board_state.turn
            action = None
            if turn == FARMER_ONE:
                action = farmer_one.get_action(board_state)
                print('farmer1 plays {0}'.format(action))
            if turn == FARMER_TWO:
                action = farmer_two.get_action(board_state)
                print('farmer2 plays {0}'.format(action))
            if turn == LANDLORD:
                action = landlord.get_action(board_state)
                print('landlord plays {0}'.format(action))
            board_state = board_state.next_state(action)
        #print('*** GAME {1} OVER, {0} WIN ***'.format(board_state.winner,lunci))
        print("\033[31;1m  *** GAME {1} OVER, {0} WIN ***  \033[0m".format(board_state.winner,lunci))
        winner = 'landlord'
        if board_state.winner in [FARMER_ONE, FARMER_TWO]:
            winner = 'farmer'
        f.write('[landlord]{0}-{1}-{2}::[farmer]{3}-{4}-{5}::[winner]{6}||\n'.format(landlord_agent,
                                                                                landlord_policy,
                                                                                landlord_depth,
                                                                                farmer_agent,
                                                                                farmer_policy,
                                                                                farmer_depth,
                                                                                winner))


if __name__ == '__main__':
    f = open('minimax_depth.txt', 'a')
    # for i in range(0, 100):
    #     DataCollector.start_game(ALPHA_BETA_AGENT, ALPHA_BETA_AGENT, EVALUATION, EVALUATION, f, landlord_depth=2,
    #                              farmer_depth=2)
    # for i in range(0, 100):
    #     DataCollector.start_game(ALPHA_BETA_AGENT, ALPHA_BETA_AGENT, EVALUATION, EVALUATION, f, landlord_depth=3,
    #                              farmer_depth=3)


    end2 = time()

    threads = []
    for i in range(20):
        t = Thread(target=DataCollector.start_game, args=(ALPHA_BETA_AGENT, ALPHA_BETA_AGENT, EVALUATION, EVALUATION, f, i,1,1))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    end3 = time()

    print('总共耗费了%.3f秒' % (end3 - end2))


    f.close()
    print ('minimax depth done')

