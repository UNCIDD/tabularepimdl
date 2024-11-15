
from tabularepimdl.Rule import Rule
import numpy as np
import pandas as pd

class MultiStrainInfectiousProcess(Rule):
    """! Simple multi strain infectious process. Takes a cross protection matrix, a list of infection state 
    columns and an array of betas. Does not allow co-infections"""
    
    def __init__(self, betas: np.array, columns, cross_protect:np.array, s_st="S", i_st="I",  
                 r_st="R", inf_to="I", stochastic=False, freq_dep=True) -> None:
        """!
        Initialize with the columns. 

        @param betas a beta for each strain
        @param columns the columns for the infection process. Should be same length and order as betas
        @param cross_protect a N(strain)*N(strain) matrix of cross protections.
        @param s_st the state for susceptibles, assumed to be S
        @param i_st the state for infectious, assumed to be I
        @param r_st the state for immune/recovered, assumed to be R
        @param inf_to the state infectious folks go to, assumed to be I
        @param stochastic, is this rule stochastic process
        @param freq_dep is this a frequency dependent model
        """
        # Check that columns and betas have the same length
        if len(columns) != len(betas):
            raise ValueError(f"The length of 'columns' ({len(columns)}) must be equal to the length of 'betas' ({len(betas)}).")

        # Check that cross_protect is a square matrix with dimensions equal to the length of betas
        if cross_protect.shape[0] != cross_protect.shape[1] or cross_protect.shape[0] != len(betas):
            raise ValueError(f"'cross_protect' must be a square matrix with dimensions equal to the length of 'betas'. "
                             f"Expected {len(betas)}x{len(betas)}, got {cross_protect.shape}.")

        # If all checks pass, proceed with initialization
        super().__init__() 
        self.betas = betas
        self.columns = columns
        self.cross_protect = cross_protect
        self.s_st = s_st
        self.i_st = i_st
        self.r_st = r_st
        self.inf_to = inf_to
        self.freq_dep = freq_dep
        self.stochastic = stochastic

    def get_deltas(self, current_state, dt = 1.0, stochastic = None):
        """
        @param current_state, a data frame (at the moment) w/ the current epidemic state
        @param dt, the size of the timestep
        """
        #print('MultiStrain rule\n') #debug
        if stochastic is None:
            stochastic = self.stochastic

        if self.freq_dep:
            betas = self.betas/(current_state['N'].sum())
        else:
            betas = self.betas
        
        ##get number infectious of each type
        #print('Multistrain, current_state is\n', current_state)#debug
        infectious = ((current_state[self.columns] == self.i_st).multiply(current_state.N, axis=0)).sum(axis=0) #when no value 'I' exists in columns, infectious values are all zeros
        infectious = np.array(infectious)
        #print('infectious value: ', infectious) #debug
        if sum(infectious)==0: #there are cases that infecious==0
            return None
        
        ##calculate the strain specific FOI (force of infection) for each row for each strain.

        #first get the cross protections
        row_beta_mult = 1-current_state[self.columns].apply(
            lambda x: (np.array((x==self.r_st)) * self.cross_protect).max(axis=1),
            axis=1,
            result_type='expand'
            )
        #print('row beta mult is\n', row_beta_mult) #debug
        #now we turn that into a strain specific probablity of infection

        ## This line does a few things:
        #   - it calculates cros protection
        #   - it makes the FOI 0 for folks when folks are not susceptible to a strain
        
        row_beta = (row_beta_mult * betas *
                     (current_state[self.columns]==self.s_st).values)
        #print('row beta is\n ', row_beta) #debug
        # This makes the probablity of infectoin 0 when folks are infected with a different strain...
        # i.e., no coinfections!
        # Maybe slightly problematic given the strong assumption of only being infected with 1 strain...max() is better.
        row_beta = row_beta.multiply(
                1-(current_state[self.columns] == self.i_st).max(axis=1), axis=0 #changed sum() to max() to make sure only one strain stands out
            )
        prI = 1-(np.exp(-dt*row_beta)).apply(lambda x: np.power(x, infectious), axis=1)
        #print('prI is ', prI) #debug
        #deltas can only happen to rows where we have and FOI>1
        deltas =  current_state.loc[prI.sum(axis=1)>0] 
        prI = prI.loc[prI.sum(axis=1)>0] 
        prI.columns = self.columns ##Makes some later stuff easier
        #print('delta is\n', deltas) #debug
        ## now do the infectious process.
        if not stochastic:
            #first the subtractions
            deltas = deltas.assign(
                N= -deltas.N*(1 - (1-prI).product(axis=1))
            )

            #now allocate those cases prportional to prI
            tmp = pd.DataFrame()
            for col in self.columns:
                tmp2 = deltas.assign(
                        N=-deltas['N'] * (prI[col]/prI.sum(axis=1)))
                
                tmp2[col] = self.inf_to

                tmp = pd.concat([
                    tmp, 
                    tmp2
                ])
                
            deltas = pd.concat([deltas, tmp]).reset_index(drop=True) #need to reset index
            
        else:

            N_index = deltas.columns.get_loc('N')
            #multinomial draw for each delta and create the appropriate deltas.
            for i in range(prI.shape[0]):
                tmp = np.random.multinomial(deltas['N'].iloc[i], np.append(prI.iloc[i].values,[0]))
                #print("tmp is\n", tmp) #debug
                deltas.iloc[i,N_index] = -tmp[:-1].sum()
                #print('detla is\n', deltas) #debug
                ##do the additions
                for j in range(prI.shape[1]):
                    toadd = deltas.iloc[[i]]
                    toadd = toadd.assign(N=tmp[j])
                    toadd[self.columns[j]] = self.inf_to
                    deltas = pd.concat([deltas, toadd])
            deltas = deltas.reset_index(drop=True) #need to reset index of dataframe  

        
        deltas = deltas[deltas['N']!=0]
        #print('multirule delta is\n', deltas) #debug
        return deltas


    def to_yaml(self):
        rc = {
            'tabularepimdl.MultiStrainInfectiousProcess': {
                'betas': self.betas,
                'columns': self.columns,
                'cross_protect': self.cross_protect,
                's_st': self.s_st,
                'i_st': self.i_st,
                'r_st': self.r_st,
                'inf_to': self.inf_to,
                'freq_dep':self.freq_dep,
                'stochastic':self.stochastic
            }
        }

        return rc    