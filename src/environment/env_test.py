from pathlib import Path
import numpy as np
import gym
from gym import spaces, logger

from gym.utils import seeding
import matplotlib.pyplot as plt
from pycel import ExcelCompiler


class Parameters:
    # (Avoid sampling random variables here: they would not be resampled upon reset())
    # problem-specific parameters

    techs = 3  # number of technologies (Offshore wind power, blue hydrogen, green hydrogen)
    # fmt: off
    reward_types = 6 # capital expenditure (capex), operating expenditure (opex), revenue, carbon emissions, total jobs supported, total economic impact
    steps_per_episode = 20 # number of years in the planning horizon (2031 -> 2050 = 20)
    # fmt: on
    # This 'Pathways to Net Zero' environment manipulates a spreadsheet loaded in memory. The following 20 columns correspond to years 2031 to 2050 in tabs named 'Outputs' and 'CCUS':
    # fmt: off
    pathways2Net0ColumnInds = np.array(['P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF','AG','AH','AI'])
    # fmt: on
    # The following 20 rows correspond to years 2031 to 2050 in tabs named 'BREEZE', 'GALE', and 'STORM':
    pathways2Net0RowInds = np.arange(36, 36 + steps_per_episode)
    # pathways2Net0ColumnInds[state.step_count] and pathways2Net0RowInds[state.step_count] will locate the current year's column / row respectively
    
    # Multiplicative noise is applied to all costs. The parameters of this randomisation are:
    noise_mu = 1.0
    noise_sigma = 0.1  
    noise_clipping = 0.5  # (i.e., costs are reduced by 50% at the most)
    noise_sigma_factor = np.sqrt(0.1) # this factor is applied to make CCUS capex & opex less volatile than other costs  
    # The costs in the Carbon capture utilisation and storage (CCUS) tab to be randomised are capex, opex, and carbon price, with these row numbers:
    pathways2Net0RandomRowInds_CCUS = np.array([23, 24, 26])
    # The costs in the 'Outputs' tab to be randomised are Offshore wind - Devex, Capex, and Opex, Green Hydrogen - Capex, Fixed Opex, and Variable Opex, Blue Hydrogen - price, Gas feedstock price, Capex, Fixed opex, Variable opex, and Natural gas cost, with these row numbers:
    # fmt: off
    pathways2Net0RandomRowInds_Outputs = np.array([148, 149, 150, 153, 154, 155, 158, 159, 163, 164, 165, 166])
    # fmt: on
    # multiplicative noise's mu and sigma, and clipping point:
    noise_mu = 1.0
    noise_sigma = 0.1  # or try 0.1, 0.0, np.sqrt(0.001), 0.02, np.sqrt(0.0003), 0.015, 0.01, np.sqrt(0.00001), 0.001
    noise_clipping = 0.5  # or try 0.001, 0.1, 0.5 (i.e., original costs are reduced by 50% at the most)
    noise_sigma_factor = np.sqrt(0.1) # as in https://github.com/rangl-labs/netzerotc/issues/36, CCUS capex & opex (CCUS row 23 and 24) should have smaller standard deviations
    stochastic_sigma = False  # set to False to use one single noise_sigma; set to True to randomly switch between two different std:
    # noise_sigma_low = 0.001
    # noise_sigma_high = np.sqrt(0.00001)
    # OR, sample a sigma from a uniform distribution centered at noise_sigma with total 2-side range of noise_sigma_range:
    noise_sigma_range = 0.002
    noise_observability = False  # set to True to make the observation_space contain randomized costs/prices; set to False to restrict the observation_space to contain only the state.step_count

    

class State:
    def __init__(self, seed=None, param=Parameters()):
        np.random.seed(seed=seed)
        self.initialise_state(param)

    def initialise_state(self, param):
        # create local copy of spreadsheet model to be manipulated
        self.pathways2Net0 = param.pathways2Net0

        # create an array of costs for the current year and populate with 2030 costs (column 'O' in 'CCUS' and 'Outputs' tabs):
        self.randomized_costs = np.ones(
            len(param.pathways2Net0RandomRowInds_CCUS)
            + len(param.pathways2Net0RandomRowInds_Outputs)
        )
        for costRowID in np.arange(len(param.pathways2Net0RandomRowInds_CCUS)):
            self.randomized_costs[costRowID] = param.pathways2Net0.evaluate(
                "CCUS!O" + str(param.pathways2Net0RandomRowInds_CCUS[costRowID])
            )
        for costRowID in np.arange(len(param.pathways2Net0RandomRowInds_Outputs)):
            self.randomized_costs[
                len(param.pathways2Net0RandomRowInds_CCUS) + costRowID
            ] = param.pathways2Net0.evaluate(
                "Outputs!O" + str(param.pathways2Net0RandomRowInds_Outputs[costRowID])
            )
        self.noise_observability = param.noise_observability

        # time variables
        # NOTE: our convention is to update step_count at the beginning of the gym step() function
        self.step_count = -1
        self.steps_per_episode = param.steps_per_episode
        
        # initial jobs supported in 2030
        self.jobs = np.float32(
            110484
        )  
        # variable to record jobs created each year
        self.jobs_increment = np.zeros(1, dtype=np.float32)  # initialized as 0
        # fmt: off
        # initial economic impact in 2030
        self.econoImpact = np.float32(49938.9809739566)
        # initial technology deployments in 2030
        self.deployments = np.array([param.pathways2Net0.evaluate('GALE!P35'), 
                                     param.pathways2Net0.evaluate('GALE!X35'), 
                                     param.pathways2Net0.evaluate('GALE!Y35')], 
                                    dtype=np.float32) 
        # initial CO2 emissions in 2030
        self.emission_amount = np.float32(param.pathways2Net0.evaluate('CCUS!O63')) 
        # fmt: on

        # histories
        self.observations_all = []
        self.actions_all = []
        self.rewards_all = []
        self.weightedRewardComponents_all = []
        self.deployments_all = []
        self.emission_amount_all = []

    def to_observation(self):
        observation = (self.step_count,) + tuple(
            self.randomized_costs
        )  
        if self.noise_observability == False:
            observation = (self.step_count,)


        return observation

    def is_done(self):
        done = bool(self.step_count >= self.steps_per_episode - 1)
        return done



def record(state, action, reward, weightedRewardComponents):
    state.observations_all.append(state.to_observation())
    state.actions_all.append(action)
    state.rewards_all.append(reward)
    state.weightedRewardComponents_all.append(weightedRewardComponents)
    state.deployments_all.append(state.deployments)
    state.emission_amount_all.append(state.emission_amount)


def observation_space(self):
    obs_low = np.full_like(self.state.to_observation(), 0, dtype=np.float32)
    obs_low[0] = -1  # first entry of obervation is the timestep
    obs_high = np.full_like(self.state.to_observation(), 1e5, dtype=np.float32)
    obs_high[0] = self.param.steps_per_episode  # first entry of obervation is the timestep
    if self.state.noise_observability == True:
        obs_high[5] = 1e6  
        obs_high[7] = 1e6  
    result = spaces.Box(obs_low, obs_high, dtype=np.float32)
    return result


def action_space(self):
    # action specifies yearly increments in offshore wind power, blue hydrogen, and green hydrogen respectively
    # lower limit on increments is zero
    act_low = np.zeros(self.param.techs, dtype=np.float32)
    # upper limits on increments depend on the technology
    act_high = np.float32([27, 25, 24]) 
    result = spaces.Box(act_low, act_high, dtype=np.float32)
    return result


def apply_action(action, state, param):
    # copy model from state to param
    param.pathways2Net0 = state.pathways2Net0

    # each technology gives rewards of various types (ie costs and revenues)
    # create an array to hold the reward components (aggregated over all technologies):
    weightedRewardComponents = np.zeros(
        param.reward_types
    )  

    # read in the current deployment for offshore wind power
    offshoreWind = param.pathways2Net0.evaluate(
        # "GALE!P" + str(param.pathways2Net0RowInds[state.step_count] - 1)
        "GALE!S" + str(param.pathways2Net0RowInds[state.step_count] - 1)
    )

    # add the increment of offshore wind for this timestep (specified by the action), imposing a maximum deployment
    offshoreWind = np.clip(offshoreWind + action[0], offshoreWind, 380)
    
    # similarly for blue and green hydrogen
    blueHydrogen = param.pathways2Net0.evaluate(
        "GALE!X" + str(param.pathways2Net0RowInds[state.step_count] - 1)
    )
    blueHydrogen = np.clip(blueHydrogen + action[1], blueHydrogen, 270)
    greenHydrogen = param.pathways2Net0.evaluate(
        "GALE!Y" + str(param.pathways2Net0RowInds[state.step_count] - 1)
    )
    greenHydrogen = np.clip(greenHydrogen + action[2], greenHydrogen, 253)
    
    # record the new deployments in an array
    state.deployments = np.array(
        [offshoreWind, blueHydrogen, greenHydrogen], dtype=np.float32
    )
    
    # evaluate the model cells containing the deployment values for the current timestep (for offshore wind power, blue hydrogen and green hydrogen respectively)
    # this enables the current timestep's deployment values to be entered into the model 
    param.pathways2Net0.evaluate(
        # "GALE!P" + str(param.pathways2Net0RowInds[state.step_count])
        "GALE!S" + str(param.pathways2Net0RowInds[state.step_count])
    )
    param.pathways2Net0.evaluate(
        "GALE!X" + str(param.pathways2Net0RowInds[state.step_count])
    )
    param.pathways2Net0.evaluate(
        "GALE!Y" + str(param.pathways2Net0RowInds[state.step_count])
    )
    # similarly, evaluate the current timestep's capex, opex, revenue, and emissions values for all technologies
    # fmt: off
    capex_all = np.float32([param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'24'), 
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'28'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'32'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'36'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'41')])
    opex_all = np.float32([param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'25'), 
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'29'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'33'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'37'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'42')])
    revenue_all = np.float32([param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'26'), 
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'30'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'34'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'38'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'43')])
    # fmt: on
    emissions = np.float32(
        param.pathways2Net0.evaluate(
            "CCUS!" + param.pathways2Net0ColumnInds[state.step_count] + "68"
        )
    )
    
    # enter the deployment values for this timestep into the model
    param.pathways2Net0.set_value(
        # "GALE!P" + str(param.pathways2Net0RowInds[state.step_count]), offshoreWind
        "GALE!S" + str(param.pathways2Net0RowInds[state.step_count]), offshoreWind
    )
    param.pathways2Net0.set_value(
        "GALE!X" + str(param.pathways2Net0RowInds[state.step_count]), blueHydrogen
    )
    param.pathways2Net0.set_value(
        "GALE!Y" + str(param.pathways2Net0RowInds[state.step_count]), greenHydrogen
    )
    # re-evaluate the current timestep's capex, opex, revenue, and emissions values for all technologies
    # fmt: off
    capex_all = np.float32([param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'24'), 
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'28'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'32'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'36'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'41')])
    opex_all = np.float32([param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'25'), 
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'29'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'33'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'37'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'42')])
    revenue_all = np.float32([param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'26'), 
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'30'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'34'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'38'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'43')])
    # fmt: on
    # read gross carbon emissions (before CCUS) from model
    state.emission_amount = np.float32(
        param.pathways2Net0.evaluate(
            "CCUS!" + param.pathways2Net0ColumnInds[state.step_count] + "63"
        )
    )
    # read net carbon emissions (after CCUS) from model
    emissions = np.float32(
        param.pathways2Net0.evaluate(
            "CCUS!" + param.pathways2Net0ColumnInds[state.step_count] + "68"
        )
    )
    # calculate the total capex, opex, revenue and emissions
    weightedRewardComponents[0] = np.sum(capex_all)
    weightedRewardComponents[1] = np.sum(opex_all)
    weightedRewardComponents[2] = np.sum(revenue_all)
    weightedRewardComponents[3] = emissions
    weightedRewardComponents[5] = state.econoImpact
    # calculate numer of jobs supported as 0.25 * (capex + opex + 1050) / 0.05:
    weightedRewardComponents[4] = (
        0.25 * (weightedRewardComponents[0] + weightedRewardComponents[1] + 1050) / 0.05
    )
    state.jobs_increment = weightedRewardComponents[-2] - state.jobs
    state.jobs = weightedRewardComponents[-2]
    # calculate reward for this timestep: revenue - (capex + opex + emissions) + timestep * (increment in jobs) 
    reward = (
        weightedRewardComponents[2] - np.sum(weightedRewardComponents[[0, 1, 3]]) + (state.step_count * state.jobs_increment)
    )  
    # copy model from param to state
    state.pathways2Net0 = param.pathways2Net0
    return state, reward, weightedRewardComponents


def verify_constraints(state):
    verify = True
    return verify


def randomise(state, action, param):
    # copy model from state to param
    param.pathways2Net0 = state.pathways2Net0

    # noise will be applied by multiplication 
    
    # evaluate capex, opex, revenue, and emissions for each technology:
    # fmt: off
    np.float32([param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'24'), 
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'28'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'32'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'36'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'41')])
    np.float32([param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'25'), 
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'29'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'33'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'37'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'42')])
    np.float32([param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'26'), 
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'30'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'34'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'38'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'43')])
    np.float32(param.pathways2Net0.evaluate('CCUS!'+param.pathways2Net0ColumnInds[state.step_count]+'68'))
    # fmt: on
    rowInds_CCUS = param.pathways2Net0RandomRowInds_CCUS
    rowInds_Outputs = param.pathways2Net0RandomRowInds_Outputs
    # specify smaller standard deviation for CCUS capex & opex 
    noise_sigma_CCUS = np.full(len(rowInds_CCUS), param.noise_sigma) * np.array([param.noise_sigma_factor, param.noise_sigma_factor, 1.0])
    # generate Gaussian noise N~(1,0.1), clipped at a (positive) minimum value:
    multiplicativeNoise_CCUS = np.maximum(
        param.noise_clipping,
        np.random.randn(len(rowInds_CCUS)) * noise_sigma_CCUS + param.noise_mu,
    )
    multiplicativeNoise_Outputs = np.maximum(
        param.noise_clipping,
        np.random.randn(len(rowInds_Outputs)) * param.noise_sigma + param.noise_mu,
    )
    # for each technology:
    # for each of its costs and revenues:
    # multiply the values at the current and all future timesteps by the same (independent) random number
    year_counter = 0
    # for each year in the model:
    for yearColumnID in param.pathways2Net0ColumnInds[state.step_count :]:
        # for each of the CCUS and emissions costs and revenues:
        for costRowID in np.arange(len(rowInds_CCUS)):
            # read the current cost / revenue
            currentCost = param.pathways2Net0.evaluate(
                "CCUS!" + yearColumnID + str(rowInds_CCUS[costRowID])
            )
            # apply noise 
            param.pathways2Net0.set_value(
                "CCUS!" + yearColumnID + str(rowInds_CCUS[costRowID]),
                multiplicativeNoise_CCUS[costRowID] * currentCost,
            )
            # for closed-loop mode, record the current timestep's random costs:
            if year_counter == 0:
                state.randomized_costs[costRowID] = (
                    multiplicativeNoise_CCUS[costRowID] * currentCost
                )
        # similarly for all other costs and revenues:
        for costRowID in np.arange(len(rowInds_Outputs)):
            currentCost = param.pathways2Net0.evaluate(
                "Outputs!" + yearColumnID + str(rowInds_Outputs[costRowID])
            )
            param.pathways2Net0.set_value(
                "Outputs!" + yearColumnID + str(rowInds_Outputs[costRowID]),
                multiplicativeNoise_Outputs[costRowID] * currentCost,
            )
            if year_counter == 0:
                state.randomized_costs[len(rowInds_CCUS) + costRowID] = (
                    multiplicativeNoise_Outputs[costRowID] * currentCost
                )
        # set blue hydrogen price = blue hydrogen gas feedstock price + 20:
        param.pathways2Net0.set_value(
            "Outputs!" + yearColumnID + "158",
            param.pathways2Net0.evaluate("Outputs!" + yearColumnID + "159") + 20.0,
        )
        if year_counter == 0:
            state.randomized_costs[
                len(rowInds_CCUS) + 6
            ] = param.pathways2Net0.evaluate("Outputs!" + yearColumnID + "158")

        year_counter = year_counter + 1
    # copy model from param to state
    state.pathways2Net0 = param.pathways2Net0
    return state


def reset_param(param):
    # use param.pathways2Net0_reset (the reference model) to reset the randomised costs and revenues in param.pathways2Net0 (the working model) 
    # tabs to reset:
    spreadsheets = np.array(["GALE", "CCUS", "Outputs"])
    # columns to reset in each tab:
    columnInds_BySheets = np.array(
        [
            # np.array(["P", "X", "Y"]),
            np.array(["S", "X", "Y"]),
            param.pathways2Net0ColumnInds,
            param.pathways2Net0ColumnInds,
        ]
    )
    # rows to reset in each tab:    
    rowInds_BySheets = np.array(
        [
            param.pathways2Net0RowInds,
            param.pathways2Net0RandomRowInds_CCUS,
            param.pathways2Net0RandomRowInds_Outputs,
        ]
    )
    # for each tab to reset:
    for iSheet in np.arange(len(spreadsheets)):
        # for each column to reset:
        for iColumn in columnInds_BySheets[iSheet]:
            # for each row to reset:
            for iRow in rowInds_BySheets[iSheet]:
                # reset cell to reference value
                param.pathways2Net0.set_value(
                    spreadsheets[iSheet] + "!" + iColumn + str(iRow),
                    param.pathways2Net0_reset.evaluate(
                        spreadsheets[iSheet] + "!" + iColumn + str(iRow)
                    ),
                )
    return param


def cal_reset_diff(param):
    # a helper function to check that reset_param works correctly
    abs_diff = 0.0
    # reload the model:
    workbooks_dir = Path(__file__).resolve().parent.parent / "compiled_workbook_objects"
    # pathways2Net0_loaded = ExcelCompiler.from_file(filename=f"{workbooks_dir}/PathwaysToNetZero_Simplified_Anonymized_Compiled")
    pathways2Net0_loaded = ExcelCompiler.from_file(filename=f"{workbooks_dir}/PathwaysToNetZero_Simplified_Anonymized_Modified_Compiled")
    spreadsheets = np.array(["GALE", "CCUS", "Outputs"])
    columnInds_BySheets = np.array(
        [
            # np.array(["P", "X", "Y"]),
            np.array(["S", "X", "Y"]),
            param.pathways2Net0ColumnInds,
            param.pathways2Net0ColumnInds,
        ]
    )
    rowInds_BySheets = np.array(
        [
            param.pathways2Net0RowInds,
            param.pathways2Net0RandomRowInds_CCUS,
            param.pathways2Net0RandomRowInds_Outputs,
        ]
    )
    for iSheet in np.arange(len(spreadsheets)):
        for iColumn in columnInds_BySheets[iSheet]:
            for iRow in rowInds_BySheets[iSheet]:
                if param.pathways2Net0.evaluate(spreadsheets[iSheet] + "!" + iColumn + str(iRow)) != None and pathways2Net0_loaded.evaluate(spreadsheets[iSheet] + "!" + iColumn + str(iRow)) != None:
                    abs_diff = abs_diff + np.abs(
                        param.pathways2Net0.evaluate(spreadsheets[iSheet] + "!" + iColumn + str(iRow)) - pathways2Net0_loaded.evaluate(spreadsheets[iSheet] + "!" + iColumn + str(iRow))
                    )
                else:
                    if param.pathways2Net0.evaluate(spreadsheets[iSheet] + "!" + iColumn + str(iRow)) != None:
                        abs_diff = abs_diff + np.abs(param.pathways2Net0.evaluate(spreadsheets[iSheet] + "!" + iColumn + str(iRow)))
                    if pathways2Net0_loaded.evaluate(spreadsheets[iSheet] + "!" + iColumn + str(iRow)) != None:
                        abs_diff = abs_diff + np.abs(pathways2Net0_loaded.evaluate(spreadsheets[iSheet] + "!" + iColumn + str(iRow)))
    # abs_diff should be 0 if reset_param works correctly:
    return abs_diff


def plot_episode(state, fname):
    # a helper function to plot each timestep in the most recent episode
    fig, ax = plt.subplots(2, 2)

    # plot cumulative total rewards and deployments for the 3 technologies:
    ax1 = plt.subplot(221)
    plt.plot(np.cumsum(state.rewards_all), label='cumulative reward',color='black')
    plt.xlabel("time, avg reward: " + str(np.mean(state.rewards_all)))
    plt.ylabel("cumulative reward")
    plt.legend(loc='upper left', fontsize='xx-small')
    plt.tight_layout()

    ax2 = ax1.twinx()
    ax2.plot(np.array(state.deployments_all)[:,0],label="offshore wind")
    ax2.plot(np.array(state.deployments_all)[:,1],label="blue hydrogen")
    ax2.plot(np.array(state.deployments_all)[:,2],label="green hydrogen")
    ax2.plot(np.array(state.emission_amount_all),label="CO2 emissions amount") 
    ax2.set_ylabel("deployments and CO2 emissions")
    plt.legend(loc='lower right',fontsize='xx-small')
    plt.tight_layout()

    # plot a subset of the observations:
    plt.subplot(222)
    # first 5 elements of observations are step counts and first 4 randomized costs
    plt.plot(np.array(state.observations_all)[:,0], label="step counts", color='black')
    if state.noise_observability == True:        
        plt.plot(np.array(state.observations_all)[:,1], label="CCS Capex £/tonne")
        plt.plot(np.array(state.observations_all)[:,2], label="CCS Opex £/tonne")
        plt.plot(np.array(state.observations_all)[:,3], label="Carbon price £/tonne")
        plt.plot(np.array(state.observations_all)[:,4], label="Offshore wind Devex £/kW")
        plt.plot(np.array(state.observations_all)[:,5], label="Offshore wind Capex £/kW")
    plt.xlabel("time")
    plt.ylabel("observations")
    plt.legend(loc='lower right',fontsize='xx-small')
    plt.tight_layout()

    # plot the agent's actions:
    plt.subplot(223)
    # plt.plot(np.array(state.actions_all)[:,0],label="offshore wind capacity [GW]")
    plt.plot(np.array(state.actions_all)[:,0],label="offshore wind to power [TWh]")
    plt.plot(np.array(state.actions_all)[:,1],label="blue hydrogen energy [TWh]")
    plt.plot(np.array(state.actions_all)[:,2],label="green hydrogen energy [TWh]")    
    plt.xlabel("time")
    plt.ylabel("actions")
    plt.legend(title="increment in",loc='lower right',fontsize='xx-small')
    plt.tight_layout()

    # plot jobs and increments in jobs:
    plt.subplot(224)
    to_plot = np.vstack((np.array(state.weightedRewardComponents_all)[:,4],
                        np.hstack((np.nan,np.diff(np.array(state.weightedRewardComponents_all)[:,4]))))).T    
    plt.plot(to_plot[:,0], label="jobs")
    plt.plot(to_plot[:,1], label="increment in jobs")
    plt.xlabel("time")
    plt.ylabel("jobs and increments")
    plt.legend(loc='lower left', fontsize='xx-small')
    plt.tight_layout()

    plt.savefig(fname)


def score(state):
    value1 = np.sum(state.rewards_all)
    return {"value1": value1}


class GymEnv(gym.Env):
    def __init__(self):
        self.seed()
        self.load_workbooks()
        self.initialise_state()
    
    def load_workbooks(self):
        self.param = Parameters()
        workbooks_dir = Path(__file__).resolve().parent.parent / "compiled_workbook_objects"
        # load a working model and a reference model:      
        self.param.pathways2Net0 = ExcelCompiler.from_file(filename=f"{workbooks_dir}/PathwaysToNetZero_Simplified_Anonymized_Modified_Compiled")
        self.param.pathways2Net0_reset = ExcelCompiler.from_file(filename=f"{workbooks_dir}/PathwaysToNetZero_Simplified_Anonymized_Modified_Compiled")

    def initialise_state(self):

        self.param = reset_param(self.param)
        self.state = State(seed=self.current_seed, param=self.param)
        self.action_space = action_space(self)
        self.observation_space = observation_space(self)

    def reset(self):
        self.initialise_state()
        observation = self.state.to_observation()
        return observation
    
    def check_reset(self):
        reset_diff = cal_reset_diff(self.param)
        return reset_diff

    def step(self, action):
        self.state.step_count += 1
        self.state = randomise(self.state, action, self.param)
        self.state, reward, weightedRewardComponents = apply_action(action, self.state, self.param)
        observation = self.state.to_observation()
        done = self.state.is_done()
        record(self.state, action, reward, weightedRewardComponents)
        return observation, reward, done, {}

    def seed(self, seed=None):
        self.current_seed = seed

    def score(self):
        if self.state.is_done():
            return score(self.state)
        else:
            return None

    def plot(self, fname="episode.png"):
        plot_episode(self.state, fname)

    def render(self):
        pass

    def close(self):
        pass
