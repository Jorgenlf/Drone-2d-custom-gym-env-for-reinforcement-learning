import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data for scenario_metrics table
scenario_data = {
    'Scenario': ['Corridor', 'S-corridor', 'Parallel', 'S-parallel', 'Perpendicular', 'Large', 'Impossible'],
    'Reactive_AAPE': [104, 104, 111, 87, 119, 90, 87],
    'Reactive_SR': [88, 71, 55, 3, 28, 71, 0],
    'Reactive_FR': [12, 29, 45, 97, 72, 29, 100],
    'Reactive_CR': [12, 29, 45, 97, 71, 29, 85],
    'Static_AAPE': [113, 115, 112, 84, 128, 44, 59],
    'Static_SR': [21, 0, 9, 2, 21, 93, 0],
    'Static_FR': [79, 100, 91, 98, 79, 7, 100],
    'Static_CR': [48, 45, 91, 96, 79, 7, 100]
}

# Data for stagesmetrics table
stages_data = {
    'Stage': ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Stage 5'],
    'Reactive_AAPE': [7, 119, 18, 35, 35],
    'Reactive_SR': [100, 96, 94, 48, 49],
    'Reactive_FR': [0, 4, 6, 52, 51],
    'Reactive_CR': [0, 0, 5, 49, 48],
    'Static_AAPE': [4, 115, 14, 19, 8],
    'Static_SR': [92, 79, 69, 11, 15],
    'Static_FR': [8, 21, 31, 89, 85],
    'Static_CR': [0, 0, 7, 62, 79]
}

#Mellow green color for reactive agent and light purple for static agent
colors = ['#77dd77', '#b19cd9']


#increase/tune text size
plt.rcParams.update({'font.size': 20})

#Plot AAPE for each scenario and stage for reactive and static in a barplot
def plot_aape_scenarios_stages():
    # Create dataframe from scenario_data
    df = pd.DataFrame(scenario_data, columns=['Scenario', 'Reactive_AAPE', 'Static_AAPE'])

    # Set the index to be "Scenario" so they will be used as labels
    df.set_index('Scenario', inplace=True)
    
    # Plot a bar chart
    df.plot(kind='bar', figsize=(10, 6),color =colors)

    # Set the title and labels
    plt.title('Average of Average Path Error (AAPE) for different scenarios')
    plt.ylabel('AAPE [cm]',fontweight='bold')
    plt.xlabel('Scenario',fontweight='bold')
    plt.legend(['Reactive', 'Static'])
    plt.xticks(rotation=0)


    # Show the plot
    plt.show()

    # Create dataframe from stages_data
    df = pd.DataFrame(stages_data, columns=['Stage', 'Reactive_AAPE', 'Static_AAPE'])

    # Set the index to be "Stage" so they will be used as labels
    df.set_index('Stage', inplace=True)
    
    # Plot a bar chart
    df.plot(kind='bar', figsize=(10, 6),color =colors)

    # Set the title and labels
    plt.title('Average of Average Path Error (AAPE) for different stages')
    plt.ylabel('AAPE [cm]',fontweight='bold')
    plt.xlabel('Stage',fontweight='bold')
    plt.legend(['Reactive', 'Static'])
    plt.xticks(rotation=0)


    # Show the plot
    plt.show()


#Plot SR for each scenario and stage for reactive and static in a barplot
def plot_sr_scenarios_stages():
    # Create dataframe from scenario_data
    df = pd.DataFrame(scenario_data, columns=['Scenario', 'Reactive_SR', 'Static_SR'])

    # Set the index to be "Scenario" so they will be used as labels
    df.set_index('Scenario', inplace=True)

    # Plot a bar chart
    df.plot(kind='bar', figsize=(10, 6),color = colors)

    # Set the title and labels
    plt.title('Success Rate (SR) for different scenarios')
    plt.ylabel('Percentage',fontweight='bold')
    plt.xlabel('Scenario',fontweight='bold')
    plt.legend(['Reactive', 'Static'])
    plt.xticks(rotation=0)

    # Show the plot
    plt.show()

    # Create dataframe from stages_data
    df = pd.DataFrame(stages_data, columns=['Stage', 'Reactive_SR', 'Static_SR'])

    # Set the index to be "Stage" so they will be used as labels
    df.set_index('Stage', inplace=True)

    # Plot a bar chart
    df.plot(kind='bar', figsize=(10, 6),color = colors)

    # Set the title and labels
    plt.title('Success Rate (SR) for different stages')
    plt.ylabel('Percentage',fontweight='bold')
    plt.xlabel('Stage',fontweight='bold')
    plt.legend(['Reactive', 'Static'])
    plt.xticks(rotation=0)

    # Show the plot
    plt.show()

#Plot CR for each scenario and stage for reactive and static in a barplot
def plot_cr_scenarios_stages():
    # Create dataframe from scenario_data
    df = pd.DataFrame(scenario_data, columns=['Scenario', 'Reactive_CR', 'Static_CR'])

    # Set the index to be "Scenario" so they will be used as labels
    df.set_index('Scenario', inplace=True)

    # Plot a bar chart
    df.plot(kind='bar', figsize=(10, 6),color = colors)

    # Set the title and labels
    plt.title('Collision Rate (CR) for different scenarios')
    plt.ylabel('Percentage',fontweight='bold')
    plt.xlabel('Scenario',fontweight='bold')
    plt.legend(['Reactive', 'Static'])
    plt.xticks(rotation=0)

    # Show the plot
    plt.show()

    # Create dataframe from stages_data
    df = pd.DataFrame(stages_data, columns=['Stage', 'Reactive_CR', 'Static_CR'])

    # Set the index to be "Stage" so they will be used as labels
    df.set_index('Stage', inplace=True)

    # Plot a bar chart
    df.plot(kind='bar', figsize=(10, 6),color = colors)

    # Set the title and labels
    plt.title('Collision Rate (CR) for different stages')
    plt.ylabel('Percentage',fontweight='bold')
    plt.xlabel('Stage',fontweight='bold')
    plt.legend(['Reactive', 'Static'])
    plt.xticks(rotation=0)

    # Show the plot
    plt.show()

#Plot FR for each scenario and stage for reactive and static in a barplot
def plot_fr_scenarios_stages():
    # Create dataframe from scenario_data
    df = pd.DataFrame(scenario_data, columns=['Scenario', 'Reactive_FR', 'Static_FR'])

    # Set the index to be "Scenario" so they will be used as labels
    df.set_index('Scenario', inplace=True)

    # Plot a bar chart
    df.plot(kind='bar', figsize=(10, 6),color = colors)

    # Set the title and labels
    plt.title('Failure Rate (FR) for different scenarios')
    plt.ylabel('Percentage',fontweight='bold')
    plt.xlabel('Scenario',fontweight='bold')
    plt.legend(['Reactive', 'Static'])
    plt.xticks(rotation=0)

    # Show the plot
    plt.show()

    # Create dataframe from stages_data
    df = pd.DataFrame(stages_data, columns=['Stage', 'Reactive_FR', 'Static_FR'])

    # Set the index to be "Stage" so they will be used as labels
    df.set_index('Stage', inplace=True)

    # Plot a bar chart
    df.plot(kind='bar', figsize=(10, 6),color = colors)

    # Set the title and labels
    plt.title('Failure Rate (FR) for different stages')
    plt.ylabel('Percentage',fontweight='bold')
    plt.xlabel('Stage',fontweight='bold')
    plt.legend(['Reactive', 'Static'])
    plt.xticks(rotation=0)

    # Show the plot
    plt.show()


plot_aape_scenarios_stages()
plot_cr_scenarios_stages()
plot_fr_scenarios_stages()
plot_sr_scenarios_stages()





#OLD
colors2 = ['#77dd77', '#fdbf6f', '#4daf4a', '#b19cd9', '#d8a6e6', '#e0b5f2']
#Plot SR,CR,FR for each scenario and stage for reactive and static in a barplot
def plot_sr_cr_fr_scenarios_stages():
    # Create dataframe from scenario_data
    df = pd.DataFrame(scenario_data, columns=['Scenario', 'Reactive_SR', 'Reactive_CR', 'Reactive_FR', 'Static_SR', 'Static_CR', 'Static_FR'])

    # Set the index to be "Scenario" so they will be used as labels
    df.set_index('Scenario', inplace=True)

    # Plot a bar chart
    df.plot(kind='bar', figsize=(10, 6),color = colors2)
    #set colors for reactive and static agent    
    # Set the title and labels
    plt.title('Success Rate (SR), Failure Rate (FR) and Collision Rate (CR) for different scenarios')
    plt.ylabel('Percentage')
    plt.xlabel('Scenario',fontweight='bold')
    plt.legend(['Reactive SR', 'Reactive CR', 'Reactive FR', 'Static SR', 'Static CR', 'Static FR'])
    plt.xticks(rotation=0)


    # Show the plot
    plt.show()

    # Create dataframe from stages_data
    df = pd.DataFrame(stages_data, columns=['Stage', 'Reactive_SR', 'Reactive_CR', 'Reactive_FR', 'Static_SR', 'Static_CR', 'Static_FR'])

    # Set the index to be "Stage" so they will be used as labels
    df.set_index('Stage', inplace=True)

    # Plot a bar chart
    df.plot(kind='bar', figsize=(10, 6),color = colors2)

    # Set the title and labels
    plt.title('Success Rate (SR), Failure Rate (FR) and Collision Rate (CR) 100 for different stages')
    plt.ylabel('Percentage')
    plt.xlabel('Stage',fontweight='bold')
    plt.legend(['Reactive SR', 'Reactive CR', 'Reactive FR', 'Static SR', 'Static CR', 'Static FR'])
    plt.xticks(rotation=0)

    # Show the plot
    plt.show()    
# plot_sr_cr_fr_scenarios_stages()