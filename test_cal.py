import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_line_from_list_of_lists(data):
    # Convert the list of lists into a DataFrame
    df = pd.DataFrame(data)
    
    # Plotting
    sns.set(style="whitegrid")  # Set the style of the plot
    plt.figure(figsize=(10, 6))  # Set the figure size
    
    # Plot each line using Seaborn
    sns.lineplot(data=df, dashes=False, ci="sd")
        
    # Set labels and title
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('Line Plot from List of Lists')
    
    # Save the plot
    plt.savefig("walid.png")
    
    # Show plot
    plt.show()

# Example usage:
data = [
    [10, 20, 30],
    [15, 25, 35],
    [12, 22, 32],
    [18, 28, 38]
]

plot_line_from_list_of_lists(data)
