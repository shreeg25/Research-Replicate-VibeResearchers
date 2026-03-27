import matplotlib.pyplot as plt

def plot_routing_distribution():
    # Data extracted from the MoE Router
    experts = ['Expert 1', 'Expert 2', 'Expert 3', 'Expert 4']
    tokens = [4, 0, 0, 60]
    
    # Professional research paper styling
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Generate the bar chart
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']
    bars = ax.bar(experts, tokens, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add the exact token counts on top of each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1.5, int(yval), 
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Formatting
    ax.set_ylabel('Geographical Tokens Processed', fontweight='bold', fontsize=12)
    ax.set_title('MoE Domain Routing Distribution (Wayanad Zero-Shot)', fontweight='bold', fontsize=14, pad=15)
    ax.set_ylim(0, 70) # Add headroom for the text
    
    # Save the graph
    plt.tight_layout()
    plt.savefig('routing_distribution.png', dpi=300)
    print("Graph saved successfully as 'routing_distribution.png'.")
    plt.show()

if __name__ == "__main__":
    plot_routing_distribution()