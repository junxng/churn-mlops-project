import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from src.Churn.config.configuration import ConfigurationManager

def visualize_customer_churn(df: pd.DataFrame) -> str:
    """
    Generates a pie chart visualizing the customer churn distribution.

    Args:
        df (pd.DataFrame): DataFrame with a 'Churn_RATE' column.

    Returns:
        str: The path to the saved pie chart image.
    """
    config_manager = ConfigurationManager()
    viz_config = config_manager.get_visualization_config()
    
    output_dir = viz_config.output_dir
    pie_config = viz_config.pie_chart

    os.makedirs(output_dir, exist_ok=True)
    plt.switch_backend('Agg')

    try:
        churn_counts = df['Churn_RATE'].value_counts().sort_index()
        sizes = [churn_counts.get(0, 0), churn_counts.get(1, 0)]
        
        fig_pie, ax = plt.subplots(figsize=(8, 6))
        ax.pie(
            sizes,
            labels=pie_config.labels,
            autopct='%1.1f%%',
            startangle=pie_config.start_angle,
            colors=pie_config.colors,
            wedgeprops={'edgecolor': pie_config.edge_color}
        )
        ax.set_title(pie_config.title)
        ax.axis('equal')

        timestamp = int(time.time())
        pie_plot_path = os.path.join(output_dir, f"Churn_RATE_piechart_{timestamp}.png")
        plt.savefig(pie_plot_path)
        plt.close(fig_pie)

        return pie_plot_path

    except Exception as e:
        print(f"Error generating visualization: {e}")
        # Return a path to a placeholder or error image if needed
        return os.path.join(output_dir, "visualization_failed.png")
