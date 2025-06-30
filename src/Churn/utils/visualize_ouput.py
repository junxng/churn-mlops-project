import os
import time
import matplotlib.pyplot as plt
import pandas as pd

def visualize_customer_churn(df: pd.DataFrame, output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.switch_backend('Agg')  # Non-GUI backend for server use

    try:
        churn_counts = df['Churn_RATE'].value_counts().sort_index()
        labels = ['Not Churned', 'Churned']
        sizes = [churn_counts.get(0, 0), churn_counts.get(1, 0)]
        colors = ['#2ecc71', '#e74c3c']  
        fig_pie = plt.figure(figsize=(8, 6))
        plt.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': 'black'}
        )
        plt.title('Customer Churn Distribution')

        timestamp = int(time.time())
        pie_plot_path = os.path.join(output_dir, f"Churn_RATE_piechart_{timestamp}.png")
        plt.savefig(pie_plot_path)
        plt.close(fig_pie)

        return pie_plot_path

    except Exception as e:
        print(f"Error generating visualization: {e}")
        return os.path.join(output_dir, "visualization_failed.png")
