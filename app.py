from flask import Flask, render_template, request, send_file, session, redirect, url_for
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # use non-interactive backend for Flask
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import io
import base64
import os
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # change in production


def create_plot():
    """Convert current matplotlib figure to base64 PNG for embedding in HTML"""
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=200, bbox_inches='tight')
    img.seek(0)
    data = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return data


# -------------------- ROUTES --------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file and file.filename.endswith('.csv'):
        upload_folder = os.path.join('static', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        filepath = os.path.join(upload_folder, file.filename)
        file.save(filepath)
        # quick validation
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
        except Exception as e:
            print("Error reading CSV:", e)
            return redirect(url_for('index'))
        if df.empty:
            print("Uploaded file empty")
            return redirect(url_for('index'))
        session['filepath'] = filepath
        session['filename'] = file.filename
        return redirect(url_for('analysis'))
    return redirect(url_for('index'))


@app.route('/analysis')
def analysis():
    if 'filepath' not in session:
        return redirect(url_for('index'))
    df = pd.read_csv(session['filepath'], encoding='utf-8-sig')
    plots = {}

    # Distributions
    plt.figure(figsize=(15, 4))
    cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    for i, c in enumerate(cols):
        plt.subplot(1, 3, i + 1)
        sns.histplot(df[c], kde=True)
        plt.title(c)
    plt.tight_layout()
    plots['distributions'] = create_plot()

    # Gender pie + income boxplot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    if 'Gender' in df.columns:
        counts = df['Gender'].value_counts()
        plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
    else:
        plt.text(0.5, 0.5, "No Gender Column", ha='center')
    plt.title("Gender Distribution")
    plt.subplot(1, 2, 2)
    if 'Gender' in df.columns:
        sns.boxplot(data=df, x='Gender', y='Annual Income (k$)')
        plt.title("Income by Gender")
    else:
        plt.text(0.5, 0.5, "No Gender Column", ha='center')
    plt.tight_layout()
    plots['gender_analysis'] = create_plot()

    # Correlation matrix
    plt.figure(figsize=(7, 6))
    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        sns.heatmap(numeric.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title("Correlation Matrix")
    else:
        plt.text(0.5, 0.5, "No numeric columns", ha='center')
    plots['correlation'] = create_plot()

    return render_template('analysis.html',
                           plots=plots,
                           filename=session.get('filename', 'dataset.csv'),
                           data_preview=df.head(10).to_html(classes='table table-striped'),
                           data_shape=df.shape,
                           data_description=df.describe().to_html(classes='table table-striped'))


@app.route('/cluster', methods=['POST'])
def perform_clustering():
    if 'filepath' not in session:
        return redirect(url_for('index'))

    # load dataset
    df = pd.read_csv(session['filepath'], encoding='utf-8-sig')

    # form inputs
    try:
        n_clusters = int(request.form.get('n_clusters', 5))
    except:
        n_clusters = 5
    clustering_type = request.form.get('clustering_type', 'bivariate')

    plots = {}
    results = {}

    # choose features
    if clustering_type == 'univariate_income':
        X = df[['Annual Income (k$)']].values
    elif clustering_type == 'multivariate':
        # include Age, Income, Score, and numeric gender encoding
        df_tmp = df.copy()
        if 'Gender' in df_tmp.columns:
            df_tmp = pd.get_dummies(df_tmp, columns=['Gender'], drop_first=True)
        numeric = df_tmp.select_dtypes(include=[np.number])
        # ensure we pick relevant numeric columns only
        use_cols = [c for c in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'] if c in numeric.columns]
        X = numeric[use_cols].values
        X = StandardScaler().fit_transform(X)
    else:
        # bivariate default: Income vs Spending Score
        X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

    # run kmeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    # silhouette (only if >1 cluster and less than n samples)
    try:
        if len(set(df['Cluster'])) > 1 and len(df) > len(set(df['Cluster'])):
            sil = silhouette_score(X, df['Cluster'])
            results['silhouette_score'] = float(sil)
        else:
            results['silhouette_score'] = None
    except Exception:
        results['silhouette_score'] = None

    # Plot clustering visualization (income vs spending) if columns exist
    plt.figure(figsize=(12, 5))
    if 'Annual Income (k$)' in df.columns and 'Spending Score (1-100)' in df.columns:
        scatter = plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
                              c=df['Cluster'], cmap='tab10', alpha=0.7, s=50)
        # plot centroids (project centroids to the same space if bivariate)
        if clustering_type == 'bivariate':
            plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                        marker='X', s=150, c='black', label='Centroids')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.title('Income vs Spending (Clustered)')
        plt.colorbar(scatter)
        plt.grid(alpha=0.3)
    else:
        plt.text(0.5, 0.5, "Income or Spending Score missing", ha='center')
    plots['clustering_result'] = create_plot()

    # cluster sizes & summary
    results['cluster_sizes'] = df['Cluster'].value_counts().sort_index()
    summary = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].agg(['mean', 'min', 'max']).round(2)
    results['cluster_summary'] = summary

    # -----------------------
    # PERSONA GENERATION (using real data)
    # -----------------------
    personas = []
    for c in sorted(df['Cluster'].unique()):
        subset = df[df['Cluster'] == c]
        avg_age = round(float(subset['Age'].mean()), 2)
        avg_income = round(float(subset['Annual Income (k$)'].mean()), 2)
        avg_score = round(float(subset['Spending Score (1-100)'].mean()), 2)
        size = len(subset)
        # gender majority
        top_gender = None
        if 'Gender' in subset.columns:
            top_gender = subset['Gender'].value_counts().idxmax()
        # top purchased items
        top_items = []
        if 'Purchased Item' in subset.columns:
            top_items = subset['Purchased Item'].value_counts().head(3).index.tolist()

        # persona label logic (rules + top items)
        if avg_income > 80 and avg_score > 65:
            label = "High-income Frequent Buyers"
            desc = "Affluent customers with high engagement; prefer premium products."
        elif avg_income > 80 and avg_score <= 65:
            label = "High-income Low Engagement"
            desc = "High purchasing power but low recent engagement — need personalized nudges."
        elif avg_score > 70 and avg_income < 60:
            label = "Young Impulsive / Trend Seekers"
            desc = "High spenders relative to income; respond well to trends and flash sales."
        elif avg_income < 40 and avg_score < 40:
            label = "Budget-conscious Shoppers"
            desc = "Price-sensitive; react to discounts, bundles, and offers."
        else:
            label = "Mainstream Shoppers"
            desc = "Moderate income and spending; good targets for regular promotions."

        personas.append({
            'cluster': int(c),
            'size': int(size),
            'avg_age': avg_age,
            'avg_income': avg_income,
            'avg_score': avg_score,
            'top_gender': top_gender,
            'top_items': top_items,
            'label': label,
            'description': desc
        })

    results['personas'] = personas

    # -----------------------
    # RECOMMENDATIONS (simple rules using top_items and persona)
    # -----------------------
    recommendations = {}
    for p in personas:
        recs = []
        # If top items exist, recommend similar + cross-sell
        if p['top_items']:
            for it in p['top_items']:
                if it.lower() in ['clothes', 'shoes', 'handbag']:
                    recs.append('Fashion Combos & Accessory Bundles')
                elif it.lower() in ['cosmetics', 'jewelry', 'watches']:
                    recs.append('Premium Accessories & Gift Packs')
                else:
                    recs.append(f'Popular: {it}')
            # ensure uniqueness
            recs = list(dict.fromkeys(recs))
        # persona-specific suggestions
        if 'High-income' in p['label']:
            recs = recs + ['Loyalty Membership', 'Personalized Concierge Offers']
        elif 'Budget' in p['label']:
            recs = recs + ['Discount Coupons', 'Value Bundles']
        elif 'Trend' in p['label'] or 'Young' in p['label']:
            recs = recs + ['Flash Sales', 'Limited-time Drops', 'Influencer-collab Products']
        else:
            recs = recs + ['Seasonal Promotions', 'Combo Offers']
        recommendations[p['cluster']] = recs
    results['recommendations'] = recommendations

    # -----------------------
    # REAL PURCHASE HEATMAP (cluster vs Purchased Item)
    # -----------------------
    if 'Purchased Item' in df.columns:
        pivot = pd.crosstab(df['Cluster'], df['Purchased Item'])
        # normalize by cluster size to show preference proportion (optional)
        pivot_norm = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
        plt.figure(figsize=(10, max(3, 0.5 * pivot_norm.shape[0])))
        sns.heatmap(pivot_norm, annot=True, fmt='.2f', cmap='YlGnBu')
        plt.title('Cluster vs Purchased Item (proportion per cluster)')
        plt.ylabel('Cluster')
        plt.xlabel('Purchased Item')
        plots['purchase_heatmap'] = create_plot()

        # bar chart of most popular items per cluster
        plt.figure(figsize=(10, 5))
        pivot.plot(kind='bar', stacked=False, legend=True)
        plt.title('Raw Counts: Items purchased per cluster')
        plt.tight_layout()
        plots['items_bar'] = create_plot()
    else:
        plots['purchase_heatmap'] = None
        plots['items_bar'] = None

    # Save clustered CSV back (so download has Cluster column)
    df.to_csv(session['filepath'], index=False)

    return render_template('results.html',
                           plots=plots,
                           results=results,
                           n_clusters=n_clusters,
                           clustering_type=clustering_type,
                           filename=session.get('filename', 'dataset.csv'))


@app.route('/download')
def download_results():
    if 'filepath' not in session:
        return "No clustered data available"
    return send_file(session['filepath'],
                     mimetype='text/csv',
                     as_attachment=True,
                     download_name='customer_segmentation_results.csv')


@app.route('/reset')
def reset():
    if 'filepath' in session:
        try:
            os.remove(session['filepath'])
        except Exception:
            pass
    session.clear()
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
