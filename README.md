# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Preprocess → Encode gender, select features (Age, Income, Spending Score), and standardize using StandardScaler.
2. Find Optimal K → Run KMeans for K=2 to 10, plot Elbow & Silhouette Score, pick best K=5.
3. Cluster → Fit KMeans with K=5, assign cluster labels, print Silhouette Score & Cluster Summary.
4. Visualize & Export → Plot 2D, 3D scatter & bar charts, save as .png, export CSV.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: SUDESAN T
RegisterNumber: 212225240161 
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from google.colab import files

df = pd.read_csv(next(iter(files.upload())))
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

X = StandardScaler().fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

K = range(2, 11)
inertia, sil = zip(*[(KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_,
                      silhouette_score(X, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X))) for k in K])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, y, title, color in zip(axes, [inertia, sil], ['Elbow Method', 'Silhouette Score vs K'], ['b', 'r']):
    ax.plot(K, y, f'{color}o-', linewidth=2, markersize=8)
    ax.set(xlabel='Number of Clusters (K)', title=title); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('elbow_silhouette.png', dpi=150); plt.show()

optimal_k = 5
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
df['Cluster'] = KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit_predict(X)

print(f"Silhouette Score (K={optimal_k}): {silhouette_score(X, df['Cluster']):.4f}")
summary = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().round(2)
summary['Count'] = df['Cluster'].value_counts().sort_index()
print("\nCluster Summary:\n", summary)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (x, y) in zip(axes, [('Annual Income (k$)', 'Spending Score (1-100)'), ('Age', 'Spending Score (1-100)'), ('Age', 'Annual Income (k$)')]):
    [ax.scatter(df[df['Cluster']==i][x], df[df['Cluster']==i][y], c=colors[i], label=f'Cluster {i}', s=60, alpha=0.8, edgecolors='white') for i in range(optimal_k)]
    ax.set(xlabel=x, ylabel=y, title=f'{x} vs {y}'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.suptitle('K-Means Customer Segmentation', fontsize=15, fontweight='bold'); plt.tight_layout()
plt.savefig('cluster_plots.png', dpi=150); plt.show()

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
[ax.scatter(df[df['Cluster']==i]['Age'], df[df['Cluster']==i]['Annual Income (k$)'], df[df['Cluster']==i]['Spending Score (1-100)'],
            c=colors[i], label=f'Cluster {i}', s=60, alpha=0.8) for i in range(optimal_k)]
ax.set(xlabel='Age', ylabel='Annual Income (k$)', zlabel='Spending Score', title='3D Customer Segmentation'); ax.legend()
plt.tight_layout(); plt.savefig('3d_clusters.png', dpi=150); plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, feat in zip(axes, ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']):
    vals = [df[df['Cluster']==i][feat].mean() for i in range(optimal_k)]
    bars = ax.bar([f'Cluster {i}' for i in range(optimal_k)], vals, color=colors, edgecolor='white')
    [ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{v:.1f}', ha='center', fontsize=9) for b, v in zip(bars, vals)]
    ax.set(title=f'Avg {feat} per Cluster', ylabel=feat); ax.tick_params(axis='x', rotation=30); ax.grid(axis='y', alpha=0.3)
plt.suptitle('Feature Distribution Across Clusters', fontsize=13, fontweight='bold'); plt.tight_layout()
plt.savefig('feature_distribution.png', dpi=150); plt.show()

df.to_csv('customers_segmented.csv', index=False)
print("\nSegmented data saved to 'customers_segmented.csv'")

```

## Output:

<img width="901" height="451" alt="{FDE62262-1198-4D4B-AC89-2F3E9A3E184B}" src="https://github.com/user-attachments/assets/bd4e8dbd-4b26-4eca-8648-972068661d93" />

<img width="1000" height="305" alt="{F5967984-71D3-459C-9827-508AFDADFDEE}" src="https://github.com/user-attachments/assets/3bfb67d9-5445-45be-916d-d040fe48339d" />

<img width="539" height="425" alt="{10EF911E-7A42-4E05-846A-661D11291AE4}" src="https://github.com/user-attachments/assets/f90405c6-d988-4e73-920b-9fd22c560db9" />

<img width="960" height="285" alt="{D6C7D1A0-0845-4BDB-B5A5-542113AE0D39}" src="https://github.com/user-attachments/assets/6995a0a1-35ff-4a8a-bb8e-430bf6ac6507" />

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
