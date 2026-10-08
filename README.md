# Recipe Clustering with Unsupervised Learning

Clustering 10,000 recipes by ingredient text and nutrition data: KMeans with model selection, hierarchical clustering, and DBSCAN outlier detection.

Coursework from **Applied Unsupervised Learning in Python** (University of Michigan, More Applied Data Science with Python specialization). All notebooks run end to end with outputs.

## Data

`data/` holds the course provided recipe dataset (12 MB):

- `RAW_recipes_processed.csv` — 10,000 recipes with ingredient text plus nutritional columns (calories, total fat, sugar, sodium, protein, saturated fat, carbs as percent daily values)

## What was done

**Part 1 — KMeans on ingredient text** (`notebooks/01-kmeans-tfidf.ipynb`)
Vectorized ingredient text with TFIDF (1 to 2 grams, 1,000 features), then ran KMeans for K 2 through 9 and picked the best K with the Calinski Harabasz score. K = 2 won clearly with a score of 245.46. The two clusters rediscover a sweet versus savory split: cluster 0 leans sweet (sugar, flour, baking, butter, cinnamon, vanilla, milk, eggs, apples, water) while cluster 1 leans savory (pepper, cheese, garlic, fresh, oil, salt, onion, sauce, ground, chicken). Charts: `charts/chart_ch_scores.png`, `charts/chart_top_terms.png`, `charts/chart_pca_kmeans.png`.

**Part 2 — Hierarchical clustering** (`notebooks/02-hierarchical-clustering.ipynb`)
Complete linkage agglomerative clustering on 10 sample recipes (TFIDF reduced to 2 PCA dimensions). Three recipe pairs merge below distance 0.2: the two dessert recipes (apple a day milk shake with bananas 4 ice cream pie, distance 0.10), the squash dish with marinated olives (0.11), and the breakfast pizza with amish tomato ketchup (0.13). Chart: `charts/chart_dendrogram.png`.

**Part 3 — DBSCAN outlier detection** (`notebooks/03-dbscan-outliers.ipynb`)
DBSCAN on the first 100 recipes, once on TFIDF text (2D PCA, eps 0.145 picked from the K distance elbow, min samples 5) and once on standardized nutritional columns (2D PCA, eps 1.0, min samples 5). The text view flags 4 outliers (a gelatin dessert, two cheesecakes, a taco dip); the nutrition view flags 11 outliers (rich desserts, fried food, heavy sauces). Two recipes are outliers in both views: bananas 4 ice cream pie and the best chocolate chip cheesecake ever. Charts: `charts/chart_kdistance_elbow.png`, `charts/chart_dbscan_text.png`, `charts/chart_dbscan_nutrition.png`.

## Tech

Python, pandas, NumPy, scikit learn (KMeans, DBSCAN, PCA, TFIDF), SciPy (hierarchical linkage), matplotlib, Jupyter.
