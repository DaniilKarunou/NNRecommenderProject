# Neural Network Recommender

This project is a neural network-based recommender system that utilizes various models to provide personalized recommendations to users. It uses user-item interaction data to learn patterns and preferences and generate recommendations based on the learned representations.

## Installation

To run the code and reproduce the results, make sure you have the following packages installed:

- Python (version X.X)
- PyTorch (version X.X)
- Pandas (version X.X)
- NumPy (version X.X)
- scikit-learn (version X.X)
- hyperopt (version X.X)
- livelossplot (version X.X)

You can install the required packages by running the following command:
```
pip install -r requirements.txt
```

## Models Used

The recommender system utilizes the following models:

- NeuMFModel: Neural Collaborative Filtering model that combines matrix factorization and multi-layer perceptron (MLP) components.
- GMFModel: Generalized Matrix Factorization model that learns user and item embeddings to make recommendations. (2 variations)
- NNModelv1: Multi-Layer Perceptron model that learns user and item embeddings to make recommendations.
- NNModelv2: Multi-Layer Perceptron model that learns user and item embeddings to make recommendations with addictional 
  dropout layer to prevent overfitting.

## Results

The performance of the recommender system was evaluated using the following evaluation metrics: HR@k (Hit Rate at k) and NDCG@k (Normalized Discounted Cumulative Gain at k).

The achieved results on the best of recommender (NNModelv2) are as follows:

| Recommender    | HR@1    | HR@3    | HR@5    | HR@10   | NDCG@1   | NDCG@3   | NDCG@5   | NDCG@10  |
|----------------|---------|---------|---------|---------|----------|----------|----------|----------|
| NNRecommender  | 0.000677| 0.008125| 0.010494| 0.024035| 0.000677 | 0.005332 | 0.006337 | 0.010473 |


