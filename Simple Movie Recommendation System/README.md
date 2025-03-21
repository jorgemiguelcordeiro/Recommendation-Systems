# Simple Movie Recommendation System

A minimalist collaborative filtering recommendation system built with PyTorch that predicts user ratings for movies using basic embedding techniques.

## Overview

This project implements a straightforward recommendation system designed for learning purposes. Using a basic matrix factorization approach, this beginner-friendly system demonstrates fundamental concepts in recommendation systems without complicated architectures or extensive feature engineering.

## Features

- Minimal PyTorch implementation of collaborative filtering
- Simple user and movie embeddings
- Basic MSE loss function
- RMSE (Root Mean Square Error) evaluation
- Designed for educational purposes


## Model Architecture

The model uses a deliberately simple architecture with small embeddings and a single linear layer:

```python
class RecSysModel(nn.Module):
    def __init__(self, num_users, num_movies):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, 32)
        self.movie_embed = nn.Embedding(num_movies, 32)
        self.out = nn.Linear(64, 1)
```

## Limitations

This project is intentionally simplified and lacks:
- Advanced features like attention mechanisms
- Incorporation of content-based features
- Hyperparameter tuning
- Production-ready optimizations

## Purpose

This project was created as a learning exercise to understand the basics of:
- Embedding-based recommendation systems
- PyTorch implementation of collaborative filtering
- Training loops and evaluation metrics


