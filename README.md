# Hybrid Recommendation System

This project is a hybrid recommendation system built with Python, Fast API, scikit-learn, pandas, numpy, and more. It uses a combination of collaborative filtering, content-based filtering, and popularity-based filtering to generate recommendations.

## Features

* Hybrid recommendation system with adjustable weights for each component
* Supports cold start recommendation for new users
* Can be configured with environment.yml (conda), requirements.txt (plain python), or Dockerfile (docker)

## Usage

### With Conda

1. Install conda if you haven't already
2. Run `conda env create -f environment.yml` to create a new environment
3. Activate the environment with `conda activate recomender`
4. Run `python main.py` to start the API

### With plain Python

1. Install the requirements with `pip install -r requirements.txt`
2. Run `python main.py` to start the API

### With Docker

1. Build the docker image with `docker build -t recomender .`
2. Run the container with `docker run -p 8000:8000 recomender`
3. The API will be available at `http://localhost:8000`

## API Endpoints

### GET /users/{user_id}/recommendations

Returns a list of recommended products for the given user

### GET /products/{product_id}/similar

Returns a list of similar products to the given product

### GET /products/popular

Returns a list of popular products

### GET /products/new

Returns a list of new products

### POST /users/{user_id}/interactions

Creates a new interaction for the given user and product

## Configuration

The following environment variables can be used to configure the API:

* `RECOMENDER_CF_WEIGHT`: The weight of the collaborative filtering component (default: 0.5)
* `RECOMENDER_CB_WEIGHT`: The weight of the content-based filtering component (default: 0.3)
* `RECOMENDER_PB_WEIGHT`: The weight of the popularity-based filtering component (default: 0.2)
* `RECOMENDER_COLD_START_THRESHOLD`: The minimum number of interactions required for a user to be considered "warm" (default: 10)
* `RECOMENDER_COLD_START_RECOMMENDATIONS`: The number of recommendations to return for cold start users (default: 10)

## Development

To develop the API, run `python main.py` and use a tool like `curl` or a web browser to test the endpoints.
