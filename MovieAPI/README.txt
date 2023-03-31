README

This application provides a simple movie filter API using Flask and SQLAlchemy. The API has the following endpoints:

    /movie_filter - GET request that returns movies based on filter criteria such as movie ID, title, year of release, running time, genre, and average rating.

    /users - GET request that returns a list of all users.

    /ratings - GET request that returns a list of all ratings.

    /movies/<movie_id>/ratings - GET request that returns all ratings for a specific movie.

    /users/<user_id>/ratings - GET request that returns all ratings given by a specific user.

The API uses SQLite as its database and performs queries using SQLAlchemy ORM. The data is returned in JSON format.

To use the API, make a GET request to the appropriate endpoint. You can use the following filter criteria for the /movie_filter endpoint:

    id: Movie ID (integer)
    title: Movie title (string)
    yearOfRelease: Year of release (integer)
    runningTime: Running time in minutes (integer)
    genre: Movie genre (string)
    averageRating: Average rating of the movie (float)

You can also combine multiple filters in a single request. If no filters are specified, all movies will be returned.

Make sure to include the appropriate endpoint and filter criteria in your GET request.


Here are some example CURL commands for each endpoint in the app:

    Get movies filtered by title, year of release, running time, and genre:
curl http://localhost:8000/movie_filter/Toy%20Story/1995/81/Animation

    Get all users:
curl http://localhost:8000/users

    Get all ratings:
curl http://localhost:8000/ratings

    Get ratings for a specific movie (where <movie_id> is replaced with the ID of the movie):
curl http://localhost:8000/movies/<movie_id>/ratings

    Get ratings for a specific user (where <user_id> is replaced with the ID of the user):
curl http://localhost:8000/users/<user_id>/ratings

    Update ratings for a specific movie from a specific user]
curl "http://127.0.0.1:8000/update_rating?user_id=1&movie_id=1&new_rating=2"

