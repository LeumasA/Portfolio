from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from flask import Flask, jsonify, request
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func, Numeric, desc
Base = declarative_base()

class Rating(Base):
    __tablename__ = 'ratings'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    movie_id = Column(Integer, ForeignKey('movies.id'))
    rating = Column(Integer)
    user = relationship('User', back_populates='ratings')
    movie = relationship('Movie', back_populates='ratings')

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    ratings = relationship('Rating', back_populates='user')

class Movie(Base):
    __tablename__ = 'movies'

    id = Column(Integer, primary_key=True)
    title = Column(String)
    year_of_release = Column(Integer)
    running_time = Column(Integer)
    genre = Column(String)
    ratings = relationship('Rating', back_populates='movie')

    @property
    def avg_rating(self):
        if self.ratings:
            return sum([rating.rating for rating in self.ratings])/len(self.ratings)
        else:
            return None


app = Flask(__name__)



# define API endpoints
#@app.route('/movie_filter', methods=['GET'])
#@app.route('/movie_filter/<int:id>/<string:title>/')
@app.route('/movie_filter/<string:title>/<int:yearOfRelease>/<int:runningTime>/<string:genre>')
def get_movies(id=None, title=None, yearOfRelease=None, runningTime=None, genre=None, averageRating=None):
    engine = create_engine('sqlite:///movies.db')
    Session = sessionmaker(bind=engine)
    session = Session()

    query = session.query(
        Movie.id,
        Movie.title,
        Movie.year_of_release.label('yearOfRelease'),
        Movie.running_time.label('runningTime'),
        Movie.genre,
        func.round(func.avg(Rating.rating).cast(Numeric(precision=2, scale=1)), 1).label('averageRating')
    ).join(Rating).group_by(Movie.id)

    if id is not None:
        query = query.filter(Movie.id == id)

    if title is not None:
        query = query.filter(Movie.title == title)

    if yearOfRelease is not None:
        query = query.filter(Movie.year_of_release == yearOfRelease)

    if runningTime is not None:
        query = query.filter(Movie.running_time == runningTime)

    if genre is not None:
        query = query.filter(Movie.genre == genre)

    if averageRating is not None:
        query = query.having(func.round(func.avg(Rating.rating).cast(Numeric(precision=2, scale=1)), 1) == averageRating)

    movies = query.all()

    movies_list = []
    for movie in movies:
        movie_dict = {
            'id': movie.id,
            'title': movie.title,
            'yearOfRelease': movie.yearOfRelease,
            'runningTime': movie.runningTime,
            'genre': movie.genre,
            'averageRating': movie.averageRating
        }
        movies_list.append(movie_dict)

    returncode=200
    if not movies_list:
        returncode = 404

    if not isinstance("id", int):
        returncode = 400

    response=jsonify(movies_list)
    response.status_code=returncode

    return response



@app.route('/users')
def get_users():
    engine = create_engine('sqlite:///movies.db')
    # Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    users = session.query(User).all()
    users_list = []
    for user in users:
        users_list.append({
            'id': user.id,
            'name': user.name
        })
    return jsonify(users_list)

@app.route('/ratings')
def get_ratings():
    engine = create_engine('sqlite:///movies.db')
    # Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    ratings = session.query(Rating).all()
    ratings_list = []
    for rating in ratings:
        ratings_list.append({
            'id': rating.id,
            'user_id': rating.user_id,
            'movie_id': rating.movie_id,
            'rating': rating.rating
        })
    return jsonify(ratings_list)

@app.route('/movies/<int:movie_id>/ratings')
def get_movie_ratings(movie_id):
    engine = create_engine('sqlite:///movies.db')
    # Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    movie = session.query(Movie).filter_by(id=movie_id).first()
    if movie is None:
        return jsonify({'error': 'Movie not found'}), 404
    ratings = movie.ratings
    ratings_list = []
    for rating in ratings:
        ratings_list.append({
            'user_id': rating.user_id,
            'rating': rating.rating
        })
    return jsonify(ratings_list)

@app.route('/users/<int:user_id>/ratings')
def get_user_ratings(user_id):
    engine = create_engine('sqlite:///movies.db')
    # Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    user = session.query(User).filter_by(id=user_id).first()
    if user is None:
        return jsonify({'error': 'User not found'}), 404
    ratings = user.ratings
    ratings_list = []
    for rating in ratings:
        ratings_list.append({
            'movie_id': rating.movie_id,
            'rating': rating.rating
        })
    return jsonify(ratings_list)


@app.route('/top_five_all')
def get_top_rated_movies():
    engine = create_engine('sqlite:///movies.db')
    Session = sessionmaker(bind=engine)
    session = Session()

    query = session.query(
        Movie.id,
        Movie.title,
        Movie.year_of_release.label('yearOfRelease'),
        Movie.running_time.label('runningTime'),
        Movie.genre,
        func.round(func.avg(Rating.rating).cast(Numeric(precision=2, scale=1)), 1).label('averageRating')
    ).join(Rating).group_by(Movie.id).order_by(desc('averageRating')).limit(5)

    movies = query.all()

    movies_list = []
    for movie in movies:
        movie_dict = {
            'id': movie.id,
            'title': movie.title,
            'yearOfRelease': movie.yearOfRelease,
            'runningTime': movie.runningTime,
            'genre': movie.genre,
            'averageRating': movie.averageRating
        }
        movies_list.append(movie_dict)

    return jsonify(movies_list)

@app.route('/top5_user_ratings/<int:user_id>')
def get_top5_user_ratings(user_id):
    engine = create_engine('sqlite:///movies.db')
    Session = sessionmaker(bind=engine)
    session = Session()

    query = session.query(
        Movie.title,
        Rating.rating
    ).join(Rating).filter(Rating.user_id == user_id).order_by(Rating.rating.desc()).limit(5)

    top_ratings = query.all()

    top_ratings_list = []
    for rating in top_ratings:
        rating_dict = {
            'title': rating.title,
            'rating': rating.rating
        }
        top_ratings_list.append(rating_dict)

    return jsonify(top_ratings_list)


#@app.route('/update_rating/<int:user_id>/<int:movie_id>/<int:new_rating>', methods=['POST'])
@app.route('/update_rating')

def update_rating():
    engine = create_engine('sqlite:///movies.db')
    Session = sessionmaker(bind=engine)
    session = Session()
    #return request.args
    user_id= request.args.get('user_id')
    movie_id= request.args.get('movie_id')
    new_rating= request.args.get('new_rating')



    # check if movie and user exist
    movie = session.query(Movie).filter(Movie.id == movie_id).first()
    user = session.query(User).filter(User.id == user_id).first()

    if not movie:
        return jsonify({'message': f'Movie with id {movie_id} not found.'}), 404
    if not user:
        return jsonify({'message': f'User with id {user_id} not found.'}), 404

    # check if rating for this user and movie already exists
    rating = session.query(Rating).filter(Rating.user_id == user_id, Rating.movie_id == movie_id).first()

    if rating:
        # rating already exists, update it
        rating.rating = new_rating
        session.commit()

        return jsonify({'message': f'Rating updated for user {user_id} and movie {movie_id}.'}), 200
    else:
        # rating doesn't exist, create it
        rating = Rating(user_id=user_id, movie_id=movie_id, rating=new_rating)
        session.add(rating)
        session.commit()

        return jsonify({'message': f'Rating created for user {user_id} and movie {movie_id}.'}), 201



if __name__ == '__main__':
    app.run(port=8000)