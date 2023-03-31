from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Movie(Base):
    __tablename__ = 'movies'

    id = Column(Integer, primary_key=True)
    title = Column(String)
    year_of_release = Column(Integer)
    running_time = Column(Integer)
    genre = Column(String)
    ratings = relationship('Rating', back_populates='movie')

    @property
    def average_rating(self):
        if self.ratings:
            return sum([rating.rating for rating in self.ratings]) / len(self.ratings)
        else:
            return None

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    ratings = relationship('Rating', back_populates='user')

class Rating(Base):
    __tablename__ = 'ratings'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    movie_id = Column(Integer, ForeignKey('movies.id'))
    rating = Column(Integer)
    user = relationship('User', back_populates='ratings')
    movie = relationship('Movie', back_populates='ratings')


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///movies.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

movie1 = Movie(title='The Shawshank Redemption', year_of_release=1994, running_time=142, genre='Drama')
movie2 = Movie(title='The Godfather', year_of_release=1972, running_time=175, genre='Drama')
movie3 = Movie(title='The Dark Knight', year_of_release=2008, running_time=152, genre='Action')
movie4 = Movie(title='Inception', year_of_release=2010, running_time=148, genre='Action')
movie5 = Movie(title='The Matrix', year_of_release=1999, running_time=136, genre='Action')
movie6 = Movie(title='Forrest Gump', year_of_release=1994, running_time=142, genre='Drama')
movie7 = Movie(title='Pulp Fiction', year_of_release=1994, running_time=154, genre='Crime')
movie8 = Movie(title='Fight Club', year_of_release=1999, running_time=139, genre='Drama')
movie9 = Movie(title='The Silence of the Lambs', year_of_release=1991, running_time=118, genre='Thriller')
movie10 = Movie(title='Goodfellas', year_of_release=1990, running_time=145, genre='Crime')
movie11 = Movie(title='The Godfather: Part II', year_of_release=1974, running_time=202, genre='Drama')
movie12 = Movie(title='Se7en', year_of_release=1995, running_time=127, genre='Crime')
movie13 = Movie(title='The Usual Suspects', year_of_release=1995, running_time=106, genre='Thriller')
movie14 = Movie(title='The Departed', year_of_release=2006, running_time=151, genre='Crime')
movie15 = Movie(title='Gladiator', year_of_release=2000, running_time=155, genre='Action')
movie16 = Movie(title='The Lion King', year_of_release=1994, running_time=88, genre='Animation')
movie17 = Movie(title='Jurassic Park', year_of_release=1993, running_time=127, genre='Science Fiction')
movie18 = Movie(title='Star Wars: Episode IV - A New Hope', year_of_release=1977, running_time=121, genre='Science Fiction')
movie19 = Movie(title='The Terminator', year_of_release=1984, running_time=107, genre='Science Fiction')
movie20 = Movie(title='Alien', year_of_release=1979, running_time=117, genre='Science Fiction')
movie21 = Movie(title='The Exorcist', year_of_release=1973, running_time=122, genre='Horror')
session.add_all([movie1, movie2, movie3, movie4, movie5, movie6, movie7, movie8, movie9, movie10,
                 movie11, movie12, movie13, movie14, movie15, movie16, movie17, movie18, movie19,
                 movie20, movie21])

user1 = User(name='John')
user2 = User(name='Jane')
user3 = User(name='Bob')
user4 = User(name='Alice')
user5 = User(name='David')
user6 = User(name='Michael')
user7 = User(name='Sarah')
user8 = User(name='Emily')
user9 = User(name='William')
user10 = User(name='Oliver')
session.add_all([user1, user2, user3, user4, user5, user6, user7, user8, user9, user10])


rating1 = Rating(user=user1, movie=movie1, rating=5)
rating2 = Rating(user=user1, movie=movie2, rating=4)
rating3 = Rating(user=user1, movie=movie3, rating=5)
rating4 = Rating(user=user2, movie=movie1, rating=4)
rating5 = Rating(user=user2, movie=movie2, rating=5)
rating6 = Rating(user=user2, movie=movie3, rating=4)
rating7 = Rating(user=user3, movie=movie1, rating=3)
rating8 = Rating(user=user3, movie=movie3, rating=4)
rating9 = Rating(user=user3, movie=movie4, rating=5)
rating10 = Rating(user=user4, movie=movie2, rating=4)
rating11 = Rating(user=user4, movie=movie4, rating=5)
rating12 = Rating(user=user4, movie=movie5, rating=3)
rating13 = Rating(user=user5, movie=movie1, rating=5)
rating14 = Rating(user=user5, movie=movie3, rating=4)
rating15 = Rating(user=user5, movie=movie4, rating=5)
rating16 = Rating(user=user6, movie=movie2, rating=5)
rating17 = Rating(user=user6, movie=movie4, rating=4)
rating18 = Rating(user=user6, movie=movie5, rating=3)
rating19 = Rating(user=user7, movie=movie1, rating=4)
rating20 = Rating(user=user7, movie=movie2, rating=5)
rating21 = Rating(user=user7, movie=movie3, rating=4)
rating22 = Rating(user=user8, movie=movie2, rating=4)
rating23 = Rating(user=user8, movie=movie3, rating=5)
rating24 = Rating(user=user8, movie=movie5, rating=3)
rating25 = Rating(user=user9, movie=movie1, rating=3)
rating26 = Rating(user=user9, movie=movie4, rating=4)
rating27 = Rating(user=user9, movie=movie5, rating=5)
rating28 = Rating(user=user10, movie=movie2, rating=4)
rating29 = Rating(user=user10, movie=movie4, rating=5)
rating30 = Rating(user=user10, movie=movie5, rating=3)
rating31 = Rating(user=user1, movie=movie6, rating=5)
rating32 = Rating(user=user1, movie=movie7, rating=4)
rating33 = Rating(user=user1, movie=movie8, rating=5)
rating34 = Rating(user=user2, movie=movie6, rating=4)
rating35 = Rating(user=user2, movie=movie7, rating=5)
rating36 = Rating(user=user2, movie=movie8, rating=4)
rating37 = Rating(user=user3, movie=movie6, rating=3)
rating38 = Rating(user=user3, movie=movie8, rating=4)
rating39 = Rating(user=user3, movie=movie9, rating=5)
rating40 = Rating(user=user4, movie=movie7, rating=4)
rating41 = Rating(user=user4, movie=movie9, rating=5)
rating42 = Rating(user=user4, movie=movie10, rating=3)
rating43 = Rating(user=user5, movie=movie6, rating=4)
rating44 = Rating(user=user5, movie=movie8, rating=3)
rating45 = Rating(user=user5, movie=movie9, rating=4)
rating46 = Rating(user=user6, movie=movie7, rating=5)
rating47 = Rating(user=user6, movie=movie9, rating=4)
rating48 = Rating(user=user6, movie=movie10, rating=5)
rating49 = Rating(user=user7, movie=movie6, rating=3)
rating50 = Rating(user=user7, movie=movie7, rating=4)
rating51 = Rating(user=user7, movie=movie10, rating=4)
rating52 = Rating(user=user8, movie=movie6, rating=4)
rating53 = Rating(user=user8, movie=movie8, rating=5)
rating54 = Rating(user=user8, movie=movie10, rating=3)
rating55 = Rating(user=user9, movie=movie7, rating=4)
rating56 = Rating(user=user9, movie=movie9, rating=5)
rating57 = Rating(user=user9, movie=movie10, rating=4)
rating58 = Rating(user=user10, movie=movie6, rating=3)
rating59 = Rating(user=user10, movie=movie8, rating=4)
rating60 = Rating(user=user10, movie=movie10, rating=5)

session.add_all([rating1, rating2, rating3, rating4, rating5, rating6, rating7, rating8, rating9, rating10, rating11, rating12, rating13, rating14, rating15, rating16, rating17, rating18, rating19, rating20, rating21, rating22, rating23, rating24, rating25, rating26, rating27, rating28, rating29, rating30, rating31, rating32, rating33, rating34, rating35, rating36, rating37, rating38, rating39,
                 rating40, rating41, rating42, rating43, rating44, rating45, rating46, rating47, rating48,
                 rating49, rating50, rating51, rating52, rating53, rating54, rating55, rating56, rating57,
                 rating58, rating59, rating60])



session.commit()

