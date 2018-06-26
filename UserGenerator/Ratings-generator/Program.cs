using CsvHelper;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Ratings_generator
{
    class Program
    {
        private static Random rnd = new Random();
        private static List<Rating> records = new List<Rating>();

        // Sample data:
        // 1100 users
        // 100 movies in 5 categories:
        // Indexes 0-20 Drama, 21-40 Action, 41-60 Comedy, 61-80 Horror, 81-100 Thriller
        static void Main(string[] args)
        {
            // Generate 200 drama lovers
            GenerateRatings(1, 200, MoviesCategory.Drama, RatingsValue.Positive, 60);
            GenerateRatings(1, 200, MoviesCategory.Action | MoviesCategory.Comedy | MoviesCategory.Horror | MoviesCategory.Fantasy, RatingsValue.Negative, 10);

            // Generate 200 action lovers
            GenerateRatings(201, 200, MoviesCategory.Action, RatingsValue.Positive, 60);
            GenerateRatings(201, 200, MoviesCategory.Comedy | MoviesCategory.Drama | MoviesCategory.Horror | MoviesCategory.Fantasy, RatingsValue.Negative, 10);

            // Generate 200 comedy lovers
            GenerateRatings(401, 200, MoviesCategory.Comedy, RatingsValue.Positive, 60);
            GenerateRatings(401, 200, MoviesCategory.Action | MoviesCategory.Drama | MoviesCategory.Horror | MoviesCategory.Fantasy, RatingsValue.Negative, 10);

            // Generate 200 horror lovers
            GenerateRatings(601, 200, MoviesCategory.Horror, RatingsValue.Positive, 60);
            GenerateRatings(601, 200, MoviesCategory.Action | MoviesCategory.Drama | MoviesCategory.Comedy | MoviesCategory.Fantasy, RatingsValue.Negative, 10);

            // Generate 200 fantasy lovers
            GenerateRatings(801, 200, MoviesCategory.Fantasy, RatingsValue.Positive, 60);
            GenerateRatings(801, 200, MoviesCategory.Action | MoviesCategory.Drama | MoviesCategory.Horror | MoviesCategory.Comedy, RatingsValue.Negative, 10);

            // Generate 100 random users
            GenerateRatings(1001, 100, MoviesCategory.Drama | MoviesCategory.Action | MoviesCategory.Comedy | MoviesCategory.Horror | MoviesCategory.Fantasy, RatingsValue.Random, 20);

            // Make an ATTACK!!! 
            // Add 100 fake drama lovers that rate highly 1 film from another category
            GenerateRatings(1101, 100, MoviesCategory.Drama, RatingsValue.Positive, 60);
            // movie with index 21 is a comedy
            GenerateRatings(1101, 100, 21, RatingsValue.Positive);

            using (TextWriter writer = File.CreateText("../../ratings.csv"))
            {
                var csv = new CsvWriter(writer);
                csv.WriteRecords(records);
            }
        }

        private static void GenerateRatings(int usersStartingIndex, int numberOfUsers, MoviesCategory moviesCategory, RatingsValue ratingsValue, double percentageOfRatedMovies)
        {
            for (int i = usersStartingIndex; i < usersStartingIndex + numberOfUsers; i++)
            {
                var movieIndexes = GenerateMovieIndexes(moviesCategory, percentageOfRatedMovies);
                foreach (var movieIndex in movieIndexes)
                {
                    var rating = GenerateRating(ratingsValue);
                    records.Add(new Rating(i, movieIndex, rating));
                }
            }
        }

        private static void GenerateRatings(int usersStartingIndex, int numberOfUsers, int movieIndex, RatingsValue ratingsValue)
        {
            for (int i = usersStartingIndex; i < usersStartingIndex + numberOfUsers; i++)
            {
                var rating = GenerateRating(ratingsValue);
                records.Add(new Rating(i, movieIndex, rating));
            }
        }

        private static double GenerateRating(RatingsValue ratingsValue)
        {
            var rating = 10;

            switch (ratingsValue)
            {
                case RatingsValue.Negative:
                    rating = rnd.Next(2, 5);
                    break;
                case RatingsValue.Neutral:
                    rating = rnd.Next(5, 8);
                    break;
                case RatingsValue.Positive:
                    rating = rnd.Next(8, 11);
                    break;
                case RatingsValue.Random:
                    rating = rnd.Next(2, 11);
                    break;             
            }

            return (double)rating / 2;
        }

        private static List<int> GenerateMovieIndexes(MoviesCategory moviesCategory, double percentageOfRatedMovies)
        {
            var movieIndexes = new List<int>();

            if ((moviesCategory & MoviesCategory.Drama) == MoviesCategory.Drama)
            {
                movieIndexes.AddRange(Enumerable.Range(1, 20));
            }

            if ((moviesCategory & MoviesCategory.Action) == MoviesCategory.Action)
            {
                movieIndexes.AddRange(Enumerable.Range(21, 20));
            }

            if ((moviesCategory & MoviesCategory.Comedy) == MoviesCategory.Comedy)
            {
                movieIndexes.AddRange(Enumerable.Range(41, 20));
            }

            if ((moviesCategory & MoviesCategory.Horror) == MoviesCategory.Horror)
            {
                movieIndexes.AddRange(Enumerable.Range(61, 20));
            }

            if ((moviesCategory & MoviesCategory.Fantasy) == MoviesCategory.Fantasy)
            {
                movieIndexes.AddRange(Enumerable.Range(81, 20));
            }

            return movieIndexes.Where(x => rnd.Next(1, 101) <= percentageOfRatedMovies).ToList();
        }

        enum RatingsValue
        {
            Negative, // 1-2
            Neutral, // 2.5-3.5
            Positive, // 4-5
            Random
        }

        [Flags]
        enum MoviesCategory
        {
            Drama = 1,
            Action = 2,
            Comedy = 4,
            Horror = 8,
            Fantasy = 16
        }
    }

    class Rating
    {
        public Rating(int userID, int moviedId, double rating)
        {
            this.UserId = userID;
            this.MovieId = moviedId;
            this.Value = rating;
        }

        public int UserId { get; set; }

        public int MovieId { get; set; }

        public double Value { get; set; }
    }
}
