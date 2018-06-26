using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

namespace Movies_crawler
{
    class Program
    {
        private static string OMDB_API_KEY = "1608d4d2";
        private static string OMDB_API_URL = "http://www.omdbapi.com?apikey=";

        private static List<string> MovieTitles = new List<string>
        {
            // drama movies
            "Titanic",
            "Forrest Gump",
            "The Shawshank Redemption",
            "American Beauty ",
            "Schindler's List",
            "American History X",
            "One Flew Over the Cuckoo's Nest",
            "The Green Mile",
            "City of God",
            "Casablanca",
            "The Godfather",
            "Good Will Hunting",
            "Trainspotting",
            "Gran Torino",
            "Million Dollar Baby",
            "The Curious Case of Benjamin Button",
            "Beautiful Mind",
            "There Will Be Blood",
            "Magnolia",
            "Rain Man",
            // action movies
            "Mad Max: Fury Road",
            "Black Panther",
            "Wonder Woman",
            "Inception",
            "The Avengers",
            "Batman Begins",
            "Blade",
            "Bad Boys",
            "Terminator",
            "Die Hard",
            "John Wick",
            "Robocop",
            "Lethal Weapon",
            "Speed",
            "300",
            "First Blood",
            "The Delta Force",
            "The Rock",
            "Taken",
            "Crank",
            // comedy movies
            "Airplane!",
            "Dumb and Dumber",
            "Animal House",
            "Office Space",
            "Old School",
            "Clerks",
            "Happy Gilmore",
            "Meet the Parents",
            "The 40-Year-Old Virgin",
            "The Gods Must Be Crazy",
            "Trading Places",
            "There's Something About Mary",
            "Anchorman: The Legend of Ron Burgundy",
            "The Hangover",
            "American Pie",
            "Harold & Kumar Go to White Castle",
            "Scary Movie",
            "Superbad",
            "Best in Show",
            "Home Alone",
            // horror movies
            "It Follows",
            "The Mist",
            "Return of the Living Dead",
            "Scream",
            "A Nightmare on Elm Street",
            "The Texas Chain Saw Massacre",
            "The Exorcist",
            "The Omen",
            "The Ring",
            "Poltergeist",
            "Psycho",
            "The Shining",
            "The Evil Dead",
            "Identity",
            "Thesis",
            "Bram Stoker's Dracula",
            "The Collection",
            "I Spit on Your Grave",
            "The Conjuring",
            "Hush",
            // fantasy
            "The Lord of the Rings: The Fellowship of the Ring",
            "Avatar",
            "Hugo",
            "Big Fish",
            "Pan's Labyrinth",
            "Life of Pi",
            "Indiana Jones and the Last Crusade",
            "Mr. Nobody",
            "Star Wars: Episode IV - A New Hope",
            "Harry Potter and the Deathly Hallows: Part 2",
            "Midnight in Paris",
            "The Hobbit: An Unexpected Journey",
            "Harry Potter and the Sorcerer's Stone",
            "The Wizard of Oz",
            "Fantastic Beasts and Where to Find Them",
            "Doctor Strange",
            "Hereafter",
            "Dark City",
            "Sleepy Hollow",
            "Alice in Wonderland",
        };

        static void Main(string[] args)
        {
            Directory.CreateDirectory("../../data");
            using (var client = new HttpClient())
            {
                for (int i = 1; i <= MovieTitles.Count; i++)
                {
                    var response = client.GetStringAsync($"{OMDB_API_URL}{OMDB_API_KEY}&t={MovieTitles[i - 1]}&plot=full");
                    WriteToFile(response, i).Wait();
                }
            }
        }

        private static async Task WriteToFile(Task<string> response, int index)
        {
            var stringResponse = await response;
            File.WriteAllText($"../../../../data/{String.Format("{0:D3}", index)}.json", stringResponse);
        }
    }
}
