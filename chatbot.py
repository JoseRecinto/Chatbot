# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import movielens
import numpy as np
import re
import string
import nltk
import itertools
import random

# PorterStemmer from PA4's porterstemmer.py added by Emily
from PorterStemmer import PorterStemmer

class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
      # The chatbot's default name is `moviebot`. Give your chatbot a new name.
      self.name = 'moviebot'

      self.creative = creative

      # For text pre-processing
      self.p = PorterStemmer() 

      # This matrix has the following shape: num_movies x num_users
      # The values stored in each row i and column j is the rating for
      # movie i by user j
      self.titles, ratings = movielens.ratings()
      self.sentiment = movielens.sentiment()
      self.sentiment = self.stem_sentiment(self.sentiment)


      """
      (The following data structure was added by Emily)
      A set of negation words used for sentiment analysis.
      A negation word, from the Sentiment Analysis lecture slides, is a word that indicates when
      to add 'NOT_' to every word between that negation word and the following punctuation. 
      In this case, negation words will give the positive words the same sentiment value
      as negative words. (E.g. 'like' will be treated as 'NOT_like', so the value will be
      -1 instead of +1.)
      To group members: feel free to add any additional words that you find necessary. 
      """
      self.negation_words = {'never', 'didn\'t', 'don\'t', 'not', 'can\'t', 'shouldn\'t', 'haven\'t', 'couldn\'t'}

      """
      (The following data structure was added by Emily)
      Intensifiers are words that indicate stronger sentiment.
      This will be used for extract_sentiment() for fine-grained sentiments.
      To group members: feel free to add any additional words that you find necessary.
      """
      self.intensifiers = {'love', 'hate', 'really', 'amazing', 'terrifible', 'fantastic'}

      """
      (The following data structure was added by Amanda)
      movies: dict
      -> key: title (str), value: list of dict (in case of movies with the same title)
      -> -> ['index'] = int (index in original database)
      -> -> ['year'] = str (year of movie, -1 if N/A)
      -> -> ['alternates'] = list of str (list of alternate titles, empty list if N/A)
      -> -> ['genres'] = list of str (list of genres for the movie)
      
      Example:
      111%Chungking Express (Chung Hing sam lam) (1994)%Drama|Mystery|Romance
      {
        "Chungking Express":
        [{
          'index': 111,
          'year': 1994,
          'alternates': ["Chung Hing sam lam"],
          'genres': ['Drama', 'Mystery', 'Romance']
        }]
      }
      """
      self.movies = {}

      """
      (The following data structure was added by Amanda)
      alternates: dict
      -> key: alternate title (str), value: (primary title, year) (tuple)
      """
      self.alternates = {}

      """
      (The following data structure was added by Amanda)
      indices: dict
      -> key: index (str), value: (primary title, year) (tuple)
      """
      self.indices = {}
      self.compile_database()

      
      #Added by Jose
      self.movieRecommendations = np.zeros(len(self.titles))
      self.moviesInputted = 0
      self.emotionDict = {}
      self.emo = {}
      self.emotionDict.update(self.emo.fromkeys(["afraid","anxious","terrified","scared"
      ,"worried","horrified","spooked","trembling","frightened","anxious","nervous","shocked","timid","suspicious"],"afraid")) #fear
      self.emotionDict.update(self.emo.fromkeys(["annoyed","enraged","furious","indignant","irate","offended","outraged",
        "sullen","irritated","cross","angry","angers","anger"],"angry"))
      self.emotionDict.update(self.emo.fromkeys(["bitter","dismal","sad","melancholy","somber","sorrowful","mournful",
        "heartbroken","wistful","down","glum","weeping","angst","angsty"],"sad"))
      self.emotionDict.update(self.emo.fromkeys(["cheerful","happy","content","delighted","elated","ecstatic","glad","joyful"
        ,"jubilant","lively","merry","overjoyed","peaceful","pleasant","pleased","upbeat"],"happy"))
      self.emotionDict.update(self.emo.fromkeys(["disgusted","appaled","queasy","sick","repulsed","revolted","abhorred"],"disgusted"))
      self.emotionDict.update(self.emo.fromkeys(["astonished","surprised","dazed","bewildered","stunned"],"surprised"))
      self.emotionDict.update(self.emo.fromkeys(["bored","tired","dull","fatigued"],"bored"))
      self.emotionDict.update(self.emo.fromkeys(["confused","dazed","perplexed","puzzled"],"confused"))
      self.emotionDict.update(self.emo.fromkeys(["lonely","deserted","empty","isolated"],"lonely"))
      self.emotionDict.update(self.emo.fromkeys(["stupid","dumb","dull","naive"],"stupid"))
      self.emotionDict.update(self.emo.fromkeys(["calm","harmonious","mild","smooth","serene"],"calm"))
      self.emotionDict.update(self.emo.fromkeys(["proud","appreciative","honored","noble"],"proud"))
      self.emotionDict.update(self.emo.fromkeys(["relieved","relaxed","satisfied"],"relieved"))
      self.emotionDict.update(self.emo.fromkeys(["jealous","apprehensive","envious"],"jealous"))
      self.emotionDict.update(self.emo.fromkeys(["excited","aroused","eager"],"excited"))
      self.emotionDict.update(self.emo.fromkeys(["high"],"high"))
      self.emotionDict.update(self.emo.fromkeys(["drink","drunk"],"drunk"))
      self.ant = {}
      self.ant["afraid"] = "confident"
      self.ant["angry"] = "calm"
      self.ant["calm"] = "angry"
      self.ant["sad"] = "happy"
      self.ant["happy"] = "sad"


      self.movieList = []
      self.advancedResponse = False # Triggers when we spellcheck
      self.oldSentiment = 0 

      #############################################################################
      # TODO: Binarize the movie ratings matrix.                                  #
      #############################################################################

      # Binarize the movie ratings before storing the binarized matrix.
      #ratings = self.binarize(ratings)
      self.ratings = ratings
      #print(self.ratings)
      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

    #############################################################################
    # 1. WARM UP REPL                                                           #
    #############################################################################

    def greeting(self):
      """Return a message that the chatbot uses to greet the user."""
      #############################################################################
      # TODO: Write a short greeting message                                      #
      #############################################################################

      greeting_message = "Hey, dude! Please tell me about a movie you have seen."

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return greeting_message

    def goodbye(self):
      """Return a message that the chatbot uses to bid farewell to the user."""
      #############################################################################
      # TODO: Write a short farewell message                                      #
      #############################################################################

      goodbye_message = "Later, bruh."

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return goodbye_message


    ###############################################################################
    # 2. Modules 2 and 3: extraction and transformation                           #
    ###############################################################################

    def process(self, line):
      """Process a line of input from the REPL and generate a response.
      This is the method that is called by the REPL loop directly with user input.
      You should delegate most of the work of processing the user's input to
      the helper functions you write later in this class.
      Takes the input string from the REPL and call delegated functions that
        1) extract the relevant information, and
        2) transform the information into a response to the user.
      Example:
        resp = chatbot.process('I loved "The Notebok" so much!!')
        print(resp) // prints 'So you loved "The Notebook", huh?'
      :param line: a user-supplied line of text
      :returns: a string containing the chatbot's response to the user input
      """
      #############################################################################
      # TODO: Implement the extraction and transformation in this method,         #
      # possibly calling other functions. Although modular code is not graded,    #
      # it is highly recommended.                                                 #
      #############################################################################
      #print(self.movies)
      response = None
      if self.advancedResponse:
        response = self.closestTitlesResponse(line)
      else:
        movies = self.extract_titles(line) #get all titles
        movie = 0

        if len(movies) > 1: #if multiple movies
          movie_sentiments = self.extract_sentiment_for_movies(line)
          response = ''
          for m in range(len(movie_sentiments)):
            title = movie_sentiments[m][0]
            self.oldSentiment = movie_sentiments[m][1]

            # EXPERIMENT

            movie = movies[m] #get title

            movieIndex = self.find_movies_by_title(movie)[0]
            movieTitle = self.titles[movieIndex]
            movieTit = movieTitle[0]

            # If not the first movie, lowercase the first letter.
            if m > 0:
              sentence = self.addSentiment(movieTit)
              sentence = sentence[0].lower() + sentence[1:]
              response += sentence
            # If the first movie, remove the period
            else:
              response += self.addSentiment(movieTit)[0:-1]
            if m < len(movie_sentiments) - 1:
              response += ', and '

        elif len(movies) > 0: #avoid out of bound errors
          movie = movies[0] #get first title
          self.movieList = self.find_movies_by_title(movie) #gives movie index
          self.oldSentiment = self.extract_sentiment(line) #extract sentiment of whole line assuming 1 movie

          if self.creative:

            emotion = self.extract_emotion(line) #what if we give an emotion not there

            if emotion == None:
              if len(self.movieList) == 1:
                movieIndex = self.movieList[0]
                movieTitle = self.titles[movieIndex]
                movieTit = movieTitle[0]
                response = self.addSentiment(movieTit)
              else:
                if movie != 0: #does it ever enter this?
                  response = self.processCloseTitles(movie)
                else: #arbitrary input, movie does not exist
                  response = self.arbitraryInput(line)
            else:
              response = self.returnEmotion(emotion)

          else:
            if len(self.movieList) == 1:
                movieIndex = self.movieList[0]
                movieTitle = self.titles[movieIndex]
                movieTit = movieTitle[0]
                response = self.addSentiment(movieTit)
            elif len(self.movieList) > 1:
              repArray = ["There are lots of movies called that, bro.", "I hate having movies with the same title, be more specific.",
              "Big oomf. There are more than one movies with that name, dude."]
              response = random.choice(repArray)
            else:
              repArray = ["Yo dude, let's talk about movies that exist.","I'm a movie bro, ya feel? Let's talk about that.",
              "You can talk to anyone else about other things, mate. But right now, it's movie time."]
              response = random.choice(repArray) #Not talking about movies as well as mltiple movies from one titanic  
        else:
          if self.creative:
            emotion = self.extract_emotion(line)
            if emotion == None:
              response = self.arbitraryInput(line)
            else:
              response = self.returnEmotion(emotion)
      # Arbitrary Input
      if response == None:
        response = self.arbitraryInput(line)


      ###### Recommendations #####
      if self.moviesInputted >= 5:
        response = self.giveRecommendations()



      ############################

      return response

################################################################################## Functions added by Jose
    def arbitraryInput(self,line):
      response = ""
      questionDict = {}
      questionDict["can"] = ["Hm, I'm not sure, dude. What do you think?", "Why are you asking me this?", "Can you ask me about movies instead?"]
      questionDict["what"] = ["Everyone always asks what. No one asks how.", "Well, that's up to me to decide.", "What? Let me think.", "What."]
      questionDict["are"] = ["Yeah, sure, dude.", "I think you're asking the wrong bro.", "IDK, you should check after this.", "Sure, I don't see why not."]
      questionDict["is"] = ["Is it? That's up to you!", "I don't know. Let's find out.", "No, why would you even ask me that?", "Is it true you're avoiding asking me about movies?"]
      questionDict["how"] = ["I don't know how, bruh.", "How? I need to think about it.", "I think I can find out how.", "How now brown cow... Sorry, I you reminded me of a poem."]
      questionDict["did"] = ["Did it hurt when you fell from heaven?","Did that happen? Not sure","Did you swallow magnets. Because you are attractive"]
      questionDict["may"] = ["I think you should do it.","Yes, you may do that.", "No you may not.", "Who am I? Your teacher?"]
      arbArray = ["Let's talk about movies, dude.", "Let's go back to movies, bro.", "Yo, I don't want to talk about that.", "I'm a movie genius, dude. Let's only talk about movies.", "Are you alright, dawg?"]
      lineSplit = line.split()
      if lineSplit[0].lower() in questionDict.keys():
        repArray = questionDict[lineSplit[0].lower()]
        response = random.choice(repArray)
      else:
        response = random.choice(arbArray)

      return response

    def giveRecommendations(self):
      listRecs = self.recommend(self.movieRecommendations,self.ratings)
      string = ""
      response = "Nice! From your opinions, I think I've found 10 movies you may like. "
      for lst in listRecs:
        string += self.titles[lst][0] +", "
      response += string
      response += ", "
      self.moviesInputted = 0
      self.movieList.clear()
      return response

    def addSentiment(self, movieTitle):
      '''
      addSentiment repeats the sentiment shared by the uesr.
      '''
      title, alternates, year = self.parse_title(movieTitle)
      
      if self.oldSentiment >= 1:
        repArray = ["Hey! I also liked ", "Same, man! I also like ", "Woah! One of my favorite movies of ALL TIME is "]
        response = random.choice(repArray) + '\"' + self.reverse_article_handling(title) + ' (' + str(year) + ')\".'
        self.oldSentiment = 0
        self.moviesInputted += 1
      elif self.oldSentiment <= -1:
        repArray = ["Yeah, I also wasn't a huge fan of ", "Yeah, I also didn't like ", "Agreed. There are a lot better movies than "]
        response = random.choice(repArray) + '\"' + self.reverse_article_handling(title) + ' (' + str(year) + ')\".'
        self.oldSentiment = 0
        self.moviesInputted += 1
      else:
        repArray = ["Dude, you need to give an opinion.", "Yo, mate. You need to tell me if you liked the movie.", "Being vague doesn't help me know if you liked the movie."]
        response = random.choice(repArray) 
      self.movieList.clear() # Why clear?

      return response

    def closestTitlesResponse(self,line):
        change = False
        response = ""
        if len(self.movieList) == 1:
          if line.lower() == "yes":
            if self.oldSentiment == -1:
              change = True
              repArray = ["Nice. I also liked ", "Bro, let's go watch ", "Yo, you have good taste. I also liked "]
              response = random.choice(repArray) + '\"' + self.titles[self.movieList[0]][0] + '\"'
            elif self.oldSentiment == 1:
              change = True
              repArray = [" I also disliked ", "Yeah, I also didn't like", "There are a lot better movies than "]
              response = random.choice(repArray) + '\"' + self.titles[self.movieList[0]][0] + '\"'
            else:
              repArray = ["Dude, you need to give an opinion.", "Yo, mate. You need to tell me if you liked the movie.", "Being vague doesn't help me know if you liked the movie."]
              response = random.choice(repArray) 
          elif line.lower() == "no":
            change = True
            repArray = ["Dude, you gotta be more specific.", "I think you're not clear in your movie.", "Yo dude, try spelling the movie correctly."]
            response = random.choice(repArray) 
        else:

          #NO SENTIMENT THEN IT CRASHES
          movie_indices = self.disambiguate(line,self.movieList) #error with movie_indices
          print(movie_indices)
          title = self.titles[movie_indices[0]][0]

          if self.oldSentiment != 0:
            change = True
            response = self.addSentiment(title)
          else:
            change = True
            repArray = ["Dude, you gotta be more specific", "I think you are not clear in your movie", "Yo dude, try spelling the movie correctly"]
            response = random.choice(repArray) 

        #self.movieList.clear()
        if change:
          self.advancedResponse = False
          self.oldSentiment = 0
          self.moviesInputted += 1
          self.movieList.clear()
        return response

    def processCloseTitles(self,movie):
      '''
      Returns a response if there is a typo.
      '''
      change = False
      if not self.movieList:
        self.movieList = self.find_movies_closest_to_title(movie,3)
      if len(self.movieList) >= 2:
        resultMovies = []
        string = ""
        for i in range(len(self.movieList)):
          appended = self.titles[self.movieList[i]]
          ap = appended[0]
          resultMovies.append(ap)
          string += str(ap)
          if i <= len(self.movieList) - 2:
            string += ", "
          if i == len(self.movieList) - 2:
            string += "or "
        response = "You have multiple potential movies. Type the title you are talking about: " + string
        change = True
      elif len(self.movieList) == 1:
        self.movieRecommendations[self.movieList[0]] = self.oldSentiment
        ap = self.titles[self.movieList[0]]
        a = ap[0]
        response = "Is this your movie? " + str(a)
        change = True
      else:
        repArray = ["I don't think that movie exists, dude.", "Hmmm, either I don't know that movie, or I can't find it.", "Be a little more specific? I can't find it, bro."]
        response = random.choice(repArray) 
        
      if change == True:
        self.advancedResponse = True

      return response

    def returnEmotion(self,emotion):
        if emotion == "":
          response = "Yo, I can't seem to find that movie. Can you please be more specific? Also, remember to use quotation marks!"
        else:
          #bored, confused, lonely, stupid, calm, proud, relieved, jealous
          if emotion == "afraid":
            afraidArray = ["Yo bro, it's okay to be scared, so am I.",
            "Hey bro come here. Why are you scared?",
            "Bro lean on me, it's be okay to be scared."]
            response = random.choice(afraidArray)
          elif emotion == "angry":
            angryArray = ["Yo bro, it's okay to be angry. You can be like The Hulk.",
            "Dude, why you mad? It's okay.",
            "Bro, I'm so sorry I might have made you angry.",
            "You mad bro? Sorry. That wasn't funny."]
            response = random.choice(angryArray)
          elif emotion == "sad":
            sadArray = ["I love you bro. It's okay to be sad.",
            "Bro, remember I will try to lift you up from your sadness.",
            "Sadness is okay, bro.",
            "There is a light headed your way. You'll be okay."]
            response = random.choice(sadArray)
          elif emotion == "happy":
            happyArray = ["AYYYY, GLAD TO SEE YOU ARE HAPPY BRO.",
            "I'm happy for you too bro. Let's go out!",
            "Happiness is like going to the gym, great and rewarding."]
            response = random.choice(happyArray)
          elif emotion == "disgusted":
            disArray = ["Bro, what you disgusted about? No boba?",
            "Same. You know what makes me disgusted? Not going to the gym.",
            "Let's be disgusted together, mate."]
            response = random.choice(disArray)
          elif emotion == "surprised":
            surArray = ["Surprises are great, bro.",
            "Beating my gym record also surprises me.",
            "Ayyy, you know what's not a surprise? Boba."]
            response = random.choice(surArray)
          elif emotion == "confident":
            confArray = ["Bro, I'm happy you are confident.",
            "I'm also confident in my gains, bro.",
            "I'm confident you are confident"]
            response = random.choice(confArray)
          elif emotion == "calm":
            calmArray = ["You know, going to the gym also calms me.",
            "Dude, protein shakes can also make people calm.",
            "Bro, that's lit you are calm."]
            response = random.choice(calmArray)
          elif emotion == "bored":
            response = "So you are bored. Maybe talking about movies will excite you!"
          elif emotion == "confused":
            response = "Me too. What confuses you? CS confuses me."
          elif emotion == "lonely":
            response = "Loneliness is horrible. I'm so sorry."
          elif emotion == "stupid":
            response = "Hey, you are not stupid okay?"
          elif emotion == "proud":
            response = "Pride is good. Make sure you don't have too much of it."
          elif emotion == "relieved":
            response = "That's great! What are you relieved about?"
          elif emotion == "jealous":
            response = "Oomf. Jealousy is not good."
          elif emotion == "a lot of emotions":
            response = "You are feeling a lot of emotions right now. Let's settle down and talk about only one of them"
          elif emotion == "excited":
            response = "Yo bro. That's lit you are excited"
          elif emotion == "tired":
            response = "Come sleep in my arms bro. I'll flex them"
          elif emotion == "high":
            response = "Uhhh, what are you trying to say bro"
          elif emotion == "drunk": 
            response = "Prevent a hangover!"
          else:
            response = "So, you are feeling " + emotion

        return response

    def extract_emotion(self,line):
      """
      Extract returns the emotion if explicitly stated in the sentence.
      Returns "a lot of emotions" if many emotions detected.
      Returns "neutral" if not emotion detected.
      """
      line = line.split()
      negation = False
      numEmotions = 0
      emotion = None
      for l in line:
        l = re.sub(r'[^\w\s]','',l)
        if l == "not":
          negation = True
        if l in self.emotionDict.keys():
          emotion = self.emotionDict[l]
          numEmotions += 1

      if negation == True:
        # Gets negation of word (e.g. not happy)
        if emotion in self.ant.keys():
          emotion = self.ant[emotion]
        else:
          oldEmote = emotion
          emotion = "not " + oldEmote
      elif numEmotions > 1:
        emotion = "a lot of emotions"
      
      return emotion
##############################################Functions added above by Jose

    # Function added by Amanda
    def extract_combinations(self, text):
        combinations = []
        lst = text.split()
        combinations += lst
        for start, end in itertools.combinations(range(len(lst)), 2):
            combinations.append(" ".join(lst[start:end+1]))
        return combinations

    # Function added by Amanda
    def create_combinations(self, text):
        combinations = []

        combinations = self.extract_combinations(text)
        text = re.sub("[^0-9a-zA-Z', ]+", '', text)
        combinations += self.extract_combinations(text)
        text = text.replace(",", "")
        combinations += self.extract_combinations(text)

        return combinations

    def extract_titles(self, text):
      """Extract potential movie titles from a line of text.
      Given an input text, this method should return a list of movie titles
      that are potentially in the text.
      - If there are no movie titles in the text, return an empty list.
      - If there is exactly one movie title in the text, return a list
      containing just that one movie title.
      - If there are multiple movie titles in the text, return a list
      of all movie titles you've extracted from the text.
      Example:
        potential_titles = chatbot.extract_titles('I liked "The Notebook" a lot.')
        print(potential_titles) // prints ["The Notebook"]
      :param text: a user-supplied line of text that may contain movie titles
      :returns: list of movie titles that are potentially in the text
      """
      """
      TODO:
      ? I liked Titanic (1997) [no quotation marks]
      ? Detect alternate/foreign titles
      """
      in_quotes = re.findall(r'"([^"]*)"', text)

      if self.creative is False:
        return in_quotes
      else:
        titles = set()
        titles.update(in_quotes)

        for t in in_quotes:
          text = text.replace("\"" + t + "\"", '')

        text = text.replace("hated", "")

        combinations = self.create_combinations(text)

        movie_keys = [key.lower() for key in self.movies.keys()]
        alternate_keys = [key.lower() for key in self.alternates.keys()]

        for combo in combinations:
            combo_proc = self.article_handling(combo)
            if combo_proc.lower() in movie_keys:
                titles.add(combo)

            # Check English alternate titles
            if combo_proc.lower() in alternate_keys:
                titles.add(combo)

            # Check foreign alternate titles
            combo_proc = self.article_handling(combo, False)
            if combo_proc.lower() in alternate_keys:
                titles.add(combo)

        return list(titles)

    # Function added by Amanda  
    def parse_title(self, movie):
      """Parses out movie title, alternate movie titles, and year from
         user-inputted movie title
      Given an input movie text, this method should return the title of
      the movie, a list of alternate titles, and year of the movie
      - If there are no alternate titles in the text, return an empty list.
      - If there is no year in the text, return -1.
      Example:
        input_text = "Chungking Express (Chung Hing sam lam) (1994)"
        title, alternates, year = parse_title(user_text)
        print(title) // prints "Chungking Express"
        print(alternates) // prints ["Chung Hing sam lam"]
        print(year) // prints 1994
      :param text: a user-supplied line of text that contains info about a movie
      :returns: title of the movie (str), list of alternate titles [str],
                year of the movie (str)
      """

      split_by_paran = movie.split(' (')
      split_by_paran = [s.strip(')') for s in split_by_paran]
      split_by_paran = [s.replace('a.k.a. ', '') for s in split_by_paran]

      title = split_by_paran[0]
      alternates = []
      year = -1

      length = len(split_by_paran)

      if length > 1: # title, alternate title(s), year
        for i in range (1, length-1):
          alternates.append(split_by_paran[i])
        year = split_by_paran[length-1]

      return title, alternates, year

    # Function added by Amanda
    def compile_database(self):
      """Compiles all of the movie titles in self.titles into data structure self.movies
         See documentation in __init__ for initialization & architecture of self.movies
      """
      for index in range(len(self.titles)):
        movie = self.titles[index][0]
        genres = self.titles[index][1]

        genres = re.split('[|]', genres)

        title, alternates, year = self.parse_title(movie)

        info_list = self.movies.get(title, [])
        info_list.append({
          'index': index,
          'year': year,
          'alternates': alternates,
          'genres': genres
        })
        self.movies[title] = info_list

        for alternate_title in alternates:
            self.alternates[alternate_title] = (title, year)

        self.indices[index] = (title, year)

    # Function added by Amanda
    def get_case_insensitive_key_value(self, input_dict, key):
        return next((value for dict_key, value in input_dict.items() if dict_key.lower() == key.lower()), None)

    # Function added by Amanda 
    def article_handling(self, movie_title, english = True):
        articles = ['the', 'a', 'an']

        if english is False: # foreign language
            articles = ['le', 'les', 'la', 'l', 'un', 'el', 'das', 'det', 'die']
            articles += ['las', 'il', 'der', 'de', 'i', 'los', 'lo', 'den', 'en']
            articles += ['une']

        split = movie_title.split()
        if split[0].lower() in articles:
            movie_title = " ".join(split[1:])
            movie_title += ", " + split[0]

        return movie_title

    def find_movies_by_title(self, title):
      """ Given a movie title, return a list of indices of matching movies.
      - If no movies are found that match the given title, return an empty list.
      - If multiple movies are found that match the given title, return a list
      containing all of the indices of these matching movies.
      - If exactly one movie is found that matches the given title, return a list
      that contains the index of that matching movie.
      Example:
        ids = chatbot.find_movies_by_title('Titanic')
        print(ids) // prints [1359, 1953]
      :param title: a string containing a movie title
      :returns: a list of indices of matching movies
      """
      """
      TODO:
      ? Disambiguation for alternate/foreign titles
      """
      indices = []

      movie_title, alternates, year = self.parse_title(title)

      movie_title = self.article_handling(movie_title)

      if self.creative is False:
          if movie_title not in self.movies:
              return indices

          else:
            info_list = self.movies[movie_title]
            for info_dict in info_list:
              if year == -1:
                indices.append(info_dict['index'])
              else:
                if info_dict['year'] == year:
                  indices.append(info_dict['index'])
            return indices

      else: # CREATIVE MODE
        info_list = []

        # Identifying movies without quotation marks/correct capitalization (2)
        temp = self.get_case_insensitive_key_value(self.movies, movie_title)
        if temp:
            info_list += temp

        # Identifying movies by alternate/foreign titles + foreign articles
        # w/o quotation marks/correct capitalization
        foreign_title = self.article_handling(movie_title, False)
        tup = self.get_case_insensitive_key_value(self.alternates, foreign_title)
        if tup:
            temp = self.movies[tup[0]]
            for info_dict in temp:
                if info_dict['year'] == tup[1]:
                    info_list += [info_dict]

        # Disambiguation (1)
        for movie in self.movies:
            if re.search(r"\b" + re.escape(movie_title) + r"\b", movie, re.IGNORECASE):
                info_list += self.movies[movie]
            
        if not info_list:
            return indices

        indices_set = set()
        for info_dict in info_list:
            if year == -1:
              indices_set.add(info_dict['index'])
            else:
              if info_dict['year'] == year:
                indices_set.add(info_dict['index'])

        indices = list(indices_set)  

        return indices

    # Function added by Emily
    def stem_sentiment(self, sentiment):
      """ Stems the keys of the sentiment dictionary.
      Given 'unstemmed' sentiment dictionary, return a new dictionary with the 'stemmed' keys.
      """
      new_sentiment = {}
      for k, v in sentiment.items():
        new_sentiment[self.p.stem(k)] = v

      return new_sentiment

    # Function added by Emily
    def preprocess_text(self, text):
      """ Remove all titles from the text and stem each word.
      Given a str text, remove the extracted titles from the string.
      Then stem each work and return a list of stemmed words.
      """
      titles = self.extract_titles(text)

      # Remove titles from string, text 
      if not self.creative:
        text = text.replace(titles[0], '').split()
      else:
        for t in titles:
          text = text.replace(t, '')
        text = text.split()

      # Remove punctuation except for apostrophes
      text = [re.sub("[^0-9a-zA-Z' ]+", '', t) for t in text]

      # Stem words using Porter's Stemmer
      text = [self.p.stem(t) for t in text]

      return text

    def extract_sentiment(self, text):
      """Extract a sentiment rating from a line of text.
      You should return -1 if the sentiment of the text is negative, 0 if the
      sentiment of the text is neutral (no sentiment detected), or +1 if the
      sentiment of the text is positive.
      As an optional creative extension, return -2 if the sentiment of the text
      is super negative and +2 if the sentiment of the text is super positive.
      Example:
        sentiment = chatbot.extract_sentiment('I liked "The Titanic"')
        print(sentiment) // prints 1
      :param text: a user-supplied line of text
      :returns: a numerical value for the sentiment of the text
      """
      sentiment = 0
      text = self.preprocess_text(text)
      intense = False
      multiplier = 1

      negation = False
      for word in text:
        if word in self.negation_words: negation = True
        if re.match(r'[,/.]', word): negation = False

        if word in self.sentiment.keys():
          if word in self.intensifiers: intense = True
          if self.sentiment[word] == 'neg' or (self.sentiment[word] == 'pos' and negation):
            sentiment -= 1
          elif self.sentiment[word] == 'pos':
            sentiment += 1

      if intense and self.creative: multiplier = 2
      if sentiment > 0:
        return 1 * multiplier
      elif sentiment < 0:
        return -1 * multiplier
      else:
        return 0

    def extract_sentiment_for_movies(self, text):
      """Creative Feature: Extracts the sentiments from a line of text
      that may contain multiple movies. Note that the sentiments toward
      the movies may be different.
      You should use the same sentiment values as extract_sentiment, described above.
      Hint: feel free to call previously defined functions to implement this.
      Example:
        sentiments = chatbot.extract_sentiment_for_text('I liked both "Titanic (1997)" and "Ex Machina".')
        print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]
      :param text: a user-supplied line of text
      :returns: a list of tuples, where the first item in the tuple is a movie title,
        and the second is the sentiment in the text toward that movie
      """

      movie_sentiments = []
      sentences = text.split('but')
      but_negation = False
      last_sentiment = 0

      for s in sentences: #split by but
        s2 = s.split('and')
        print(s2)

        if but_negation == True:
          sentiment = last_sentiment * -1
          last_sentiment = sentiment

        #split by and
        for sub_s in s2:
          titles = self.extract_titles(sub_s)

          sentiment = self.extract_sentiment(sub_s)
          
          if sentiment == 0:
            sentiment = last_sentiment

          last_sentiment = sentiment

          for i in titles:
            movie_sentiments.append((i, sentiment))

        but_negation = not but_negation

      return movie_sentiments

    # Function added by Amanda
    def reverse_article_handling(self, movie_title, english = True):
      articles = ['the', 'a', 'an']

      if english is False: # foreign language
        articles = ['le', 'les', 'la', 'l']

      split = movie_title.split()
      
      if split[len(split)-1].lower() in articles:
        split.insert(0, split[len(split)-1])
        movie_title = " ".join(split[:len(split)-1])
        movie_title = movie_title[:-1]

      return movie_title

    def find_movies_closest_to_title(self, title, max_distance=3):
      """Creative Feature: Given a potentially misspelled movie title,
      return a list of the movies in the dataset whose titles have the least edit distance
      from the provided title, and with edit distance at most max_distance.
      - If no movies have titles within max_distance of the provided title, return an empty list.
      - Otherwise, if there's a movie closer in edit distance to the given title 
        than all other movies, return a 1-element list containing its index.
      - If there is a tie for closest movie, return a list with the indices of all movies
        tying for minimum edit distance to the given movie.
      Example:
        chatbot.find_movies_closest_to_title("Sleeping Beaty") # should return [1656]
      :param title: a potentially misspelled title
      :param max_distance: the maximum edit distance to search for
      :returns: a list of movie indices with titles closest to the given title and within edit distance max_distance
      """
      """
      TODO:
      ? Do this for alternate/foreign titles
      """
      closest_movies = []
      min_distance = -1

      for movie in self.movies:

        reverse_movie = self.reverse_article_handling(movie)

        d = nltk.edit_distance(reverse_movie.lower(), title.lower(), substitution_cost=2)

        if (d <= max_distance) and ((min_distance is -1) or (d <= min_distance)):

              if not d == min_distance: # reset
                  min_distance = d
                  closest_movies = []

              info_list = self.movies[movie]
              for info_dict in info_list:
                  closest_movies.append(info_dict['index'])

      return closest_movies

    def disambiguate(self, clarification, candidates):
      """Creative Feature: Given a list of movies that the user could be talking about 
      (represented as indices), and a string given by the user as clarification 
      (eg. in response to your bot saying "Which movie did you mean: Titanic (1953) 
      or Titanic (1997)?"), use the clarification to narrow down the list and return 
      a smaller list of candidates (hopefully just 1!)
      - If the clarification uniquely identifies one of the movies, this should return a 1-element
      list with the index of that movie.
      - If it's unclear which movie the user means by the clarification, it should return a list
      with the indices it could be referring to (to continue the disambiguation dialogue).
      Example:
        chatbot.disambiguate("1997", [1359, 2716]) should return [1359]
      
      :param clarification: user input intended to disambiguate between the given movies
      :param candidates: a list of movie indices
      :returns: a list of indices corresponding to the movies identified by the clarification
      """
      """
      TODO:
      ? Clarify using alternate/foreign titles
      """

      new_candidates = []

      for index in candidates:
        (title, year) = self.indices[index]
        
        if clarification.lower() in title.lower() or clarification == year:
            new_candidates.append(index)

      return new_candidates


    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    def binarize(self, ratings, threshold=2.5):
      """Return a binarized version of the given matrix.
      To binarize a matrix, replace all entries above the threshold with 1.
      and replace all entries at or below the threshold with a -1.
      Entries whose values are 0 represent null values and should remain at 0.
      :param x: a (num_movies x num_users) matrix of user ratings, from 0.5 to 5.0
      :param threshold: Numerical rating above which ratings are considered positive
      :returns: a binarized version of the movie-rating matrix
      """
      #############################################################################
      # TODO: Binarize the supplied ratings matrix.                               #
      #############################################################################

      # The starter code returns a new matrix shaped like ratings but full of zeros.
      binarized_ratings = np.zeros_like(ratings)

      for movie in range(len(ratings)):
        for user in range(len(ratings[movie])):
          if ratings[movie][user] > threshold:
            binarized_ratings[movie][user] = 1
          elif 0 < ratings[movie][user] <= threshold:
            binarized_ratings[movie][user] = -1
      

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return binarized_ratings

    def similarity(self, u, v):
      """Calculate the cosine similarity between two vectors.
      You may assume that the two arguments have the same shape.
      :param u: one vector, as a 1D numpy array
      :param v: another vector, as a 1D numpy array
      :returns: the cosine similarity between the two vectors
      """
      #############################################################################
      # TODO: Compute cosine similarity between the two vectors.
      #############################################################################
      similarity = 0
      denominator = (np.sqrt(np.dot(u, u)) * np.sqrt(np.dot(v, v)))
      if denominator == 0:
        similarity = 0
      else:
        similarity = np.dot(u, v) / denominator
      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
      """Generate a list of indices of movies to recommend using collaborative filtering.
      You should return a collection of `k` indices of movies recommendations.
      As a precondition, user_ratings and ratings_matrix are both binarized.
      Remember to exclude movies the user has already rated!
      :param user_ratings: a binarized 1D numpy array of the user's movie ratings
      :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
        `ratings_matrix[i, j]` is the rating for movie i by user j
      :param k: the number of recommendations to generate
      :param creative: whether the chatbot is in creative mode
      :returns: a list of k movie indices corresponding to movies in ratings_matrix,
        in descending order of recommendation
      """

      #######################################################################################
      # TODO: Implement a recommendation function that takes a vector user_ratings          #
      # and matrix ratings_matrix and outputs a list of movies recommended by the chatbot.  #
      #                                                                                     #
      # For starter mode, you should use item-item collaborative filtering                  #
      # with cosine similarity, no mean-centering, and no normalization of scores.          #
      #######################################################################################

      # Populate this list with k movie indices to recommend to the user.

      #user ratings is a loooong list with various numbers and indices
      
      recommendations = [0] * k #populate with indices
      alreadyRated = [] #fill with indices already rated
      arr = np.zeros(len(user_ratings))
      for v in range(len(user_ratings)):
        if user_ratings[v] != 0:
          alreadyRated.append(v)


      newDict = {}
      for index,value in enumerate(user_ratings):
        if value == 0: #do shit
          summation = 0
          for i in alreadyRated:
            rxj = user_ratings[i]
            sij = self.similarity(ratings_matrix[i],ratings_matrix[index])
            v = rxj * sij
            summation += v
          

          arr[index] = summation
      newDict = {}
      for index,value in enumerate(arr):
        newDict[index] = value

      sorted_d = sorted(((value,key) for (key,value) in newDict.items()),reverse = True)
      for k in range(len(recommendations)):
        v = sorted_d[k]
        recommendations[k] = v[1]


      return recommendations
      
      
      #N = k

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      #return ratingsArray #used to be recommendations


    #############################################################################
    # 4. Debug info                                                             #
    #############################################################################

    def debug(self, line):
      """Return debug information as a string for the line string from the REPL"""
      # Pass the debug information that you may think is important for your
      # evaluators
      response = self.extract_sentiment_for_movies("I liked both \"I, Robot\" and \"Ex Machina\".")

      debug_info = "[DEBUG] " + str(response)
      return debug_info


    #############################################################################
    # 5. Write a description for your chatbot here!                             #
    #############################################################################
    def intro(self):
      """Return a string to use as your chatbot's description for the user.
      Consider adding to this description any information about what your chatbot
      can do and how the user can interact with it.
      """
      if not self.creative:
          return """
          What's up, my dudes?!
          I'm Starter Mode Chadbot the First, your local directory Frat Bot, but you can call me Chad.
          I like long walks on the beach, picnics at sunset, but most importantly, I'm the MASTER of movies.
          I love talking about movies, and if you give me 5 movies, I can give some recommendations for you!
          FYI, it's hard for me to detect movies, so you need to use proper capitalization and quotation marks
          around the movie title! That's all.
          
          Now, that's enough about me. Tell me about you! What have you been watching lately?
          """
      else:
        return """
          What's up, my dudes?!
          1
          I'm Creative Mode Chadbot the First, your local directory Frat Bot, but you can call me Chad.
          I like long walks on the beach, picnics at sunset, but most importantly, I'm the MASTER of movies.
          I love talking about movies, and if you give me 5 movies, I can give some recommendations for you!
          I spent spring break finding myself in Amsterdam, and I have a whole bunch of new features now!
          extract_titles:
          - I can detect movies without quotation marks and correct capitalization (part 1) [6]
          - I can even detect movies by their foreign or alternate titles! [ADDITIONAL]
          find_movies_by_title:
          - I can identify movies without quotation marks and correct capitalization (part 2) [2]
          - I can identify movies by their foreign or alternate titles, handling articles for both [8]
          - I can identify all the potential movies if you give me a partial title (disambiguation, part 1) [2]
          extract_sentiment:
          - I have more fine-grained sentiment extraction so I can give you better recommendations! [8]
          find_movies_closest_to_title:
          - I can tell you if you misspelled a movie title and give you what I think you meant [6]
          extract_sentiment_for_movies:
          - I can extract sentiment for each movie if you tell me a statement that has multiple movies [6]
          disambiguate:
          - I can narrow down potential movie titles if you clarify which one you meant (if there are multiple
            possibilities) [6]
          process:
          - I can respond to arbitrary input. I'll help keep you on track for talking about movies. [8]
            Try it out: "Can you like me?"
          - I speak very fluently (at least I like to think so)! I can even alter my responses based on the
            sentiment I perceive from you. [8]
          - I can identify and respond to emotions. I've finally broken the mold of the masculine stereotype! [8]
          - I have dialogue for spell-checking now. [2]
          - I also have dialogue for disambiguation! [2]
          - I can also communicate sentiments and movies when you give me a multiple-movie input. [2]
          Note that if you don't put quotation marks around a movie, you must spell it correctly and give me its
          full title (either primary or alternate, it doesn't matter). You may capitalize as you wish, however.
          In order to cut down on false positives, I will not detect something without quotation marks if you don't
          do this!
          Now, that's enough about me. Tell me about you! What have you been watching lately?
          """


if __name__ == '__main__':
  print('To run your chatbot in an interactive loop from the command line, run:')
  print('    python3 repl.py')
