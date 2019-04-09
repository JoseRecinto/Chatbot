#!/usr/bin/env python

# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
#
# Usage:
#   python sanity_check.py --recommender
#   python sanity_check.py --binarize
######################################################################
from chatbot import Chatbot


import argparse
import numpy as np
import math


def assertNumpyArrayEquals(givenValue, correctValue, failureMessage):
    try:
        assert np.array_equal(givenValue, correctValue)
        return True
    except Exception:
        print(failureMessage)
        print("Expected: {}".format(correctValue))
        print("Actual: {}".format(givenValue))
        return False

def assertListEquals(givenValue, correctValue, failureMessage, orderMatters=True):
    try:
        if orderMatters:
            assert givenValue == correctValue
            return True
        givenValueSet = set(givenValue)
        correctValueSet = set(correctValue)
        assert givenValueSet == correctValueSet
        return True
    except Exception:
        print(failureMessage)
        print("Expected: {}".format(correctValue))
        print("Actual: {}".format(givenValue))
        return False

def assertEquals(givenValue, correctValue, failureMessage):
    try:
        assert givenValue == correctValue
        return True
    except Exception:
        print(failureMessage)
        print("Expected: {}".format(correctValue))
        print("Actual: {}".format(givenValue))
        return False

def test_similarity():
    print("Testing similarity() functionality...")
    chatbot = Chatbot(False)

    x = np.array([1, 1, -1, 0], dtype=float)
    y = np.array([1, 0, 1, -1], dtype=float)

    self_similarity = chatbot.similarity(x, x)
    if not math.isclose(self_similarity, 1.0):
        print('Unexpected cosine similarity between {} and itself'.format(x))
        print('Expected 1.0, calculated {}'.format(self_similarity))
        print()
        return False

    ortho_similarity = chatbot.similarity(x, y)
    if not math.isclose(ortho_similarity, 0.0):
        print('Unexpected cosine similarity between {} and {}'.format(x, y))
        print('Expected 0.0, calculated {}'.format(ortho_similarity))
        print()
        return False

    print('similarity() sanity check passed!')
    print()
    return True

def test_binarize():
    print("Testing binarize() functionality...")
    chatbot = Chatbot(False)
    if assertNumpyArrayEquals(
        chatbot.binarize(np.array([[1, 2.5, 5, 0]])),
        np.array([[-1., -1., 1., 0.]]),
        "Incorrect output for binarize(np.array([[1, 2.5, 5, 0]]))."
    ):
        print("binarize() sanity check passed!")
    print()

def test_extract_titles():
    print("Testing extract_titles() functionality...")
    chatbot = Chatbot(False)
    if assertListEquals(
        chatbot.extract_titles('I liked "The Notebook"'),
        ["The Notebook"],
        "Incorrect output for extract_titles(\'I liked \"The Notebook\"\')."
    ) and assertListEquals(
        chatbot.extract_titles('No movies here!'),
        [],
        "Incorrect output for extract_titles('No movies here!').",
    ) and assertListEquals(
        chatbot.extract_titles('I enjoyed "Titanic (1997)" and "Scream 2 (1997)"'),
        ["Titanic (1997)", "Scream 2 (1997)"],
        "Incorrect output for extract_titles('I enjoyed \"Titanic (1997)\" and \"Scream 2 (1997)\"')."
    ):
        print('extract_titles() sanity check passed!')
    print()

def test_extract_titles_creative():
    print("Testing [CREATIVE MODE] extract_titles() functionality...")
    chatbot = Chatbot(True)
    if assertListEquals(
        chatbot.extract_titles('I liked The Notebook!'),
        ["The Notebook"],
        "Incorrect output for [CREATIVE] extract_titles('I liked The Notebook')."
    ) and assertListEquals (
        chatbot.extract_titles('I thought 10 things i hate about you was great'),
        ["10 things i hate about you", "10", "hate"],
        "Incorrect output for [CREATIVE] extract_titles('I thought 10 things i hate about you was great').",
        orderMatters=False
    ) and assertListEquals(
        chatbot.extract_titles('I like \"Harry Potter\"'),
        ["Harry Potter"],
        "Incorrect output for [CREATIVE] extract_titles('I like \"Harry Potter\"')."
    ) and assertListEquals(
        chatbot.extract_titles('I liked Scream 2!'),
        ["Scream","Scream 2"],
        "Incorrect output for [CREATIVE] extract_titles('I liked Scream 2').",
        orderMatters=False
    ) and assertListEquals(
        chatbot.extract_titles('I liked "Titanic" and 10 things i hate about you...'),
        ["Titanic", "10 things i hate about you", "10", "hate"],
        "Incorrect output for [CREATIVE] extract_titles('I liked \"Titanic\" and 10 things i hate about you').",
        orderMatters=False
    ) and assertListEquals(
        chatbot.extract_titles('I liked Se7en, 10 things i hate about you, La guerre du feu, The Return of Godzilla, and I, Robot!'),
        ["10", "hate", "Se7en", "10 things i hate about you", "La guerre du feu", "I, Robot", "The Return of Godzilla", "The Return", "Godzilla"],
        "Incorrect output for [CREATIVE] extract_titles('I liked Se7en, 10 things i hate about you, La guerre du feu, The Return of Godzilla, and I, Robot!').",
        orderMatters=False
    ) and assertListEquals(
        chatbot.extract_titles('I liked "Home Alone 3"'),
        ["Home Alone 3"],
        "Incorrect output for [CREATIVE] extract_titles('I liked \"Home Alone 3\").",
        orderMatters=False
    ) and assertListEquals(
        chatbot.extract_titles('I liked happy feet'),
        ["happy", "happy feet"],
        "Incorrect output for [CREATIVE] extract_titles('I liked happy feet).",
        orderMatters=False
    ) and assertListEquals(
        chatbot.extract_titles('I liked \"Happy Feet\"'),
        ["Happy Feet"],
        "Incorrect output for [CREATIVE] extract_titles('I liked \"Happy Feet\").",
        orderMatters=False
    ):
        print('[CREATIVE MODE] extract_titles() sanity check passed!')
    print()

def test_find_movies_by_title():
    print("Testing find_movies_by_title() functionality...")
    chatbot = Chatbot(False)
    if assertListEquals(
        chatbot.find_movies_by_title("The American President"),
        [10],
        "Incorrect output for find_movies_by_title('The American President')."
    ) and assertListEquals(
        chatbot.find_movies_by_title("An American in Paris (1951)"),
        [721],
        "Incorrect output for find_movies_by_title('An American in Paris (1951)')."
    ) and assertListEquals(
        chatbot.find_movies_by_title("Titanic"),
        [1359, 2716],
        "Incorrect output for find_movies_by_title('Titanic').",
        orderMatters=False
    ) and assertListEquals(
        chatbot.find_movies_by_title("The Notebook"),
        [5448],
        "Incorrect output for find_movies_by_title('The Notebook')."
    ) and assertListEquals(
        chatbot.find_movies_by_title("Titanic (1997)"),
        [1359],
        "Incorrect output for find_movies_by_title('Titanic (1997)').",
    ) and assertListEquals(
        chatbot.find_movies_by_title("There is no title here!"),
        [],
        "Incorrect output for find_movies_by_title('There is no title here!')."
    ):
        print('find_movies_by_title() sanity check passed!')
    print()

def test_find_movies_by_title_creative():
    print("Testing [CREATIVE MODE] find_movies_by_title() functionality...")
    chatbot = Chatbot(True)
    if assertListEquals(
        chatbot.find_movies_by_title("10 things i HATE about you"),
        [2063],
        "Incorrect output for [CREATIVE] find_movies_by_title('10 things i HATE about you')."
    ) and assertListEquals(
        chatbot.find_movies_by_title("There is no title here!"),
        [],
        "Incorrect output for [CREATIVE] find_movies_by_title('There is no title here!')."
    ) and assertListEquals(
        chatbot.find_movies_by_title("Se7en"),
        [45],
        "Incorrect output for [CREATIVE] find_movies_by_title('Se7en')."
    ) and assertListEquals(
        chatbot.find_movies_by_title("La guerre du feu"),
        [2439],
        "Incorrect output for [CREATIVE] find_movies_by_title('La guerre du feu')."
    ) and assertListEquals(
        chatbot.find_movies_by_title("The Tragedy of Othello: the moor of Venice"),
        [2279],
        "Incorrect output for [CREATIVE] find_movies_by_title('Tragedy of Othello: The Moor of Venice, The')."
    ) and assertListEquals(
        chatbot.find_movies_by_title("Scream"),
        [546, 2629, 1357, 1142],
        "Incorrect output for [CREATIVE] find_movies_by_title('Scream').",
        orderMatters=False
    ) and assertListEquals(
        chatbot.find_movies_by_title("Percy Jackson"),
        [8377, 7463],
        "Incorrect output for [CREATIVE] find_movies_by_title('Percy Jackson').",
        orderMatters=False
    ) and assertListEquals(
        chatbot.find_movies_by_title("Harry potter"),
        [3812, 4325, 5399, 6294, 6735, 7274, 7670, 7842],
        "Incorrect output for [CREATIVE] find_movies_by_title('Harry potter').",
        orderMatters=False
    ):
        print('[CREATIVE MODE] find_movies_by_title sanity check passed!')
    print()

def test_extract_sentiment():
    print("Testing extract_sentiment() functionality...")
    chatbot = Chatbot(False)
    if assertEquals(
        chatbot.extract_sentiment("I like \"Titanic (1997)\"."),
        1,
        "Incorrect output for extract_sentiment(\'I like \"Titanic (1997)\".\')"
    ) and assertEquals(
        chatbot.extract_sentiment("I saw \"Titanic (1997)\"."),
        0,
        "Incorrect output for extract_sentiment(\'I saw  \"Titanic (1997)\".\')"
    ) and assertEquals(
        chatbot.extract_sentiment("I didn't enjoy \"Titanic (1997)\"."),
        -1,
        "Incorrect output for extract_sentiment(\'I didn't enjoy  \"Titanic (1997)\"\'.)"
    ) and assertEquals(
        chatbot.extract_sentiment("I saw \"Titanic (1997)\"."),
        0,
        "Incorrect output for extract_sentiment(\'I saw  \"Titanic (1997)\"\'.)"
    ) and assertEquals(
        chatbot.extract_sentiment(" \"Titanic (1997)\" started out terrible, but the ending was totally great and I loved it!"),
        1,
        "Incorrect output for extract_sentiment(\" \"Titanic (1997)\" started out terrible, but the ending was totally great and I loved it!\'.\")"
    ) and assertEquals(
        chatbot.extract_sentiment("I loved \"10 Things I Hate About You\""),
        1,
        "Incorrect output for extract_sentiment(\'I loved  \"10 Things I Hate About You\"\')"
    ) and assertEquals(
        chatbot.extract_sentiment("I saw \"The Notebook\" and it was great!"),
        1,
        "Incorrect output for extract_sentiment(\'I saw \"The Notebook\" and it was great!\')"
    ):
        print('extract_sentiment() sanity check passed!')
    print()


def test_extract_sentiment_for_movies():
    print("Testing test_extract_sentiment_for_movies() functionality...")
    chatbot = Chatbot(True)
    if assertListEquals(
        chatbot.extract_sentiment_for_movies("I liked both \"I, Robot\" and \"Ex Machina\"."),
        [("I, Robot", 1), ("Ex Machina", 1)],
        "Incorrect output for test_extract_sentiment_for_movies(\"I liked both \"I, Robot\" and \"Ex Machina\".)\"",
        orderMatters=False
    ) and assertListEquals(
        chatbot.extract_sentiment_for_movies("I liked \"I, Robot\" but not \"Ex Machina\"."),
        [("I, Robot", 1), ("Ex Machina", -1)],
        "Incorrect output for test_extract_sentiment_for_movies(\"I liked \"I, Robot\" but not \"Ex Machina\".)\"",
        orderMatters=False
    ):
        print('extract_sentiment_for_movies() sanity check passed!')
    print()

def test_find_movies_closest_to_title():
    print("Testing find_movies_closest_to_title() functionality...")
    chatbot = Chatbot(True)

    misspelled = "Sleeping Beaty"

    if assertListEquals(
        chatbot.find_movies_closest_to_title(misspelled, max_distance=3),
        [1656],
        "Incorrect output for test_find_movies_closest_to_title('{}', max_distance={})".format(misspelled, 3),
        orderMatters=False
    ) and assertListEquals(
        chatbot.find_movies_closest_to_title("The Notbook", max_distance=3),
        [5448],
        "Incorrect output for test_find_movies_closest_to_title('{}', max_distance={})".format("The Notbook", 3),
        orderMatters=False
    ) and assertListEquals(
        chatbot.find_movies_closest_to_title("Te Notbook", max_distance=2),
        [5448],
        "Incorrect output for test_find_movies_closest_to_title('{}', max_distance={})".format("Te Notbook", 3),
        orderMatters=False
    ) and assertListEquals(
        chatbot.find_movies_closest_to_title("Te Notbook", max_distance=1),
        [],
        "Incorrect output for test_find_movies_closest_to_title('{}', max_distance={})".format("Te Notbook", 3),
        orderMatters=False
    ) and assertListEquals(
        chatbot.find_movies_closest_to_title("Te", max_distance=3),
        [8082, 4511, 1664],
        "Incorrect output for test_find_movies_closest_to_title('{}', max_distance={})".format("Te", 3),
        orderMatters=False
    ) and assertListEquals(
        chatbot.find_movies_closest_to_title("BAT-MAAAN", max_distance = 3),
        [524, 5743],
        "Incorrect output for test_find_movies_closest_to_title('{}', max_distance={})".format("BAT-MAAAN", 3),
        orderMatters=False
    ) and assertListEquals(
        chatbot.find_movies_closest_to_title("Blargdeblargh", max_distance = 4),
        [],
        "Incorrect output for test_find_movies_closest_to_title('{}', max_distance={})".format("Blargdeblargh", 4),
        orderMatters=False
    ):
        print('find_movies_closest_to_title() sanity check passed!')
    print()
    return True

def test_disambiguate():
    print("Testing disambiguate() functionality...")
    chatbot = Chatbot(True)

    clarification = "1997"
    candidates = [1359, 2716]
    if assertListEquals(
        chatbot.disambiguate(clarification, candidates),
        [1359],
        "Incorrect output for disambiguate('{}', {})".format(clarification, candidates),
        orderMatters=False
    ) and assertListEquals(
        chatbot.disambiguate("2", [1142, 1357, 2629, 546]),
        [1357],
        "Incorrect output for disambiguate('{}', {})".format("2", [1142, 1357, 2629, 546]),
        orderMatters=False
    ) and assertListEquals(
        chatbot.disambiguate("Sorcerer's Stone", [3812, 4325, 5399, 6294, 6735, 7274, 7670, 7842]),
        [3812],
        "Incorrect output for disambiguate('{}', {})".format("Sorcerer's Stone", [3812, 4325, 5399, 6294, 6735, 7274, 7670, 7842]),
        orderMatters=False
    ):
        print('disambiguate() sanity check passed!')
    print()
    return True

def test_recommend():
    print("Testing recommend() functionality...")
    chatbot = Chatbot(False)

    user_ratings = np.array([1, -1, 0, 0, 0, 0])
    all_ratings = np.array([
        [1, 1, 1, 0],
        [1, -1, 0, -1],
        [1, 1, 1, 0],
        [0, 1, 1, -1],
        [0, -1, 1, -1],
        [-1, -1, -1, 0],
    ])
    recommendations = chatbot.recommend(user_ratings, all_ratings, 2)

    if assertListEquals(recommendations, [2, 3], "Recommender test failed"):
        print("recommend() sanity check passed!")
    print()

def main():
    parser = argparse.ArgumentParser(description='Sanity checks the chatbot. If no arguments are passed, all checks are run; you can use the arguments below to test specific parts of the functionality.')

    parser.add_argument('-b', '--creative', help='Tests all of the creative function', action='store_true')

    args = parser.parse_args()
    testing_creative = args.creative

    test_extract_titles()
    test_find_movies_by_title()
    test_extract_sentiment()
    test_recommend()
    test_binarize()
    test_similarity() # TODO: broken when run with starter code

    if testing_creative:
        test_extract_titles_creative()
        test_find_movies_by_title_creative()
        test_find_movies_closest_to_title()
        test_extract_sentiment_for_movies()
        test_disambiguate()

if __name__ == '__main__':
    main()