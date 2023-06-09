import random
import string

from hangman_words import words

def get_valid_word(word):
    word = random.choice(words) # randomly chooses from the list
    while '-' in word or ' ' in word:
        word = random.choice(words)

    return word.upper()

def hangman():
    word = get_valid_word(words)
    word_letters = set(word.upper()) # letters in the word
    alphabet = set(string.ascii_uppercase) # set of uppercase letters in English dictionary
    used_letters = set() # empty set of guessed words

    lives = 6
    # getting user input
    while len(word_letters) > 0 and lives > 0:
        # print used letters
        # ' '.join(['a', 'b', 'cd']) --> 'a b cd'
        print('You have', lives, 'lives left and you have used these letters: ', ' '.join(used_letters))

        # what current word is (ie W - R D)
        word_list = [letter if letter in used_letters else '-' for letter in word]
        print('Current word: ', ' '.join(word_list))
        #print(word_letters)

        guessed_letter = input('\nGuess a letter: ').upper()
        print()
        if guessed_letter in alphabet - used_letters:
            used_letters.add(guessed_letter)
            if guessed_letter in word_letters:
                word_letters.remove(guessed_letter)
            else:
                lives = lives - 1 # takes away a life if wrong
                print('\nYour letter,', guessed_letter, 'is not in the word')
        elif guessed_letter in used_letters:
            print('You have already used that character. Please try again')
        else:
            print('Invalid character. Please try again')
    if lives == 0:
        print('Sorry you died. The word was', word)
    else:
        print('You guessed the word', word,'!!')

hangman()




