import random

# User guesses the number
def guess(x):
    random_number = random.randint(1, x) # Randomly generated between 1 and the number entered by the user
    guess = 0
    while guess != random_number:
        guess = int(input(f"\nGuess a number between 1 and {x}: "))
        if guess < random_number:
            print(f"\nSorry, {guess} is too low. Guess again")
        elif guess > random_number:
            print(f"\nSorry, {guess} is too high. Guess again")

    #print(random_number)
    print(f"\nYAY!!!, {guess} is correct. Congratulations\n")

# Computer guesses the number
def computer_guess(x):
    low = 1
    high = x
    feedback = ''

    while feedback != 'c':
        if low != high:
            guess = random.randint(low, high)
        else:
            guess = low # could also be high b/c low = high
        feedback = input(f"\nGuessing the number on your mind. Is {guess} too high (H), too low (L), or correct (C): ")
        if feedback == 'h':
            print(f"\nOk, {guess} was too high. ")
            high = guess - 1
        elif feedback == 'l':
            print(f"\nOk, {guess} was too low. ")
            low = guess + 1

    print(f"\nYay!!! The computer guessed {guess} correctly!\n")

# This function makes you guess someone's age
def guess_age(x):
    guess = 0
    low = 1900

    while guess != 1958:
        guess = int(input(f'\nGuess the year I was born between {low} and {x}: '))
        if guess < 1958:
            print(f'\nToo low, I was not born in {guess}. Guess again ')
        elif guess > 1958:
            print(f'\nToo high, I was not born in {guess}. Guess again ')

    #print(guess)
    age = x - guess
    print(f'\nYay!!! You guessed {guess} correctly and my current age is {age}. \n')

#guess(10000)
#computer_guess(2023)
guess_age(2022)




