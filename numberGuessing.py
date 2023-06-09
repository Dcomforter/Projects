import random

# User guesses the number
def guess(x):
    random_number = random.randint(1, x)
    guess = 0
    while guess != random_number:
        guess = int(input(f"Guess a number between 1 and {x}: "))
        if guess < random_number:
            print(f"Sorry, {guess} is too low. Guess again")
        elif guess > random_number:
            print(f"Sorry, {guess} is too high. Guess again")

    print(random_number)
    print(f"YAY!!!, {guess} is correct. Congratulations")

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
        feedback = input(f"Guessing the number on your mind. Is {guess} too high (H), too low (L), or correct (C): ")
        if feedback == 'h':
            print(f"Ok, {guess} was too high. ")
            high = guess - 1
        elif feedback == 'l':
            print(f"Ok, {guess} was too low. ")
            low = guess + 1

    print(f"Yay!! The computer guessed {guess} correctly!")

def guess_age(x):
    guess = 0
    low = 1980
    while guess != 1985:
        guess = int(input(f'Guess the year I was born between {low} and {x}: '))
        if guess < 1985:
            print(f'Too low, I was not born in {guess}. Guess again ')
        elif guess > 1985:
            print(f'Too high, I was not born in {guess}. Guess again ')

    #print(guess)
    age = x - guess
    print(f'Yay! You guessed {guess} correctly and my current age is {age}. ')

guess(10000)
#computer_guess(2023)
#guess_age(2022)




