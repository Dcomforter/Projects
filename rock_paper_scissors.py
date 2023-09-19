# Random helps to generate numbers or characters randomly from a range or list.
import random

player_win_count = 0
computer_win_count = 0
ties_count = 0
player_name = '' 

# This function gets the choice from the player and computer, then calls on is_win() to determine who wins.
def play():
    player_name
    players_choice = '' # This initializes the player's choice, and helps to get out of the while loop below.

    while players_choice not in ['r', 'p', 's']:
        players_choice = input("What's your choice? 'r' for rock, 'p' for paper, 's' for scissors: ")
        computers_choice = random.choice(['r', 'p', 's']) # The computer randomly selects a character from the list.
        
        if players_choice not in ['r', 'p', 's']:
            print("Please type a choice only between 'r', 's', or 'p'. Try again\n")

    global player_win_count
    global computer_win_count
    global ties_count

    # Declares the game as a tie
    if players_choice == computers_choice:
        print(f'{player_name} chose \'{players_choice}\' and Computer chose \'{computers_choice}\'.')
        ties_count += 1
        return "It's a tie!!!\n"

    # Declares the player as the winner
    if is_win(players_choice, computers_choice):
        print(f'{player_name} chose \'{players_choice}\' and Computer chose \'{computers_choice}\'.')
        player_win_count += 1
        return f'{player_name} won!!!\n'

    # Declares the computer as the winner
    print(f'{player_name} chose \'{players_choice}\' and Computer chose \'{computers_choice}\'.')
    computer_win_count += 1
    return 'Computer won!!!\n'

# This function declares the player the winner based on the rules of the game
# In this game, r > s, s > p, p > r
def is_win(players_choice, computers_choice):
    # return true if player wins
    if (players_choice == 'r' and computers_choice == 's') or (players_choice == 's' and computers_choice == 'p') \
        or (players_choice == 'p' and computers_choice == 'r'):
        return True

# The program starts from here
print('\nWelcome to The First Game --> Rock, Paper and Scissors.\n')
player_name = input("What is your name?: ")
print()
response = '' # This helps to get of the while loop below
counter = 0 # Helps to get out of the inner while loop below

# TODO: Implement this function to count the number of individual wins and print out the overall winner
# Determines how many times the game is played based on response or count
while response != 'no':
    counter += 1
    print("GAME NUMBER", counter, "\n")
    print(play())

    # The loop ends once counter is greater than or equal to the set number
    if counter >= 7:
        print('Thanks for playing!!!\n')
        break

    # Keeps asking the question while counter is less than the set number
    # Also keeps track of wrong and right choices   
    while counter < 7:
        response = input('Do you want to play again? Type yes or no: ').lower()
        print()
        if response == 'yes':
            print('Game on!!!\n')
            break
        elif response != 'no' and response != 'yes':
            print('Please type a choice only between \'yes\' or \'no\'. Try again.')
        if response == 'no':
            print("Thanks for playing!!!\n")
            break

# Returns the count of individual wins based on the games played
print(f"Overall wins for {player_name} =>", player_win_count)
print("Overall wins for Computer =>", computer_win_count)
print("Overall ties for both players =>", ties_count, "\n")

# Determines the overall winner based on the count of individual wins
if player_win_count > computer_win_count:
    print(f"{player_name} is the overall winner with {player_win_count} game win(s) out of {counter} games.\n")
elif player_win_count < computer_win_count:
    print(f"Computer is the overall winner with {computer_win_count} game win(s) out of {counter} games.\n")
else:
    print(f"The game was a tie overall, out of {counter} games\n")

print("! ! ! G A M E O V E R ! ! !\n")
