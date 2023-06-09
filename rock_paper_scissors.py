import random

player_win_count = 0
computer_win_count = 0
player_name = ''

# TODO: Implement this function to account for wrong choices.
def play():
    player_name
    player = ''

    while player not in ['r', 'p', 's']:
        player = input("What's your choice? 'r' for rock, 'p' for paper, 's' for scissors: ")
        computer = random.choice(['r', 'p', 's'])
        
        if player not in ['r', 'p', 's']:
            print("Please type a choice only between 'r', 's', or 'p'. Try again\n")

    global player_win_count
    global computer_win_count

    if player == computer:
        print(f'{player_name} chose \'{player}\' and Computer chose \'{computer}\'.')
        return 'It\'s a tie!!!\n'

# In this game, r > s, s > p, p > r
    if is_win(player, computer):
        print(f'{player_name} chose \'{player}\' and Computer chose \'{computer}\'.')
        player_win_count += 1
        return f'{player_name} won!!!\n'

    print(f'{player_name} chose \'{player}\' and Computer chose \'{computer}\'.')
    computer_win_count += 1
    return 'Computer won!!!\n'

def is_win(player, computer):
    # return true if player wins
    if (player == 'r' and computer == 's') or (player == 's' and computer == 'p') \
        or (player == 'p' and computer == 'r'):
        return True

print('\nWelcome to Game 1 --> Rock, Paper and Scissors.\n')
player_name = input("What is your name?: ")
print()
response = ''
counter = 0

# TODO: Implement this function to count the number of individual wins and print out the overall winner
while response != 'no':
    counter = counter + 1
    print("GAME NUMBER", counter, "\n")
    print(play())

    if counter >= 5:
        print('Thanks for playing!!!\n')
        break

    while counter < 5:
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

print(f"Overall wins for {player_name} =>", player_win_count)
print("Overall wins for Computer =>", computer_win_count, "\n")

if player_win_count > computer_win_count:
    print(f"{player_name} is the overall winner with {player_win_count} game win(s).\n")
elif player_win_count < computer_win_count:
    print(f"Computer is the overall winner with {computer_win_count} game win(s).\n")
else:
    print("The game was a tie overall\n")

print("G A M E O V E R ! ! !\n")
