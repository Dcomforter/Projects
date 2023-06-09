# This function implements a Caeser Cipher Algorithm

def rotate_chr(character):
    rotate_by = 3
    character = character.lower()
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    # Keep punctuation and whitespace
    if character not in alphabet:
        return character
    rotated_pos = ord(character) + rotate_by
    # If the rotation is inside the alphabet
    if rotated_pos <= ord(alphabet[-1]):
        return chr(rotated_pos)
    # If the rotation goes beyond the alphabet
    return chr(rotated_pos - len(alphabet))

print("".join(map(rotate_chr, "My business has just been nominat'd as the best hospitality startup.")))
print(ord('z') + 3)
print(chr((ord('z') + 3)-26))