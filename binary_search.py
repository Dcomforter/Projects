def binarysearch(array, target, low, high):
    while low <= high:
        mid = (high + low) // 2
        if array[mid] == target:
            return mid
        elif array[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

nums = [85, 93, 12, 11, 7, 2, 22, 28, 9, 10, 22, 2017, 2022, 2, 2, 7, 12]
targ = 2017

nums.sort()
print(nums)
print(len(nums) - 1)

result = binarysearch(nums, targ, 0, len(nums) - 1)
if result == -1:
    print(f'{targ} is not found in the array')
else:
    print(targ, 'is found at index ' + str(result))
