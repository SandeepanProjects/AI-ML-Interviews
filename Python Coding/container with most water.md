def maxWater(arr):
    left = 0
    right = len(arr) - 1
    res = 0
    while left < right:
        
        # find the water stored in the container between 
        # arr[left] and arr[right]
        water = min(arr[left], arr[right]) * (right - left)
        res = max(res, water)
        
        if arr[left] < arr[right]:
            left += 1
        else:
            right -= 1
    
    return res

  
if __name__ == "__main__":
	arr = [2, 1, 8, 6, 4, 6, 5, 5]
	print(maxWater(arr))



Using Two Pointers - O(n) Time and O(1) Space
The idea is to maintain two pointers: left pointer at the beginning of the array and right pointer at the end of the array. These pointers act as the container's sides so we can calculate the amount of water stored between them using the formula: min(arr[left], arr[right]) * (right - left). 

After calculating the amount of water for the given sides, we can have three cases:

arr[left] < arr[right]: This means that we have calculated the water stored for the container of height arr[left], so increment left by 1.
arr[left] >= arr[right]: This means that we have calculated the water stored for the container of height arr[right], so decrement right by 1.
Repeat the above process till left is less than right keeping track of the maximum water stored as the result.