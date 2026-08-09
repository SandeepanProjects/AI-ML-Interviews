# Function to find the leaders in an array
def leaders(arr):
    result = []
    n = len(arr)

    # Start with the rightmost element
    maxRight = arr[-1]

    # Rightmost element is always a leader
    result.append(maxRight)

    # Traverse the array from right to left
    for i in range(n - 2, -1, -1):
        if arr[i] >= maxRight:
            maxRight = arr[i]
            result.append(maxRight)

    # Reverse the result list to maintain
    # original order
    result.reverse()

    return result

if __name__ == "__main__":
    arr = [16, 17, 4, 3, 5, 2]
    result = leaders(arr)
    print(" ".join(map(str, result)))



Using Suffix Maximum - O(n) Time and O(1) Space:
The idea is to scan all the elements from right to left in an array and keep track of the maximum till now. When the maximum changes its value, add it to the result. Finally reverse the result 