def search(arr, key):
  
    # Initialize two pointers, lo and hi, at the start
    # and end of the array
    lo = 0
    hi = len(arr) - 1

    while lo <= hi:
        mid = lo + (hi - lo) // 2

        # If key found, return the index
        if arr[mid] == key:
            return mid

        # If Left half is sorted
        if arr[mid] >= arr[lo]:
          
            # If the key lies within this sorted half,
            # move the hi pointer to mid - 1
            if key >= arr[lo] and key < arr[mid]:
                hi = mid - 1
              
            # Otherwise, move the lo pointer to mid + 1
            else:
                lo = mid + 1
          
        # If Right half is sorted
        else:
          
            # If the key lies within this sorted half,
            # move the lo pointer to mid + 1
            if key > arr[mid] and key <= arr[hi]:
                lo = mid + 1
              
            # Otherwise, move the hi pointer to mid - 1
            else:
                hi = mid - 1

    # Key not found
    return -1

if __name__ == "__main__":
    arr = [5, 6, 7, 8, 9, 10, 1, 2, 3]
    key = 3
    print(search(arr, key))

Using Single Binary Search - O(log n) Time and O(1) Space
This approach applies a modified version of binary search directly to the entire rotated array. At every iteration, the middle element is checked against the key. If it’s not the key, we determine whether the left half or right half is sorted by comparing values at arr[lo] and arr[mid]. If the left half is sorted and the key lies within its range, we adjust hi = mid - 1; otherwise, we shift lo = mid + 1. If the right half is sorted and the key lies within its range, we move lo = mid + 1; else, hi = mid - 1.