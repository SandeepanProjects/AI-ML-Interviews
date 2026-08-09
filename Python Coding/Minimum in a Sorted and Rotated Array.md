def findMin(arr):
    low, high = 0, len(arr) - 1

    while low < high:

        # If current range is sorted, first element is minimum
        if arr[low] < arr[high]:
            return arr[low]

        mid = low + (high - low) // 2

        # Minimum lies in right half
        if arr[mid] > arr[high]:
            low = mid + 1
        # Minimum lies in left half (including mid)
        else:
            high = mid

    # low == high points to the minimum element
    return arr[low]

if __name__ == "__main__":
    arr = [5, 6, 1, 2, 3, 4]
    print(findMin(arr))

Binary Search - O(log n) Time and O(1) Space
The array is sorted and then rotated, so it consists of two sorted parts, with the minimum element located at the rotation point.
Using Binary Search, we can efficiently narrow down the part that contains the minimum by checking whether the current range is already sorted and by comparing the middle element with the last element.

Steps for Implementation:

If arr[low] < arr[high], the current range is already sorted, return arr[low].
Compute mid.
If arr[mid] > arr[high], the minimum lies to the right of mid, set low = mid + 1.
Otherwise, the minimum lies at mid or to its left, set high = mid.
When low == high, that index stores the minimum element.