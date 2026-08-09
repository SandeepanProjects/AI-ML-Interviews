def productExceptSelf(arr):
    zeros = 0
    idx = -1
    prod = 1

    # Count zeros and track the index of the zero
    for i in range(len(arr)):
        if arr[i] == 0:
            zeros += 1
            idx = i
        else:
            prod *= arr[i]

    res = [0] * len(arr)

    # If no zeros, calculate the product for all elements
    if zeros == 0:
        for i in range(len(arr)):
            res[i] = prod // arr[i]
    # If one zero, set product only at the zero's index
    elif zeros == 1:
        res[idx] = prod

    return res


if __name__ == "__main__":
    arr = [10, 3, 5, 6, 2]
    res = productExceptSelf(arr)
    print(" ".join(map(str, res)))

Using Product Array - O(n) Time and O(1) Space
The idea is to handle two special cases of the input array: when it contains zero(s) and when it doesn't. If the array has no zeros, product of array at any index (excluding itself) can be calculated by dividing the total product of all elements by the current element. 

However, division by zero is undefined, so if there are zeros in the array, the logic changes.
If there is exactly one zero, the product for that index will be the product of all other non-zero elements, while the elements in rest of the indices will be zero.
If there are more than one zero, the product for all indices will be zero, since multiplying by zero results in zero.