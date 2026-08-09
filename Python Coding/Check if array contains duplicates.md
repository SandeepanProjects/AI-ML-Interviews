# Python Program check if there are any duplicates  
# in the array using Hashing

def checkDuplicates(arr):
    n = len(arr)

    # Create a set to store the unique elements
    st = set()

    # Iterate through each element
    for i in range(n):
        
        # If the element is already present, return true
        # Else insert the element into the set
        if arr[i] in st:
            return True
        else:
            st.add(arr[i])

    # If no duplicates are found, return false
    return False

if __name__ == "__main__":
    arr = [4, 5, 6, 4]
    print(checkDuplicates(arr))

By using HashSet Data Structure – O(n) Time and O(n) Space
The main idea is to insert each value into a HashSet, which only stores unique elements. If any insertion fails or if any elements are already exits in HashSet, it means a duplicate exists, so return true. If all insertions succeed, it indicates that all elements are unique, so return false.