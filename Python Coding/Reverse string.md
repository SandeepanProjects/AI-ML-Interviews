def reverseString(s):
    left = 0
    right = len(s) - 1
    
    # Convert string to a list for mutability
    s = list(s)  
    
    # Swap characters from both ends till we reach
    # the middle of the string
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1
    
    # Convert list back to string
    return ''.join(s)  

if __name__ == "__main__":
    s = "abdcfe"
    print(reverseString(s))