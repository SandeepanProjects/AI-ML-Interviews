MAX_CHAR = 26

def longestUniqueSubstr(s):
    if len(s) == 0 or len(s) == 1:
        return len(s)

    res = 0
    vis = [False] * 26

    # left and right pointer of sliding window
    left = 0
    right = 0
    while right < len(s):

        # If character is repeated, move left pointer marking
        # visited characters as false until the repeating 
        # character is no longer part of the current window
        while vis[ord(s[right]) - ord('a')] == True:
            vis[ord(s[left]) - ord('a')] = False
            left += 1

        vis[ord(s[right]) - ord('a')] = True

        # The length of the current window (right - left + 1)
        # is calculated and answer is updated accordingly.
        res = max(res, (right - left + 1))
        right += 1

    return res

if __name__ == "__main__":
    s = "geeksforgeeks"
    print(longestUniqueSubstr(s))