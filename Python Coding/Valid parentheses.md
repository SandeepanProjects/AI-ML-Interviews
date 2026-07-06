def isBalanced(s):
    st = [] 
    for c in s:
        if c == '(' or c == '{' or c == '[':
            st.append(c)
            
        # Process closing brackets
        elif c == ')' or c == '}' or c == ']':
            
            # No opening bracket
            if not st: return False 
            top = st[-1]
            if ((c == ')' and top != '(') or
                (c == '}' and top != '{') or
                (c == ']' and top != '[')):
                return False
                
            # Pop matching opening bracket
            st.pop() 
            
    # Balanced if stack is empty
    return not st 

if __name__ == '__main__':
    s = "[()()]{}"
    print("true" if isBalanced(s) else "false")