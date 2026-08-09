def maxProfit(prices):
    minSoFar = prices[0]
    res = 0

    
    for i in range(1, len(prices)):
        
        # Update the minimum value seen so far  
        minSoFar = min(minSoFar, prices[i])
        
        # Update result if we get more profit                
        res = max(res, prices[i] - minSoFar)
    
    return res

if __name__ == "__main__":
    prices = [7, 10, 1, 3, 6, 9, 2]
    print(maxProfit(prices))

One Traversal Solution - O(n) Time and O(1) Space
In order to maximize the profit, we need to minimize the cost price and maximize the selling price. So at every step, we keep track of the minimum buy price of stock encountered so far. For every price, we subtract with the minimum so far and if we get more profit than the current result, we update the result.