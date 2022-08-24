# LC Question -: Shortest Path in a Grid with Obstacles Elimination
# https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/

# 4D DP + DFS

def build_multiD(sizes, initial=None): 
    r"""
    return an N-dimensional list 
    :param sizes - an N - tuple giving the size of each dimension in turn.       
    :param initial - the initial value to be placed into the elements in the list 
    """ 
    if len(sizes) == 1: 
        return [initial] * sizes[0] 
    else: 
        return [build_multiD(sizes[1:], initial) for _ in range(sizes[0])] 


class Solution:
    def shortestPath(self, grid: List[List[int]], k: int) -> int:
        dxy = ((1,0), (-1,0), (0,1), (0,-1))
        
        n = len(grid)
        m = len(grid[0])
        
        dp = build_multiD((k+1, n, m, 4), None)
        visited = [[None]*m for _ in range(n)]
        
        def dfs(i, j, k, dir):
            if i == n - 1 and j == m - 1: 
                return 0
            
            if k == -1: 
                return float("inf")
            
            if dp[k][i][j][dir] != None: 
                return dp[k][i][j][dir]
            
            visited[i][j] = True
            ans = float("inf")

            for i1 in range(4):
                nx = dxy[i1][0] + i
                ny = dxy[i1][1] + j
                
                if (nx < 0 or ny < 0 or nx >= n or ny >= m or visited[nx][ny]):
                    continue

                res = dfs(nx, ny, k - grid[nx][ny], i1)

                if res != float("inf"):
                    ans = min(res+1, ans)
                
            visited[i][j] = False
            dp[k][i][j][dir] = ans
            return dp[k][i][j][dir]
        
        res = dfs(0, 0, k, 0)
        return res if res != float("inf") else -1
