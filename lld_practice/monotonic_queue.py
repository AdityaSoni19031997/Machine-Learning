from collections import deque
from typing import Literal

"""
==================================================================================================
                        MONOTONIC QUEUE: FORWARD VS. BACKWARD WINDOWS
==================================================================================================

+-----------------------+---------------------------------------------+----------------------------------+
| Feature / Aspect      | Forward Window (Left-to-Right)              | Backward Window (Right-to-Left)  |
+-----------------------+---------------------------------------------+----------------------------------+
| Typical Problem Types | Sliding Window Max/Min, Forward DP Jumps    | Reverse DP                       |
|                       | (e.g., Jump Game VI, LeetCode 239)          | (e.g., Fruit Purchase - LC 2944) |
|                       |                                             |                                  |
| Iteration Direction   | i = 0 -> n - 1                              | i = n -> 1                       |
|                       |                                             |                                  |
| Target Window Range   | [i - k + 1, i]  or  [i - k, i - 1]          | [i + 1, 2*i + 1]                 |
|                       |                                             |                                  |
| Invalid Condition     | Stored index j < min_valid_idx              | Stored index j > max_valid_idx   |
|                       |                                             |                                  |
| Comparison Operator   | <                                           | >                                |
|                       |                                             |                                  |
| Eviction Method Name  | evict_below(min_valid_idx)                  | evict_above(max_valid_idx)       |
|                       |                                             |                                  |
| Loop Code Pattern     | while queue and queue[0][1] < min_valid:    | while queue and queue[0][1] > max_valid:
+-----------------------+---------------------------------------------+----------------------------------+
"""


class MonotonicQueue:
    """
    An index-aware Monotonic Queue supporting O(1) amortized pushing,
    min/max lookup, and eviction for both forward and backward sliding windows.
    """
    def __init__(self, mode: Literal['min', 'max'] = 'min'):
        self.queue = deque()  # Stores tuples: (value, index)
        self.mode = mode

    def push(self, val: int | float, idx: int) -> None:
        """Pushes (val, idx) while maintaining strict monotonicity."""
        if self.mode == 'min':
            while self.queue and self.queue[-1][0] >= val:
                self.queue.pop()
        else:  # 'max'
            while self.queue and self.queue[-1][0] <= val:
                self.queue.pop()
        self.queue.append((val, idx))

    def evict_below(self, min_valid_idx: int) -> None:
        """
        FORWARD WINDOW EVICTION:
        Removes stale elements where stored index < min_valid_idx (j < min_idx).
        Use for standard left-to-right sliding windows (e.g., [i - k + 1, i]).
        """
        while self.queue and self.queue[0][1] < min_valid_idx:
            self.queue.popleft()

    def evict_above(self, max_valid_idx: int) -> None:
        """
        BACKWARD WINDOW EVICTION:
        Removes stale elements where stored index > max_valid_idx (j > max_idx).
        Use for right-to-left reverse DP windows (e.g., [i + 1, 2i + 1]).
        """
        while self.queue and self.queue[0][1] > max_valid_idx:
            self.queue.popleft()

    def evict_out_of_bounds(
        self, 
        min_idx: int | float = float('-inf'), 
        max_idx: int | float = float('inf')
    ) -> None:
        """
        UNIFIED RANGE EVICTION:
        Removes elements falling outside the closed range [min_idx, max_idx].
        Useful when a window has both lower and upper boundary constraints.
        """
        while self.queue and (self.queue[0][1] < min_idx or self.queue[0][1] > max_idx):
            self.queue.popleft()

    def get_extreme(self) -> int | float | None:
        """Returns the current optimal min/max value in O(1) time."""
        return self.queue[0][0] if self.queue else None

    def __bool__(self) -> bool:
        """Allows checking if the queue contains active elements (e.g., `if mq:`)."""
        return bool(self.queue)

