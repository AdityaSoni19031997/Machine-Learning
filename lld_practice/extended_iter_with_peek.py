
class ExtendedIter:
    """An extended iterator that wraps around an existing iterators.
    It provides extra methods:
    - `has_next()`: checks if we can still yield items.
    - `peek()`: returns the next element of our iterator, but doesn't pass by it.
    If there's nothing more to return, raises `StopIteration` error.
    """
    
    def __init__(self, i):
        self._myiter = iter(i)
        self._next_element = None
        self._has_next = 0
        self._prime()


    def has_next(self):
        """Returns true if we can call next() without raising a
        StopException."""
        return self._has_next


    def peek(self):
        """Nonexhaustively returns the next element in our iterator."""
        assert self.has_next()
        return self._next_element


    def next(self):
        """Returns the next element in our iterator."""
        if not self._has_next:
            raise StopIteration
        result = self._next_element
        self._prime()
        return result


    def _prime(self):
        """Private function to initialize the states of
        self._next_element and self._has_next.  We poke our
        self._myiter to see if it's still alive and kicking."""
        try:
            self._next_element = next(self._myiter)
            self._has_next = 1
        except StopIteration:
            self.next_element = None
            self._has_next = 0
