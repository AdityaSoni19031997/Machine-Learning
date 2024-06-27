# circuit_breaker.py

"""
# Code Reference: https://bhaveshpraveen.medium.com/implementing-circuit-breaker-pattern-from-scratch-in-python-714100cdf90b


In real world applications, services might go down and start back up (or they might just stay down). 
The idea is that when you make a remote call(HTTP Request/RPC) to another service, there are chances that the remote call might fail. 
After a certain number of failed remote calls, we stop making remote calls and send a cached response or an error as a response. 
After a specified delay, we allow one remote call to be made to the failing server, 
  if it succeeds, we allow the subsequent remote calls to be made to the server, 
  if it did not succeed, we will continue sending a cached response or an error and 
    will not make any remote calls to the failing service for some time.

When all services were working and the remote calls were returning without any errors, we call this state — “Closed”.
When the remote calls continued to fail and when we stopped making any more remote calls to the failing service, we call this state — “Open”

After a certain delay, when we make a remote call to the failing service, the state transitions from “Open” to “Half-Open”. 
If the remote call does not fail, then we transition the state from “Half Open” to “Closed” and the subsequent remote calls are allowed to be made. 
In case the remote call failed, we transition the state from “Half Open”, back to “Open” state and we wait for a certain period of time 
till we can make the next remote call (in Half Open state).

The below code is copied over the medium blog post. My ~20 mintues attempt is right below commented out.
"""

import functools
import http
import logging
from datetime import datetime

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,%(msecs)d %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


class StateChoices:
    OPEN = "open"
    CLOSED = "closed"
    HALF_OPEN = "half_open"


class RemoteCallFailedException(Exception):
    pass


class CircuitBreaker:
    def __init__(self, func, exceptions, threshold, delay):
        """
        :param func: method that makes the remote call
        :param exceptions: an exception or a tuple of exceptions to catch (ideally should be network exceptions)
        :param threshold: number of failed attempts before the state is changed to "Open"
        :param delay: delay in seconds between "Closed" and "Half-Open" state
        """
        self.func = func
        self.exceptions_to_catch = exceptions
        self.threshold = threshold
        self.delay = delay

        # by default set the state to closed
        self.state = StateChoices.CLOSED


        self.last_attempt_timestamp = None
        # keep track of failed attemp count
        self._failed_attempt_count = 0

    def update_last_attempt_timestamp(self):
        self.last_attempt_timestamp = datetime.utcnow().timestamp()

    def set_state(self, state):
        prev_state = self.state
        self.state = state
        logging.info(f"Changed state from {prev_state} to {self.state}")

    def handle_closed_state(self, *args, **kwargs):
        allowed_exceptions = self.exceptions_to_catch
        try:
            ret_val = self.func(*args, **kwargs)
            logging.info("Success: Remote call")
            self.update_last_attempt_timestamp()
            return ret_val
        except allowed_exceptions as e:
            # remote call has failed
            logging.info("Failure: Remote call")
            # increment the failed attempt count
            self._failed_attempt_count += 1

            # update last_attempt_timestamp
            self.update_last_attempt_timestamp()

            # if the failed attempt count is more than the threshold
            # then change the state to OPEN
            if self._failed_attempt_count >= self.threshold:
                self.set_state(StateChoices.OPEN)
            # re-raise the exception
            raise RemoteCallFailedException from e

    def handle_open_state(self, *args, **kwargs):
        current_timestamp = datetime.utcnow().timestamp()
        # if `delay` seconds have not elapsed since the last attempt, raise an exception
        if self.last_attempt_timestamp + self.delay >= current_timestamp:
            raise RemoteCallFailedException(f"Retry after {self.last_attempt_timestamp+self.delay-current_timestamp} secs")

        # after `delay` seconds have elapsed since the last attempt, try making the remote call
        # update the state to half open state
        self.set_state(StateChoices.HALF_OPEN)
        allowed_exceptions = self.exceptions_to_catch
        try:
            ret_val = self.func(*args, **kwargs)
            # the remote call was successful
            # now reset the state to Closed
            self.set_state(StateChoices.CLOSED)
            # reset the failed attempt counter
            self._failed_attempt_count = 0
            # update the last_attempt_timestamp
            self.update_last_attempt_timestamp()
            # return the remote call's response
            return ret_val
        except allowed_exceptions as e:
            # the remote call failed again
            # increment the failed attempt count
            self._failed_attempt_count += 1

            # update last_attempt_timestamp
            self.update_last_attempt_timestamp()

            # set the state to "OPEN"
            self.set_state(StateChoices.OPEN)

            # raise the error
            raise RemoteCallFailedException from e

    def make_remote_call(self, *args, **kwargs):
        if self.state == StateChoices.CLOSED:
            return self.handle_closed_state(*args, **kwargs)
        if self.state == StateChoices.OPEN:
            return self.handle_open_state(*args, **kwargs)


"""
# My Attempt: It's pretty simple in terms of design and functionality.

import time

class CircuitBreakerOpenException(Exception):
    pass


class CircuitBreaker:
    def __init__(self, max_failures=3, cooldown_time=10):
        self.max_failures = max_failures
        self.cooldown_time = cooldown_time
        self.failure_count = 0
        self.last_failure_time = None
        self.is_closed = True
        
    def execute(self, func, *args, **kwargs):
        if not self.is_closed:
            if self.should_try_again():
                return self.execute(func, *args, **kwargs)
            else:
                raise CircuitBreakerOpenException("Circuit breaker is open")
        try:
            result = func(*args, **kwargs)
            self.reset()
            return result
        except Exception as e:
            self.record_failure()
            raise e
    
    def record_failure(self):
        self.failure_count += 1
        
        if self.failure_count >= self.max_failures:
            self.last_failure_time = time.monotonic()
            self.is_closed = False
            print("Circuit breaker is open")
            
    def should_try_again(self):
        if self.last_failure_time is None:
            return False
        
        elapsed_time = time.monotonic() - self.last_failure_time
        if elapsed_time >= self.cooldown_time:
            self.is_closed = True
            return True
        else:
            return False
        
    def reset(self):
        self.failure_count = 0
        self.last_failure_time = None
        self.is_closed = True


# Driver Code To Execute The Above
def my_function():
    print("Executing my function")
    raise ValueError("Something went wrong")

breaker = CircuitBreaker(
	max_failures=2, cooldown_time=5
)

for i in range(5):
    try:
        result = breaker.execute(my_function)
        print("Result:", result)
    except CircuitBreakerOpenException:
        print("Circuit breaker is open")
    except Exception as e:
        print("Error:", e)
    time.sleep(2)
"""

"""
Output:

Executing my function
Error: Something went wrong
Executing my function
Circuit breaker is open
Error: Something went wrong
Circuit breaker is open
Circuit breaker is open
Executing my function
Circuit breaker is open
Error: Something went wrong
"""
