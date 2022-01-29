# just trying to minimize what all it offers
# reference python standard logging module snippets

CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0

_levelToName = {
    CRITICAL: 'CRITICAL',
    ERROR: 'ERROR',
    WARNING: 'WARNING',
    INFO: 'INFO',
    DEBUG: 'DEBUG',
    NOTSET: 'NOTSET',
}

_nameToLevel = {
    'CRITICAL': CRITICAL,
    'FATAL': FATAL,
    'ERROR': ERROR,
    'WARN': WARNING,
    'WARNING': WARNING,
    'INFO': INFO,
    'DEBUG': DEBUG,
    'NOTSET': NOTSET,
}

def getLevelName(level):
    result = _levelToName.get(level)
    if result is not None:
        return result
    result = _nameToLevel.get(level)
    if result is not None:
        return result
    return "Level %s" % level

def _checkLevel(level):
    if isinstance(level, int):
        rv = level
    elif str(level) == level:
        if level not in _nameToLevel:
            raise ValueError("Unknown level: %r" % level)
        rv = _nameToLevel[level]
    else:
        raise TypeError("Level not an integer or a valid string: %r" % level)
    return rv


class my_logger:
    def __init__(self, logger_name, logger_level):
        
        # write to console and a text file is the req...
        # error > warning > info > debug...

        self.logger_name = logger_name
        self.logger_level = _checkLevel(logger_level)
        self._cache = {}
    
    def info(self, msg, *args, **kwargs): 
        # check if we can allow this and then allow it
        if self.isEnabledFor("INFO"):
            self.log_me(log_level="INFO", log_msg=msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        if self.isEnabledFor("WARN"):
            self.log_me(log_level="WARN", log_msg=msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        if self.isEnabledFor("ERROR"):
            self.log_me(log_level="ERROR", log_msg=msg, *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        if self.isEnabledFor("DEBUG"):
            self.log_me(log_level="DEBUG", log_msg=msg, *args, **kwargs)

    def handler(self, msg_level, msg):
        msg_level = msg_level
        pass

    def isEnabledFor(self, level):
        # is the logger enabled for this level?
        try:
            return self._cache[level]
        except KeyError:
            try:
                is_enabled = self._cache[level] = (level >= self.getEffectiveLevel())
            finally:
                pass
            return is_enabled
        
    def setLevel(self, level):
        self.level = _checkLevel(level)
    
    def log_me(self, log_level, log_msg, *args, **kwargs):
        # we need to handle handlers here as well..
        # create a log_record (prepare the message here)
        # 
        pass

    def getEffectiveLevel(self):
        # get the effective level for this logger...
        logger = self
        while logger:
            if logger.level:
                return logger.level
            logger = logger.parent
        return NOTSET
    
    def __repr__(self):
        level = getLevelName(self.getEffectiveLevel())
        return '<%s %s (%s)>' % (self.__class__.__name__, self.name, level)
